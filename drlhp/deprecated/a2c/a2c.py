import logging
import os.path as osp
import queue
import time

from copy import deepcopy
import cloudpickle
import easy_tf_log
import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf
import cv2

from drlhp.deprecated.a2c import logger
from drlhp.deprecated.a2c.utils import (discount_with_dones,
                                  find_trainable_variables, mse)
from drlhp.deprecated.a2c.common.math_util import explained_variance
from drlhp.deprecated.a2c.common.misc_util import set_global_seeds
from drlhp.pref_db import Segment


class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs,
                 nsteps,
                 nstack,
                 num_procs,
                 lr_scheduler,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 alpha=0.99,
                 epsilon=1e-5):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=num_procs,
            inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        train_batch = nenvs * nsteps
        # CHANGE: A2C has separate variables for n_batch_step and n_batch_train
        print("Create placeholders")

        #TODO check that this works for other action spaces

        # CHANGE: A2C hs tehse have shape None rather than shape train_batch
        A = tf.placeholder(tf.int32, [train_batch] + list(ac_space.shape))
        ADV = tf.placeholder(tf.float32, [train_batch])
        R = tf.placeholder(tf.float32, [train_batch])
        LR = tf.placeholder(tf.float32, [])

        print("Initialize policy objects")
        # CHANGE: A2C allows you to pass in policy_kwargs at this juncture
        # This would make it way easier to do
        step_model = policy(
            sess, ob_space, ac_space, nenvs, 1, nenvs, reuse=False)
        train_model = policy(
            sess, ob_space, ac_space, nenvs, nsteps, train_batch, reuse=True)

        neglogpac = train_model.proba_distribution.neglogp(A)

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.value_fn), R))
        entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
        loss = pg_loss + vf_loss * vf_coef - entropy * ent_coef

        # CHANGE: A2C uses tf_util.get_trainable_vars("model")
        params = find_trainable_variables("model")

        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        print("Create trainer")
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)
        # CHANGE: In A2C, global_variables_initializer() is run here

        def train(obs, states, rewards, masks, actions, values):
            # Equivalent of _train_step() in A2C
            advs = rewards - values
            n_steps = len(obs)
            for _ in range(n_steps):
                cur_lr = lr_scheduler.value()
            td_map = {
                train_model.obs_ph: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }

            if states:
                # TODO make this work for newer stateful policies
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy, cur_lr

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.sess = sess
        # Why var_list=params?
        # Otherwise we'll also save optimizer parameters,
        # which take up a /lot/ of space.
        # Why save_relative_paths=True?
        # So that the plain-text 'checkpoint' file written uses relative paths,
        # which seems to be needed in order to avoid confusing saver.restore()
        # when restoring from FloydHub runs.
        self.saver = tf.train.Saver(
            max_to_keep=1, var_list=params, save_relative_paths=True)
        tf.global_variables_initializer().run(session=sess)

    def load(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def save(self, ckpt_path, step_n):
        saved_path = self.saver.save(self.sess, ckpt_path, step_n)
        print("Saved policy checkpoint to '{}'".format(saved_path))


class Runner(object):
    def __init__(self,
                 env,
                 model,
                 nsteps,
                 nstack,
                 gamma,
                 gen_segments,
                 seg_pipe,
                 reward_predictor,
                 episode_vid_queue):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs

        # CHANGE: In A2C, this is defined as being of shape
        # (n_env*n_steps, nh, nw, nc)
        # Assuming that env.observation_space.shape = (nh, nw, nc)
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)

        # CHANGE: In A2C, this is defined as being of shape
        # (n__env, nh, nw, nc) According to the same observation space assumption
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        # The first stack of 4 frames: the first 3 frames are zeros,
        # with the last frame coming from env.reset().
        print("Got to before reset")
        print("Shape of self.obs: {}".format(self.obs.shape))
        obs = env.reset()
        print("Finished env reset")
        self.update_obs(obs)
        print("Finished updating obs")
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.gen_segments = gen_segments
        self.segment = Segment()
        self.seg_pipe = seg_pipe

        self.orig_reward = [0 for _ in range(nenv)]
        self.reward_predictor = reward_predictor

        self.episode_frames = []
        self.episode_vid_queue = episode_vid_queue
        print("Got to end of Runner creation")

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        # TODO take more general channel values
        self.obs = np.roll(self.obs, shift=-3, axis=3)
        self.obs[:, :, :, -3:] = obs[:, :, :, 0:3]

    def update_segment_buffer(self, mb_obs, mb_rewards, mb_dones):
        # Segments are only generated from the first worker.
        # Empirically, this seems to work fine.
        e0_obs = mb_obs[0]
        e0_rew = mb_rewards[0]
        e0_dones = mb_dones[0]
        assert_equal(e0_obs.shape[0], self.nsteps)
        # TODO make this general to nstack parameter
        assert(e0_obs.shape[-1] % 4 == 0)
        assert_equal(e0_rew.shape[0], self.nsteps)
        assert_equal(e0_dones.shape[0], self.nsteps)
        # TODO generalize across num_channels
        converted_image = cv2.cvtColor(e0_obs[0][:, :, -3:], cv2.COLOR_RGB2BGR)
        cv2.imwrite("eo_obs_segment_buffer.png", converted_image)
        for step in range(self.nsteps):
            self.segment.append(np.copy(e0_obs[step]), np.copy(e0_rew[step]))
            if len(self.segment) == 40 or e0_dones[step]:
                while len(self.segment) < 40:
                    # Pad to 25 steps long so that all segments in the batch
                    # have the same length.
                    # Note that the reward predictor needs the full frame
                    # stack, so we send all frames.
                    self.segment.append(e0_obs[step], 0)
                self.segment.finalise()
                try:
                    self.seg_pipe.put(self.segment, block=False)
                except queue.Full:
                    # If the preference interface has a backlog of segments
                    # to deal with, don't stop training the agents. Just drop
                    # the segment and keep on going.
                    pass
                self.segment = Segment()

    def update_episode_frame_buffer(self, mb_obs, mb_dones):
        e0_obs = mb_obs[0]
        e0_dones = mb_dones[0]
        for step in range(self.nsteps):
            # Here we only need to send the last frame (the most recent one)
            # from the 4-frame stack, because we're just showing output to
            # the user.
            # TODO make general for num_channels
            self.episode_frames.append(e0_obs[step, :, :, -3])
            if e0_dones[step]:
                self.episode_vid_queue.put(self.episode_frames)
                self.episode_frames = []

    def run(self):
        nenvs = len(self.env.remotes)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = \
            [], [], [], [], []
        mb_states = self.states

        # Run for nsteps steps in the environment
        for _ in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states,
                                                      self.dones)
            # actions here are of shape (1, 11)

            # IMPORTANT: Here we are adding multiple copies of the
            # stacked version of obs, and that's what we pass to the update_segment_buffer
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # len({obs, rewards, dones}) == nenvs
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            # SubprocVecEnv automatically resets when done
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        # i.e. from nsteps, nenvs to nenvs, nsteps
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        # The first entry was just the init state of 'dones' (all False),
        # before we'd actually run any steps, so drop it.
        mb_dones = mb_dones[:, 1:]

        # Log original rewards
        for env_n, (rs, dones) in enumerate(zip(mb_rewards, mb_dones)):
            assert_equal(rs.shape, (self.nsteps, ))
            assert_equal(dones.shape, (self.nsteps, ))
            for step_n in range(self.nsteps):
                self.orig_reward[env_n] += rs[step_n]
                if dones[step_n]:
                    easy_tf_log.tflog(
                        "orig_reward_{}".format(env_n),
                        self.orig_reward[env_n])
                    self.orig_reward[env_n] = 0

        if self.env.env_id == 'MovingDotNoFrameskip-v0':
            # For MovingDot, reward depends on both current observation and
            # current action, so encode action in the observations.
            # (We only need to set this in the most recent frame,
            # because that's all that the reward predictor for MovingDot
            # uses.)
            mb_obs[:, :, 0, 0, -1] = mb_actions[:, :]

        # Generate segments
        # (For MovingDot, this has to happen _after_ we've encoded the action
        # in the observations.)
        if self.gen_segments:
            self.update_segment_buffer(mb_obs, mb_rewards, mb_dones)

        # Replace rewards with those from reward predictor
        # (Note that this also needs to be done _after_ we've encoded the
        # action.)
        logging.debug("Original rewards:\n%s", mb_rewards)
        if self.reward_predictor:
            assert_equal(mb_obs.shape[0], nenvs)
            assert_equal(mb_obs.shape[1], self.nsteps)
            # TODO make general to stacking sizes other than 4
            assert(mb_obs.shape[-1] % 4 == 0)
            # TODO make general across num_channels
            h, w, c = mb_obs.shape[-3:]

            # TODO figure out what this reshape is doing here and whether it's necessary pre-reward-predictor
            mb_obs_allenvs = mb_obs.reshape(nenvs * self.nsteps, h, w, c)

            rewards_allenvs = self.reward_predictor.reward(mb_obs_allenvs)
            assert_equal(rewards_allenvs.shape, (nenvs * self.nsteps, ))
            mb_rewards = rewards_allenvs.reshape(nenvs, self.nsteps)
            assert_equal(mb_rewards.shape, (nenvs, self.nsteps))

            logging.debug("Predicted rewards:\n%s", mb_rewards)

        # Save frames for episode rendering
        if self.episode_vid_queue is not None:
            self.update_episode_frame_buffer(mb_obs, mb_dones)

        # Discount rewards
        mb_obs = mb_obs.reshape(self.batch_ob_shape)
        last_values = self.model.value(self.obs, self.states,
                                       self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
                zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                # Make sure that the first iteration of the loop inside
                # discount_with_dones picks up 'value' as the initial
                # value of r
                rewards = discount_with_dones(rewards + [value],
                                              dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        # Well, there's the culprit

        def flatten_correctly(arr):
            assert arr.shape[0] == 1
            new_shape = arr.shape[1:]
            return arr.reshape(new_shape)

        mb_rewards = flatten_correctly(mb_rewards)
        mb_actions = flatten_correctly(mb_actions)
        mb_values = flatten_correctly(mb_values)
        mb_masks = flatten_correctly(mb_masks)

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy,
          env,
          seed,
          start_policy_training_pipe,
          ckpt_save_dir,
          lr_scheduler,
          nsteps=5,
          nstack=4,
          total_timesteps=int(80e6),
          vf_coef=0.5,
          ent_coef=0.01,
          max_grad_norm=0.5,
          epsilon=1e-5,
          alpha=0.99,
          gamma=0.99,
          log_interval=25,
          ckpt_save_interval=1000,
          ckpt_load_dir=None,
          gen_segments=False,
          seg_pipe=None,
          reward_predictor=None,
          episode_vid_queue=None):

    tf.reset_default_graph()
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = deepcopy(env.observation_space)
    nh, nw, nc = ob_space.shape
    new_shape = (nh, nw, nc*nstack)
    # # TODO make this more general/pull zero and 255 out of existing obs space
    low, high = np.zeros(new_shape), np.full(new_shape, 255)
    ob_space.shape = new_shape
    ob_space.low = low
    ob_space.high = high

    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK

    def make_model():
        return Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nenvs=nenvs,
            nsteps=nsteps,
            nstack=nstack,
            num_procs=num_procs,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            lr_scheduler=lr_scheduler,
            alpha=alpha,
            epsilon=epsilon)

    with open(osp.join(ckpt_save_dir, 'make_model.pkl'), 'wb') as fh:
        fh.write(cloudpickle.dumps(make_model))

    print("Initialising policy...")
    if ckpt_load_dir is None:
        model = make_model()
    else:
        with open(osp.join(ckpt_load_dir, 'make_model.pkl'), 'rb') as fh:
            make_model = cloudpickle.loads(fh.read())
        model = make_model()

        ckpt_load_path = tf.train.latest_checkpoint(ckpt_load_dir)
        model.load(ckpt_load_path)
        print("Loaded policy from checkpoint '{}'".format(ckpt_load_path))

    ckpt_save_path = osp.join(ckpt_save_dir, 'policy.ckpt')
    print("Model loaded")
    runner = Runner(env=env,
                    model=model,
                    nsteps=nsteps,
                    nstack=nstack,
                    gamma=gamma,
                    gen_segments=gen_segments,
                    seg_pipe=seg_pipe,
                    reward_predictor=reward_predictor,
                    episode_vid_queue=episode_vid_queue)

    # nsteps: e.g. 5
    # nenvs: e.g. 16
    nbatch = nenvs * nsteps
    fps_tstart = time.time()
    fps_nsteps = 0
    print("Starting workers")

    # Before we're told to start training the policy itself,
    # just generate segments for the reward predictor to be trained with
    while True:
        runner.run()
        try:
            start_policy_training_pipe.get(block=False)
        except queue.Empty:
            continue
        else:
            break

    print("Starting policy training")
    print("Max val: {}".format(total_timesteps // nbatch + 1))
    for update in range(1, total_timesteps // nbatch + 1):
        # Run for nsteps

        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy, cur_lr = model.train(
            obs, states, rewards, masks, actions, values)
        fps_nsteps += nbatch

        if update % log_interval == 0 and update != 0:
            fps = fps_nsteps / (time.time() - fps_tstart)
            fps_nsteps = 0
            fps_tstart = time.time()

            print("Trained policy for {} time steps".format(update * nbatch))

            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("learning_rate", cur_lr)
            logger.dump_tabular()

        if update != 0 and update % ckpt_save_interval == 0:
            model.save(ckpt_save_path, update)

    model.save(ckpt_save_path, update)
