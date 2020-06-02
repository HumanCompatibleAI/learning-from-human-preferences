import multiprocessing as mp
from gym import Wrapper
import numpy as np
import sys
import os
import time
import os.path as osp
import queue
import logging
from drlhp.pref_db import Segment, PrefDB, PrefBuffer
from drlhp.reward_predictor import RewardPredictorEnsemble

PREFS_VAL_FRACTION = 0.2

def _save_prefs(pref_buffer, log_dir, logger):
    pref_db_train, pref_db_val = pref_buffer.get_dbs()
    train_path = osp.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    logger.info(f"Saved {len(pref_db_train)} training preferences to '{train_path}'")
    val_path = osp.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    logger.info(f"Saved {len(pref_db_val)} validation preferences to '{val_path}'")

def _load_or_create_pref_db(prefs_dir, max_prefs, logger):
    if prefs_dir is not None:
        train_path = osp.join(prefs_dir, 'train.pkl.gz')
        pref_db_train = PrefDB.load(train_path)
        logger.info("Loaded training preferences from '{}'".format(train_path))
        n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
        logger.info("({} preferences, {} segments)".format(n_prefs, n_segs))

        val_path = osp.join(prefs_dir, 'val.pkl.gz')
        pref_db_val = PrefDB.load(val_path)
        logger.info("Loaded validation preferences from '{}'".format(val_path))
        n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
        logger.info("({} preferences, {} segments)".format(n_prefs, n_segs))
    else:
        n_train = max_prefs * (1 - PREFS_VAL_FRACTION)
        n_val = max_prefs * PREFS_VAL_FRACTION
        pref_db_train = PrefDB(maxlen=n_train)
        pref_db_val = PrefDB(maxlen=n_val)
    pref_buffer = PrefBuffer(db_train=pref_db_train,
                                  db_val=pref_db_val)
    return pref_buffer

def run_pref_interface(pref_interface, seg_pipe, pref_pipe, remaining_pairs, kill_processes):
    sys.stdin = os.fdopen(0)
    print("Running pref interface func")
    pref_interface.run(seg_pipe=seg_pipe,
                       pref_pipe=pref_pipe,
                       remaining_pairs=remaining_pairs,
                       kill_processes=kill_processes)

def make_reward_predictor(reward_predictor_network, log_dir, obs_shape, logger, checkpoint_dir=None):
    reward_predictor = RewardPredictorEnsemble(
        core_network=reward_predictor_network,
        log_dir=log_dir,
        batchnorm=False,
        dropout=0.0,
        lr=7e-4,
        obs_shape=obs_shape,
        logger=logger)
    reward_predictor.init_network(load_ckpt_dir=checkpoint_dir)
    return reward_predictor

def _train_reward_predictor(reward_predictor_network, obs_shape, pref_pipe, reward_training_steps, prefs_dir, max_prefs,
                            ckpt_interval, kill_processes_flag, database_refresh_interval,
                            validation_interval, num_initial_prefs, save_prefs_flag, save_model_flag,
                            pretrained_reward_predictor_dir, log_dir, log_level, train_reward, pref_db_size):
    print("Running reward predictor func")

    reward_predictor_logger = logging.getLogger("_train_reward_predictor")
    reward_predictor_logger.setLevel(log_level)
    reward_predictor = make_reward_predictor(reward_predictor_network, log_dir, obs_shape, reward_predictor_logger,
                                             checkpoint_dir=pretrained_reward_predictor_dir)

    pref_buffer = _load_or_create_pref_db(prefs_dir, max_prefs, reward_predictor_logger)
    pref_buffer.start_recv_thread(pref_pipe)
    minimum_prefs_met = False

    while True:
        pref_db_train, pref_db_val = pref_buffer.get_dbs()
        pref_db_size.value = len(pref_db_train) + len(pref_db_val)
        if save_prefs_flag.value == 1:
            _save_prefs(pref_buffer, log_dir, reward_predictor_logger)
            save_prefs_flag.value = 0
        if kill_processes_flag.value == 1:
            pref_buffer.stop_recv_thread()
            return
        if not train_reward:
            # There might be some circumstances where we just want to collect and save preferences (for which we need to create a PrefDB)
            # but might not want to train a reward model. For those circumstances, we can set train_reward to False
            continue

        if not minimum_prefs_met:
            if len(pref_db_train) < num_initial_prefs or len(pref_db_val) < 1:
                #print(f"Current reward DB sizes: {len(pref_db_train)}, {len(pref_db_val)}")
                reward_predictor_logger.info(f"Reward dbs of length {len(pref_db_train)}, {len(pref_db_val)}, waiting for length {num_initial_prefs}, 1 to start training")
                time.sleep(1)
                continue
            else:
                minimum_prefs_met = True
        if reward_training_steps.value % database_refresh_interval == 0:
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
        cur_train_size = len(pref_db_train)
        reward_predictor_logger.info(f"Training reward predictor on {cur_train_size} preferences, iteration {reward_training_steps.value }")
        reward_predictor.train(pref_db_train, pref_db_val, validation_interval)
        reward_training_steps.value += 1
        if (save_model_flag.value == 1) or (reward_training_steps.value % ckpt_interval == 0):
            reward_predictor.save()
            save_model_flag.value = 0


class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self, env, reward_predictor_network, preference_interface,
                 train_reward=True, collect_prefs=True, nstack=4, segment_length=40,
                 n_initial_training_steps=50, n_initial_prefs=40, prefs_dir=None,
                 mp_context='spawn', pretrained_reward_predictor_dir=None, log_dir="drlhp_logs/",
                 max_prefs_in_db=10000, obs_transform_func=None, reward_predictor_ckpt_interval=10,
                 env_wrapper_log_level=logging.INFO, reward_predictor_log_level=logging.INFO):

        # Recommend using 'spawn' for non synthetic preferences and 'fork' for synthetic
        super(HumanPreferencesEnvWrapper, self).__init__(env)
        self.logger = logging.getLogger("HumanPreferencesEnvWrapper")
        self.logger.setLevel(env_wrapper_log_level)
        self.reward_predictor_log_level = reward_predictor_log_level

        # Save a bunch of init parameters as wrapper properties
        self.mp_context = mp_context
        self.train_reward = train_reward
        self.collect_prefs = collect_prefs
        self.segment_length = segment_length
        self.segments_collected = 0
        self.preference_interface = preference_interface
        self.reward_predictor_network = reward_predictor_network
        self.pretrained_reward_predictor_dir = pretrained_reward_predictor_dir
        self.obs_transform_func = obs_transform_func
        self.nstack = nstack
        self.prefs_dir = prefs_dir
        self.max_prefs = max_prefs_in_db
        self.n_initial_prefs = n_initial_prefs  # Number of prefs before you start training reward predictor
        self.n_initial_training_steps = n_initial_training_steps  # Number of reward predictor training steps before switch rewards
        self.log_dir = log_dir
        self.val_interval = 10
        self.ckpt_interval = reward_predictor_ckpt_interval
        self.reward_database_refresh_interval = 1
        self.reward_predictor_n_train = 0
        self.reward_predictor_refresh_interval = 20
        self.using_reward_from_predictor = False

        # Create Queues and Values to handle multiprocessing communication
        self.seg_pipe = mp.get_context(self.mp_context).Queue(maxsize=1)
        self.pref_pipe = mp.get_context(self.mp_context).Queue(maxsize=1)
        self.remaining_pairs = mp.get_context(self.mp_context).Value('i', 0)
        self.pref_db_size = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_pref_interface_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_reward_training_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.save_model_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.save_prefs_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.reward_training_steps = mp.get_context(self.mp_context).Value('i', 0)

        self.recent_obs_stack = []  # rolling list of last 4 observations
        self.episode_segment = Segment()
        self.collecting_segments = True
        self.obs_shape = env.observation_space.shape
        self.obs_stack = np.zeros((self.obs_shape[0], self.obs_shape[1], self.obs_shape[2] * nstack), dtype=np.uint8)
        self.pref_interface_proc = None
        self.reward_training_proc = None
        self.pref_buffer = None
        self.reward_predictor = None

    def reset(self):
        if self.collect_prefs:
            self._start_pref_interface()
        if self.train_reward or self.collect_prefs:
            self._start_reward_predictor_training()
        return self.env.reset()

    def _start_pref_interface(self):
        print("Should be starting pref interface")
        self.pref_interface_proc = mp.get_context(self.mp_context).Process(target=run_pref_interface, daemon=True,
                                                                           args=(self.preference_interface,
                                                                                 self.seg_pipe, self.pref_pipe,
                                                                                 self.remaining_pairs,
                                                                                 self.kill_pref_interface_flag))
        self.pref_interface_proc.start()

    def _start_reward_predictor_training(self):
        print("Should be starting reward predictor function")
        self.reward_training_proc = mp.get_context('fork').Process(target=_train_reward_predictor, daemon=True,
                                                                   args=(self.reward_predictor_network,
                                                                         self.obs_shape,
                                                                         self.pref_pipe,
                                                                         self.reward_training_steps,
                                                                         self.prefs_dir,
                                                                         self.max_prefs,
                                                                         self.ckpt_interval,
                                                                         self.kill_reward_training_flag,
                                                                         self.reward_database_refresh_interval,
                                                                         self.val_interval,
                                                                         self.n_initial_prefs,
                                                                         self.save_prefs_flag,
                                                                         self.save_model_flag,
                                                                         self.pretrained_reward_predictor_dir,
                                                                         self.log_dir,
                                                                         self.reward_predictor_log_level,
                                                                         self.train_reward,
                                                                         self.pref_db_size))
        self.reward_training_proc.start()

    def _update_episode_segment(self, obs, reward, done):
        if self.obs_transform_func is not None:
            obs = self.obs_transform_func(obs)
        self.episode_segment.append(np.copy(obs), np.copy(reward))
        if done:
            while len(self.episode_segment) < self.segment_length:
                self.episode_segment.append(np.copy(obs), 0)

        if len(self.episode_segment) == self.segment_length:
            self.segments_collected += 1
            self.episode_segment.finalise()
            try:
                #print("Sending segment to pref interface!")
                self.seg_pipe.put(self.episode_segment, block=False)
            except queue.Full:
                # If the preference interface has a backlog of segments
                # to deal with, don't stop training the agents. Just drop
                # the segment and keep on going.
                pass
            self.episode_segment = Segment()

    def save_prefs(self):
        self.save_prefs_flag.value = 1

    def save_reward_predictor(self):
        self.save_model_flag.value = 1

    def stop_segment_collection(self):
        self.collecting_segments = False

    def start_segment_collection(self):
        self.collecting_segments = True
        self.episode_segment = Segment()

    def load_reward_predictor(self):
        if self.reward_predictor is None:
            self.logger.info("Loading reward predictor; will use model reward now")
            self.reward_predictor = RewardPredictorEnsemble(
                core_network=self.reward_predictor_network,
                log_dir=self.log_dir,
                batchnorm=False,
                dropout=0.0,
                lr=7e-4,
                obs_shape=self.obs_shape,
                logger=self.logger)
        self.reward_predictor_n_train = self.reward_training_steps.value
        self.reward_predictor.init_network(self.pretrained_reward_predictor_dir)#.init_network(self.reward_predictor.checkpoint_dir)

    def step(self, action):
        sufficiently_trained = self.reward_predictor is None and self.reward_training_steps.value >= self.n_initial_training_steps
        pretrained_model = self.reward_predictor is None and self.pretrained_reward_predictor_dir is not None
        should_update_model = self.reward_training_steps.value - self.reward_predictor_n_train > self.reward_predictor_refresh_interval
        if sufficiently_trained or pretrained_model or should_update_model:
            if sufficiently_trained:
                self.logger.info("Model is sufficiently trained, switching to it for reward")
            if should_update_model:
                self.logger.info("Updating model used for env reward")
            self.load_reward_predictor()
            self.using_reward_from_predictor = True
        obs, reward, done, info = self.env.step(action)
        if self.collecting_segments:
            self._update_episode_segment(obs, reward, done)
        if self.reward_predictor is not None:
            predicted_reward = self.reward_predictor.reward(np.array([np.array(obs)]))
            return obs, predicted_reward, done, info
        else:
            return obs, reward, done, info


    def _cleanup_processes(self):
        self.logger.debug("Sending kill flags to processes")
        self.kill_reward_training_flag.value = 1
        self.kill_pref_interface_flag.value = 1

        self.logger.debug("Joining processes that are running")
        if self.reward_training_proc is not None:
            self.reward_training_proc.join()
        if self.pref_interface_proc is not None:
            self.pref_interface_proc.join()

        self.logger.debug("Closing seg pipe")
        self.seg_pipe.close()
        self.seg_pipe.join_thread()
        self.logger.debug("Closing pref pipe")
        self.pref_pipe.close()
        self.pref_pipe.join_thread()

    def close(self):
        self._cleanup_processes()
        self.env.close()
