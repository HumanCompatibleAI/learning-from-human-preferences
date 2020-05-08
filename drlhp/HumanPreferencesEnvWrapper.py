import multiprocessing as mp
from gym import Wrapper
import numpy as np
import sys
import os
import time
import os.path as osp
import queue
from drlhp.pref_db import Segment, PrefDB, PrefBuffer
from drlhp.params import parse_args, PREFS_VAL_FRACTION
from drlhp.reward_predictor import RewardPredictorEnsemble
from functools import partial
from drlhp.utils import ForkedPdb


def _save_prefs(pref_buffer, log_dir):
    pref_db_train, pref_db_val = pref_buffer.get_dbs()
    train_path = osp.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    print(f"Saved {len(pref_db_train)} training preferences to '{train_path}'")
    val_path = osp.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    print(f"Saved {len(pref_db_val)} validation preferences to '{val_path}'")

def _load_or_create_pref_db(prefs_dir, max_prefs):
    if prefs_dir is not None:
        train_path = osp.join(prefs_dir, 'train.pkl.gz')
        pref_db_train = PrefDB.load(train_path)
        print("Loaded training preferences from '{}'".format(train_path))
        n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
        print("({} preferences, {} segments)".format(n_prefs, n_segs))

        val_path = osp.join(prefs_dir, 'val.pkl.gz')
        pref_db_val = PrefDB.load(val_path)
        print("Loaded validation preferences from '{}'".format(val_path))
        n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
        print("({} preferences, {} segments)".format(n_prefs, n_segs))
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
    pref_interface.run(seg_pipe=seg_pipe,
                       pref_pipe=pref_pipe,
                       remaining_pairs=remaining_pairs,
                       kill_processes=kill_processes)

def make_reward_predictor(reward_predictor_network, log_dir, obs_shape, checkpoint_dir=None):
    reward_predictor = RewardPredictorEnsemble(
        core_network=reward_predictor_network,
        log_dir=log_dir,
        batchnorm=False,
        dropout=0.0,
        lr=7e-4,
        obs_shape=obs_shape)
    reward_predictor.init_network(load_ckpt_dir=checkpoint_dir)
    return reward_predictor

def _train_reward_predictor(reward_predictor_network, obs_shape, pref_pipe, reward_training_steps, prefs_dir, max_prefs,
                            ckpt_interval, kill_processes_flag, database_refresh_interval,
                            validation_interval, num_initial_prefs, save_prefs_flag, log_dir):
    # TODO implement a training steps counter that this can increment

    reward_predictor = make_reward_predictor(reward_predictor_network, log_dir, obs_shape)

    pref_buffer = _load_or_create_pref_db(prefs_dir, max_prefs)
    pref_buffer.start_recv_thread(pref_pipe)
    pref_db_train, pref_db_val = pref_buffer.get_dbs()
    minimum_prefs_met = False
    while True:
        if save_prefs_flag.value == 1:
            _save_prefs(pref_buffer, log_dir)
            save_prefs_flag.value = 0
        if kill_processes_flag.value == 1:
            pref_buffer.stop_recv_thread()
            return
        if not minimum_prefs_met:
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            if len(pref_db_train) < num_initial_prefs:
                print(f"REWARD: Reward db of length {len(pref_db_train)}, waiting for length {num_initial_prefs}")
                time.sleep(5)
                continue
            else:
                minimum_prefs_met = True
        if reward_training_steps.value % database_refresh_interval == 0:
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
        cur_train_size = len(pref_db_train)
        print(f"Training reward predictor on {cur_train_size} preferences, iteration {reward_training_steps.value }")
        reward_predictor.train(pref_db_train, pref_db_val, validation_interval)
        if reward_training_steps.value  > 0 and reward_training_steps.value % ckpt_interval == 0:
            reward_predictor.save()
        reward_training_steps.value += 1

class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self, env, reward_predictor_network, preference_interface,
                 just_pretrain=False, just_collect_prefs=False,
                 nstack=4, segment_length=40, n_initial_training_steps=50, n_initial_prefs=40,
                 mp_context='spawn', pretrained_reward_predictor_dir=None, log_dir="drlhp_logs/"):

        # Recommend using 'spawn' for non synthetic preferences and 'fork' for synthetic
        super(HumanPreferencesEnvWrapper, self).__init__(env)
        self.mp_context = mp_context
        self.seg_pipe = mp.get_context(self.mp_context ).Queue(maxsize=1)
        self.pref_pipe = mp.get_context(self.mp_context ).Queue(maxsize=1)
        self.remaining_pairs = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_pref_interface_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_reward_training_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.save_prefs_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.reward_training_steps = mp.get_context(self.mp_context).Value('i', 0)


        self.recent_obs_stack = [] # rolling list of last 4 observations
        self.train_reward = True # A boolean for whether reward predictor is frozen or actively being trained
        self.obs_shape = env.observation_space.shape
        self.nstack = nstack
        self.episode_segment = Segment()
        self.segment_length = segment_length
        self.preference_interface = preference_interface
        self.obs_stack = np.zeros((self.obs_shape[0], self.obs_shape[1], self.obs_shape[2] * nstack), dtype=np.uint8)

        self.pref_interface_proc = None
        self.reward_training_proc = None

        self.reward_predictor_network = reward_predictor_network
        self.pretrained_reward_predictor_dir = pretrained_reward_predictor_dir
        if self.pretrained_reward_predictor_dir is not None:
            self.reward_predictor = make_reward_predictor(reward_predictor_network, log_dir,
                                                          self.obs_shape,
                                                          checkpoint_dir=self.pretrained_reward_predictor_dir)
        else:
            self.reward_predictor = None


        self.prefs_dir = None

        self.n_prefs_train = 100
        self.n_prefs_test = 100
        self.n_initial_prefs = n_initial_prefs # Number of prefs before you start training reward predictor
        self.n_initial_training_steps = n_initial_training_steps # Number of reward predictor training steps before switch rewards
        self.reward_predictor_sleep_interval = 10
        self.pref_buffer = None
        self.max_prefs = 200
        self.log_dir = log_dir
        self.val_interval = 10
        self.ckpt_interval = 5
        self.reward_database_refresh_interval = 2


        self.just_pretrain = just_pretrain
        self.just_collect_prefs = just_collect_prefs

        self.last_reward_training_time = time.time()
        self.last_reward_training_size = 0
        self.reward_training_pref_diff = 5

        self.reward_predictor_n_train = 0
        self.reward_predictor_refresh_interval = 20

        self._start_pref_interface()
        self._start_reward_predictor_training()





    def _start_pref_interface(self):
        self.pref_interface_proc = mp.get_context(self.mp_context).Process(target=run_pref_interface, daemon=True,
                                                                           args=(self.preference_interface,
                                                                                 self.seg_pipe, self.pref_pipe,
                                                                                 self.remaining_pairs,
                                                                                 self.kill_pref_interface_flag))
        self.pref_interface_proc.start()

    def _start_reward_predictor_training(self):
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
                                                                          self.log_dir))
        self.reward_training_proc.start()
    #
    # def _pretrain_reward_predictor(self):
    #     print(f"Pretraining reward predictor for {self.n_initial_epochs} epochs")
    #     pref_db_train, pref_db_val = self.pref_buffer.get_dbs()
    #     for i in range(self.n_initial_epochs):
    #         print("Reward predictor training epoch {}".format(i))
    #         self.reward_predictor.train(pref_db_train, pref_db_val, self.val_interval)
    #         self.reward_training_iters += 1
    #         if i and i % self.ckpt_interval == 0:
    #             self.reward_predictor.save()
    #     self.start_policy_training_flag.put(True)

    def _update_episode_segment(self, obs, reward, done):
        self.episode_segment.append(np.copy(obs), np.copy(reward))
        if done:
            while len(self.episode_segment) < self.segment_length:
                self.episode_segment.append(np.copy(obs), 0)

        if len(self.episode_segment) == self.segment_length:
            self.episode_segment.finalise()
            try:
                self.seg_pipe.put(self.episode_segment, block=False)
            except queue.Full:
                # If the preference interface has a backlog of segments
                # to deal with, don't stop training the agents. Just drop
                # the segment and keep on going.
                pass
            self.episode_segment = Segment()

    def save_prefs(self):
        self.save_prefs_flag.value = 1

    def load_reward_predictor(self):
        if self.reward_predictor is None:
            print("Loading reward predictor; switching over to model reward now")
        self.reward_predictor_n_train = self.reward_training_steps.value
        self.reward_predictor = RewardPredictorEnsemble(
            core_network=self.reward_predictor_network,
            log_dir=self.log_dir,
            batchnorm=False,
            dropout=0.0,
            lr=7e-4,
            obs_shape=self.obs_shape)
        self.reward_predictor.init_network(self.reward_predictor.checkpoint_dir)

    def step(self, action):
        if (self.reward_predictor is None and self.reward_training_steps.value >= self.n_initial_training_steps)\
                or (self.reward_training_steps.value - self.reward_predictor_n_train > self.reward_predictor_refresh_interval):
            print("Loading reward predictor")
            self.load_reward_predictor()
        obs, reward, done, info = self.env.step(action)
        self._update_episode_segment(obs, reward, done)
        if self.reward_predictor is not None:
            predicted_reward = self.reward_predictor.reward(np.array([np.array(obs)]))
            return obs, predicted_reward, done, info
        else:
            return obs, reward, done, info

    def cleanup_processes(self):
        self.kill_reward_training_flag.value = 1
        self.kill_pref_interface_flag.value = 1
        #self.pref_buffer.stop_recv_thread()
        if self.reward_training_proc is not None:
            self.reward_training_proc.join()
        if self.pref_interface_proc is not None:
            self.pref_interface_proc.join()
        self.seg_pipe.close()
        self.seg_pipe.join_thread()
        self.pref_pipe.close()
        self.pref_pipe.join_thread()

