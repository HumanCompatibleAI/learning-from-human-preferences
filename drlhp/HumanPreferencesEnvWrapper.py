from gym import Wrapper
import numpy as np
import sys
import os
import time
import os.path as osp
import queue
from multiprocessing import Process, Queue
from drlhp.pref_db import Segment, PrefDB, PrefBuffer
from drlhp.params import parse_args, PREFS_VAL_FRACTION
from drlhp.reward_predictor import RewardPredictorEnsemble
from drlhp.utils import ForkedPdb




class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self, env, reward_predictor_network, preference_interface,
                 just_pretrain=False, just_collect_prefs=False,
                 nstack=4, segment_length=40, n_initial_epochs=0, n_initial_prefs=40):
        super(HumanPreferencesEnvWrapper, self).__init__(env)
        self.seg_pipe = Queue(maxsize=1)
        self.pref_pipe = Queue(maxsize=1)
        self.start_policy_training_flag = Queue(maxsize=1)
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
        self.prefs_dir = None

        self.n_prefs_train = 100
        self.n_prefs_test = 100
        self.n_initial_prefs = n_initial_prefs
        self.n_initial_epochs = n_initial_epochs
        self.reward_predictor_sleep_interval = 10
        self.pref_buffer = None
        self.max_prefs = 200
        self.log_dir = "drlhp_logs/"
        self.val_interval = 10
        self.ckpt_interval = 10
        self.just_pretrain = just_pretrain
        self.just_collect_prefs = just_collect_prefs

        self.reward_training_iters = 0
        self.last_reward_training_time = time.time()
        self.last_reward_training_size = 0
        self.reward_training_pref_diff = 5

        if just_collect_prefs:
            self.reward_predictor = None
        else:
            self.reward_predictor = RewardPredictorEnsemble(
                core_network=reward_predictor_network,
                log_dir=self.log_dir,
                batchnorm=False,
                dropout=0.0,
                lr=7e-4,
                obs_shape=self.obs_shape)
            self.reward_predictor.init_network()

        self._start_pref_interface()
        self._load_or_create_pref_db()
        if n_initial_prefs > 0:
            self._train_reward_predictor()

    def _save_prefs(self):
        pref_db_train, pref_db_val = self.pref_buffer.get_dbs()
        train_path = osp.join(self.log_dir, 'train.pkl.gz')
        pref_db_train.save(train_path)
        print("Saved training preferences to '{}'".format(train_path))
        val_path = osp.join(self.log_dir, 'val.pkl.gz')
        pref_db_val.save(val_path)
        print("Saved validation preferences to '{}'".format(val_path))

    def _start_pref_interface(self):
        def f():
            sys.stdin = os.fdopen(0)
            self.preference_interface.run(seg_pipe=self.seg_pipe,
                                          pref_pipe=self.pref_pipe)
        self.pref_interface_proc = Process(target=f, daemon=True)
        self.pref_interface_proc.start()


    def _load_or_create_pref_db(self):
        if self.prefs_dir is not None:
            train_path = osp.join(self.prefs_dir, 'train.pkl.gz')
            pref_db_train = PrefDB.load(train_path)
            print("Loaded training preferences from '{}'".format(train_path))
            n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))

            val_path = osp.join(self.prefs_dir, 'val.pkl.gz')
            pref_db_val = PrefDB.load(val_path)
            print("Loaded validation preferences from '{}'".format(val_path))
            n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))
        else:
            n_train = self.max_prefs * (1 - PREFS_VAL_FRACTION)
            n_val = self.max_prefs * PREFS_VAL_FRACTION
            pref_db_train = PrefDB(maxlen=n_train)
            pref_db_val = PrefDB(maxlen=n_val)
        self.pref_buffer = PrefBuffer(db_train=pref_db_train,
                                 db_val=pref_db_val)
        self.pref_buffer.start_recv_thread(self.pref_pipe)


    def _pretrain_reward_predictor(self):
        print(f"Pretraining reward predictor for {self.n_initial_epochs} epochs")
        pref_db_train, pref_db_val = self.pref_buffer.get_dbs()
        for i in range(self.n_initial_epochs):
            print("Reward predictor training epoch {}".format(i))
            self.reward_predictor.train(pref_db_train, pref_db_val, self.val_interval)
            if i and i % self.ckpt_interval == 0:
                self.reward_predictor.save()
        self.start_policy_training_flag.put(True)

    def _train_reward_predictor(self):
        # cur_time = time.time()
        # if cur_time - self.last_reward_training_time > self.reward_predictor_sleep_interval:
        pref_db_train, pref_db_val = self.pref_buffer.get_dbs()
        cur_train_size = len(pref_db_train)
        if cur_train_size - self.last_reward_training_size >= self.reward_training_pref_diff:
            print(f"Training reward predictor on {cur_train_size} preferences")
            self._save_prefs()
            self.reward_predictor.train(pref_db_train, pref_db_val, self.val_interval)
            if self.reward_training_iters and self.reward_training_iters % self.ckpt_interval == 0:
                self.reward_predictor.save()
            self.reward_training_iters += 1
            self.last_reward_training_size = cur_train_size
            self.last_reward_training_time = time.time()

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



    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_episode_segment(obs, reward, done)
        predicted_reward = self.reward_predictor.reward(np.array([np.array(obs)]))
        self._train_reward_predictor()
        return obs, predicted_reward, done, info
