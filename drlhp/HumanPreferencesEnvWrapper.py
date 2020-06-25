import multiprocessing as mp
from gym import Wrapper, Env
import numpy as np
import sys
import os
import time
import os.path as osp
import queue
import logging
from drlhp.pref_db import Segment, PrefDB, PrefBuffer
from drlhp.pref_interface import PrefInterface
from drlhp.reward_predictor import RewardPredictorEnsemble
from drlhp.reward_predictor_core_network import net_cnn
from typing import Callable
import tensorflow as tf

PREFS_VAL_FRACTION = 0.2


def _save_prefs(pref_buffer: PrefBuffer,
                log_dir: str,
                logger: logging.Logger):
    """
    Saves the preferences stored in the databases on a given PrefBuffer to directories within `log_dir`

    :param pref_buffer: The PrefBuffer containing train and validation DBs we want to save
    :param log_dir: The directory to which we want our `train|val.pkl.gz` files to be saved
    :param logger: The logger object we want to use to log progress within this function
    """
    pref_db_train, pref_db_val = pref_buffer.get_dbs()
    train_path = osp.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    logger.info(f"Saved {len(pref_db_train)} training preferences to '{train_path}'")
    val_path = osp.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    logger.info(f"Saved {len(pref_db_val)} validation preferences to '{val_path}'")


def _load_or_create_pref_db(prefs_dir: str,
                            max_prefs: int,
                            logger: logging.Logger) -> PrefBuffer:
    """
    Create a PrefBuffer containing of two PrefDBs, either by loading them from disk (if `prefs_dir` is not None)
    or creating them from scratch.

    :param prefs_dir: Directory which PrefDBs should be loaded from; if None, they should be created anew
    :param max_prefs: The total number of preferences we want to store in the PrefDBs (split across both train and val DBs)
    :param logger: The logger object we want to use to log progress within this function

    :return: A PrefBuffer containing your PrefDBs
    """

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


def _run_pref_interface(pref_interface: PrefInterface,
                        seg_pipe: mp.Queue,
                        pref_pipe: mp.Queue,
                        remaining_pairs: mp.Value,
                        kill_processes: mp.Value,
                        log_level: int = logging.INFO):
    """
    Basically a large lambda function for calling pref_interface.run(); meant to be used as the target of a
    multiprocessing Process.

    :param pref_interface: The PrefInterface object you want to run
    :param seg_pipe: A multiprocessing Queue in which the env will add new segments for the PrefInterface to pair and
                     request preferences for
    :param pref_pipe: A multiprocessing Queue for the PrefInterface to add preferences once collected, which will make
                      them accessible to the PrefDB in which they are stored and used for reward predictor training
    :param remaining_pairs: A multiprocessing Value that the PrefInterface can use to keep track of the remaining pairs
                            of segments it has to get preferences for, so that information is accessible externally
    :param kill_processes: A multiprocessing Value that will be set to 1 if we want to terminate running processes
                           (specifically, it will trigger pref_interface.run() to return so we can easily join
                           the process)
    """
    #sys.stdin = os.fdopen(0)

    print("Running pref interface func")
    pref_interface.run(seg_pipe=seg_pipe,
                       pref_pipe=pref_pipe,
                       remaining_pairs=remaining_pairs,
                       kill_processes=kill_processes,
                       log_level=log_level)


def _make_reward_predictor(reward_predictor_network: Callable,
                           log_dir: str,
                           obs_shape: tuple,
                           logger: logging.Logger,
                           checkpoint_dir: str = None) -> RewardPredictorEnsemble:
    """
    A helper function for making a RewardPredictorEnsemble and initiating it with a checkpoint, if one is present.
    If `checkpoint_dir` is None, the reward predictor will be initialized randomly

    :param reward_predictor_network: A Tensorflow-trainable callable that takes in observations and outputs rewards
    :param log_dir: A string path specifying where you want RewardPredictorEnsemble to store logs and other artifacts
    :param obs_shape: A tuple specifying the observation shape that you'll want your RewardPredictorEnsemble to take in
    :param logger: The logger object we want to use to log progress within this function
    :param checkpoint_dir: Optional, a string path specifying the checkpoint directory from which you want to load a
                           saved RewardPredictorEnsemble

    :return: Your newly-created RewardPredictorEnsemble
    """
    reward_predictor = RewardPredictorEnsemble(
        core_network=reward_predictor_network,
        log_dir=log_dir,
        batchnorm=False,
        dropout=0.0,
        lr=7e-4,
        obs_shape=obs_shape,
        logger=logger)
    print("RewardPredictorEnsemble created")
    reward_predictor.init_network(load_ckpt_dir=checkpoint_dir)
    print("RewardPredictorEnsemble initialized")
    return reward_predictor


def _train_reward_predictor(reward_predictor_network: Callable,
                            train_reward: bool,
                            pretrained_reward_predictor_dir: str,
                            obs_shape: tuple,
                            pref_pipe: mp.Queue,
                            pref_db_size: int,
                            prefs_dir: str,
                            max_prefs: int,
                            ckpt_interval: int,
                            num_initial_prefs: int,
                            reward_training_steps: mp.Value,
                            database_refresh_interval: int,
                            validation_interval: int,
                            kill_processes_flag: mp.Value,
                            save_prefs_flag: mp.Value,
                            save_model_flag: mp.Value,
                            log_dir: str,
                            log_level: int # logging levels are technically ints
                            ):
    """
    A function, meant to be run inside a multiprocessing process, to create training and validation PrefDBs, and
    train a reward predictor using the preferences stored in those DBs.


    :param reward_predictor_network: A callable mapping from input obs to reward scalar
    :param obs_shape: A tuple specifying the input observation shape you want your reward model to take in
    :param pref_pipe: A multiprocessing queue for the PrefInterface to send segment pairs with preferences attached to
                      them to the PrefBuffer
    :param reward_training_steps: A multiprocessing value for keeping track of reward training steps
    :param prefs_dir: A string path specifying where existing preference DBs are stored on disk; if None, new
                      empty PrefDBs are created
    :param max_prefs: The max number of preferences to store in your PrefDBs, across both training and validation
    :param ckpt_interval: The interval of reward training steps on which to save a checkpoint of our reward predictor
    :param kill_processes_flag: A multiprocessing Value that will be set to 1 when we want to terminate processes;
                                this will trigger this function to return, making it easier to join the process
    :param database_refresh_interval: The interval of reward training steps on which to update the PrefDBs being used
                                      to train our reward predictor
    :param validation_interval: The interval of reward training steps on which to perform validation of the reward model
    :param num_initial_prefs: How many preferences our training PrefDB must have before we start training
                              the reward model
    :param save_prefs_flag: A multiprocessing Value that will be set to 1 when we want to save preferences
    :param save_model_flag: A multiprocessing Value that will be set to 1 when we want to trigger a model save
    :param pretrained_reward_predictor_dir: A string path specifying where a pre-trained reward model is saved;
                                            if None, assumes none exist, and initializes reward model from random
    :param log_dir: A strong path specifying a directory where logs and artifacts will be saved
    :param log_level: The log level you want for the logger within this function
    :param train_reward: A boolean specifying whether you want to actually train a reward model, or just use this
                         function to create a set of PrefDBs so they can be filled with preferences.
    :param pref_db_size: A multiprocessing Value used to store the aggregated size of our PrefDBs, so that size can be
                         queried externally
    """

    reward_predictor_logger = logging.getLogger("_train_reward_predictor")
    reward_predictor_logger.setLevel(log_level)
    reward_predictor_logger.info("Reward predictor works at all")
    # Create a RewardPredictorEnsemble using the specified core network and obs shape
    reward_predictor = _make_reward_predictor(reward_predictor_network,
                                              log_dir,
                                              obs_shape,
                                              reward_predictor_logger,
                                              checkpoint_dir=pretrained_reward_predictor_dir)
    # Create a PrefBuffer that receives preferences from the PrefInterfaces and store them in PrefDBs
    pref_buffer = _load_or_create_pref_db(prefs_dir, max_prefs, reward_predictor_logger)
    pref_buffer.start_recv_thread(pref_pipe)
    minimum_prefs_met = False

    while True:
        pref_db_train, pref_db_val = pref_buffer.get_dbs()
        current_train_size = len(pref_db_train)
        current_val_size = len(pref_db_val)
        pref_db_size.value = current_train_size + current_val_size

        # If there has been an external trigger telling us to save preferences, do so, then reset it to 0 so we
        # won't save on subsequent iterations unless the flag is set again
        if save_prefs_flag.value == 1:
            _save_prefs(pref_buffer, log_dir, reward_predictor_logger)
            save_prefs_flag.value = 0

        # If there's been an external trigger telling this process to die, stop the receiving thread on the PrefBuffer,
        # and then return from the function
        if kill_processes_flag.value == 1:
            pref_buffer.stop_recv_thread()
            return
        if not train_reward:
            # There might be some circumstances where we just want to collect and save preferences (for which we need
            # to create a PrefDB using this function) but might not want to actually train a reward model.
            continue

        if not minimum_prefs_met:
            # Confirm that we have at least `num_initial_prefs` training examples, and 1 validation example
            if current_train_size < num_initial_prefs or current_val_size < 1:
                reward_predictor_logger.debug(f"Reward dbs of length ({len(pref_db_train)}, {len(pref_db_val)}), waiting for minimum length ({num_initial_prefs}, 1) to start training")
                time.sleep(1)
                continue
            else:
                minimum_prefs_met = True
        if reward_training_steps.value % database_refresh_interval == 0:
            pref_db_train, pref_db_val = pref_buffer.get_dbs()

        print(f"Training reward predictor on {current_train_size} preferences, testing on {current_val_size}, iteration {reward_training_steps.value }")
        reward_predictor_logger.info(f"Training reward predictor on {current_train_size} preferences, testing on {current_val_size}, iteration {reward_training_steps.value }")
        reward_predictor.train(pref_db_train, pref_db_val, validation_interval)
        reward_training_steps.value += 1
        if (save_model_flag.value == 1) or (reward_training_steps.value % ckpt_interval == 0):
            _save_prefs(pref_buffer, log_dir, reward_predictor_logger)
            reward_predictor.save()
            save_model_flag.value = 0


class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self,
                 env: Env,
                 reward_predictor_network: Callable = net_cnn,
                 train_reward: bool = True,
                 collect_prefs: bool = True,
                 segment_length: int = 40,
                 mp_context: str = 'spawn',
                 prefs_dir: str = None,
                 log_dir: str = "drlhp_logs/",
                 max_prefs_in_db: int = 10000,
                 obs_transform_func: Callable = None,
                 n_initial_training_steps: int = 50,
                 n_initial_prefs: int = 40,
                 pretrained_reward_predictor_dir: str = None,
                 reward_predictor_ckpt_interval: int = 10,
                 reward_predictor_refresh_interval: int = 10,
                 validation_interval: int = 10,
                 reward_database_refresh_interval: int = 1,
                 synthetic_prefs: bool = True,
                 max_pref_interface_segs: int = 50,
                 zoom_ratio: int = 4,
                 channels: int = 3,
                 env_wrapper_log_level: int = logging.INFO,
                 reward_predictor_log_level: int = logging.INFO,
                 pref_interface_log_level: int = logging.INFO
                 ):
        """
        A Wrapper that collects segments from the observations returned through its internal env's .step() function,
        and sends them to a PrefInterface that queries either humans or a synthetic reward oracle for preferences.

        It also manages creating and training a reward prediction network, using preferences stored in a PrefDB as
        training examples. When a minimum number of training steps has been reached, it loads the trained reward
        predictor network and starts using that as the returned reward, rather than underlying environment reward

        :param env: Underlying environment
        :param reward_predictor_network: Callable mapping between input obs and reward scalar
        :param train_reward: A boolean specifying whether or not the env should train a reward predictor
        :param collect_prefs: A boolean specifying whether or not the env should collect preferences in a PrefDB
        :param segment_length: How many observations long a segment should be before it's sent to the PrefInterface
        :param mp_context: A string specifying the multiprocessing context we want to use for this env's processes
        :param prefs_dir: An string path specifying where an existing set of PrefDBs are stored, if any exist
        :param log_dir: An string path specifying where logs and artifacts from this run should be saved
        :param max_prefs_in_db: The maximum number of preferences to store across both train and validation PrefDBs
        :param obs_transform_func: An optional transformation function to transform the observation returned by our
                                    internal environment into the observation that should be concatenated to form our
                                    segments (for example, if the underlying environment is a Dict space, your transform
                                    func could be obs['pov'])
        :param n_initial_training_steps: How many training steps should be performed before we switch to using a
                                        trained reward model as our returned environment reward
        :param n_initial_prefs: How many preferences to collect before starting to train our reward predictor


        :param pretrained_reward_predictor_dir: An string path specifying where a pretrained reward predictor
                                                is saved, if one exists

        :param reward_predictor_refresh_interval: Interval of reward predictor training steps on which to update the
                                                  reward predictor used by the env to calculate reward
        :param validation_interval: Interval of reward predictor training steps on which to perform validation
        :param reward_database_refresh_interval: Interval of reward predictor training steps on which to refresh the
                                                 PrefDBs used for training/validation

        :param reward_predictor_ckpt_interval: The interval of reward training steps on which we should automatically
                                               checkpoint the reward prediction model

        :param synthetic_prefs: If True, we use the reward function of the environment to calculate prefs; if False,
                                we query for human preferences using a GUI interface


        :param max_pref_interface_segs: The maximum number of segments that will be stored and paired with one another by
                                        the preference interface
        :param zoom_ratio: How much images should be zoomed when they're displayed to humans in the GUI (ignored if using
                            synthetic preferences)
        :param channels: The number of channels the images you'll show to humans will have. (Can't be inferred from
                         observation space shape because common usage involves a FrameStack wrapper, which will stack
                         frames along the channel dimension)
        :param env_wrapper_log_level: The log level of the logger corresponding to the wrapper as a whole
        :param reward_predictor_log_level: The log level of the logger corresponding to the reward predictor training function
        :param pref_interface_log_level: The log level of the logger used by the preference interface
        """

        # TODO maybe move creation of the Pref Interface inside rather than have it created externally?

        # Recommend using 'spawn' for non synthetic preferences and 'fork' for synthetic
        super(HumanPreferencesEnvWrapper, self).__init__(env)
        self.logger = logging.getLogger("HumanPreferencesEnvWrapper")
        self.logger.setLevel(env_wrapper_log_level)
        self.reward_predictor_log_level = reward_predictor_log_level
        self.pref_interface_log_level = pref_interface_log_level

        self.obs_shape = env.observation_space.shape

        self.preference_interface = PrefInterface(synthetic_prefs=synthetic_prefs,
                                                  max_segs=max_pref_interface_segs,
                                                  log_dir=log_dir,
                                                  channels=channels,
                                                  zoom=zoom_ratio)

        # Save a bunch of init parameters as wrapper properties
        self.synthetic_prefs = synthetic_prefs
        self.mp_context = mp_context
        self.train_reward = train_reward
        self.collect_prefs = collect_prefs
        self.segment_length = segment_length
        self.reward_predictor_network = reward_predictor_network
        self.pretrained_reward_predictor_dir = pretrained_reward_predictor_dir
        self.obs_transform_func = obs_transform_func
        self.prefs_dir = prefs_dir
        self.max_prefs = max_prefs_in_db
        self.n_initial_prefs = n_initial_prefs
        self.n_initial_training_steps = n_initial_training_steps
        self.log_dir = log_dir
        self.ckpt_interval = reward_predictor_ckpt_interval
        self.reward_predictor_refresh_interval = reward_predictor_refresh_interval
        self.val_interval = validation_interval
        self.reward_database_refresh_interval = reward_database_refresh_interval


        # Setting counter and status variables to initial values
        self.segments_collected = 0
        self.reward_predictor_n_train = 0
        self.using_reward_from_predictor = False
        self.force_return_true_reward = False
        self.collecting_segments = True
        self.last_true_reward = None

        # Create empty observation stack and new segment
        self.recent_obs_stack = []
        self.episode_segment = Segment()
        self.reward_predictor_checkpoint_dir = os.path.join(log_dir, 'reward_predictor_checkpoints')

        # Create Queues and Values to handle multiprocessing communication
        # TODO figure out how to make the mechanics of this work with larger Queues, so we don't drop segments on the
        # TODO ground due to timing issues
        self.seg_pipe = mp.get_context(self.mp_context).Queue(maxsize=5)
        self.pref_pipe = mp.get_context(self.mp_context).Queue(maxsize=1)
        self.remaining_pairs = mp.get_context(self.mp_context).Value('i', 0)
        self.pref_db_size = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_pref_interface_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.kill_reward_training_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.save_model_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.save_prefs_flag = mp.get_context(self.mp_context).Value('i', 0)
        self.reward_training_steps = mp.get_context(self.mp_context).Value('i', 0)

        # Create placeholder parameters for things that we'll initialize later
        self.pref_interface_proc = None
        self.reward_training_proc = None
        self.pref_buffer = None
        self.reward_predictor = None

        # If we want to collect preferences, we need to start a PrefInterface-running process
        if self.collect_prefs:
            self._start_pref_interface()
        # If we want to save preferences and/or train a reward model, we need to start a reward predictor training
        # process (which also handles creating a PrefDB in which preferences are stored/saved)
        if self.train_reward or self.collect_prefs:
            self._start_reward_predictor_training()

    def _start_pref_interface(self):
        self.pref_interface_proc = mp.get_context(self.mp_context).Process(target=_run_pref_interface, daemon=True,
                                                                           args=(self.preference_interface,
                                                                                 self.seg_pipe,
                                                                                 self.pref_pipe,
                                                                                 self.remaining_pairs,
                                                                                 self.kill_pref_interface_flag,
                                                                                 self.pref_interface_log_level))
        self.pref_interface_proc.start()

    def _start_reward_predictor_training(self):
        self.reward_training_proc = mp.get_context('spawn').Process(target=_train_reward_predictor, daemon=True,
                                                                   args=(self.reward_predictor_network,
                                                                         self.train_reward,
                                                                         self.pretrained_reward_predictor_dir,
                                                                         self.obs_shape,
                                                                         self.pref_pipe,
                                                                         self.pref_db_size,
                                                                         self.prefs_dir,
                                                                         self.max_prefs,
                                                                         self.ckpt_interval,
                                                                         self.n_initial_prefs,
                                                                         self.reward_training_steps,
                                                                         self.reward_database_refresh_interval,
                                                                         self.val_interval,
                                                                         self.kill_reward_training_flag,
                                                                         self.save_prefs_flag,
                                                                         self.save_model_flag,
                                                                         self.log_dir,
                                                                         self.reward_predictor_log_level))
        self.reward_training_proc.start()

    def _update_episode_segment(self, obs, reward, done):
        """
        Takes observation from most recent environment step and adds it to existing segment. If segment has reached
        desired length, finalize it and send it to the PrefInterface via seg_pipe

        :param obs: A (possibly stacked) observation from the underlying environment
        :param reward: Underlying environment reward (used for synthetic preferences)
        :param done: Whether the episode has terminated, in which case we should pad the rest of the segment and
                    then start a new one
        :return:
        """
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

    def _load_reward_predictor(self, model_load_dir):
        if self.reward_predictor is None:
            self.logger.info(f"Loading reward predictor from {model_load_dir}; will use its model reward now")
            self.reward_predictor = RewardPredictorEnsemble(
                core_network=self.reward_predictor_network,
                log_dir=self.log_dir,
                batchnorm=False,
                dropout=0.0,
                lr=7e-4,
                obs_shape=self.obs_shape,
                logger=self.logger)
        self.reward_predictor_n_train = self.reward_training_steps.value

        self.reward_predictor.init_network(model_load_dir)

    def step(self, action):
        # Check whether we have only just hit the point of the model having trained for enough steps

        minimum_training_steps_reached = self.reward_training_steps.value >= self.n_initial_training_steps
        sufficiently_trained = self.reward_predictor is None and minimum_training_steps_reached

        # Check whether we have an existing pretrained model we've not yet loaded in
        pretrained_model = self.reward_predictor is None and self.pretrained_reward_predictor_dir is not None

        # Check whether we should update our existing reward predictor with a new one because we've done enough
        # training steps since we last updated
        should_update_model = minimum_training_steps_reached and (self.reward_training_steps.value - self.reward_predictor_n_train > self.reward_predictor_refresh_interval)

        # If any of these things are true, we load a model in
        if sufficiently_trained or pretrained_model or should_update_model:
            if sufficiently_trained:
                self.logger.info("Model is sufficiently trained, switching to it for reward")
                model_load_dir = self.reward_predictor_checkpoint_dir
            elif should_update_model:
                self.logger.info("Updating model used for env reward")
                model_load_dir = self.reward_predictor_checkpoint_dir
            else:
                model_load_dir = self.pretrained_reward_predictor_dir
                self.logger.info("Loading pretrained model for env reward")
            self._load_reward_predictor(model_load_dir)
            self.using_reward_from_predictor = True
        obs, reward, done, info = self.env.step(action)

        if self.collecting_segments:
            self._update_episode_segment(obs, reward, done)

        if self.reward_predictor is not None and not self.force_return_true_reward:
            # If we have self.force_return_true_reward set, the environment will return the true
            # underlying reward (meant for evaluation purposes)
            predicted_reward = self.reward_predictor.reward(np.array([np.array(obs)]))[0]
            self.last_true_reward = reward
            return obs, predicted_reward, done, info
        else:
            return obs, reward, done, info

    def switch_to_true_reward(self):
        if not self.using_reward_from_predictor:
            raise Warning("Environment has no reward predictor loaded, and is thus returning true reward")
        elif self.force_return_true_reward:
            raise Warning("Environment already returning true reward, no change")
        else:
            self.using_reward_from_predictor = False
            self.force_return_true_reward = True

    def switch_to_predicted_reward(self):
        """
        Note: this only works to undo a prior forcing of true reward
        if a reward model is already loaded, it can't cause a reward model to exist if it isn't present
        """
        if not self.force_return_true_reward:
            raise Warning("Environment already returning predicted reward, no change")
        else:
            self.using_reward_from_predictor = True
            self.force_return_true_reward = False


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
        self.logger.debug("env.close() was called")
        self._cleanup_processes()
        self.env.close()
