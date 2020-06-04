import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import pytest
import gym
from stable_baselines.common.atari_wrappers import FrameStack
from drlhp.pref_interface import PrefInterface
from drlhp.reward_predictor_core_network import net_cnn
from drlhp.HumanPreferencesEnvWrapper import HumanPreferencesEnvWrapper
import logging
from realistic_benchmarks.wrappers import ActionMeaningsWrapper
import time
from random import randint

# Create dummy environment


class DummyEnv(gym.Env):
    """
    A simplistic class that lets us mock up a gym Environment that is sufficient for our purposes
    without actually going through the whole convoluted registration process.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Dict):
            assert isinstance(action, dict)
        return self.observation_space.sample(), randint(0, 10), None, None

    def reset(self):
        return self.observation_space.sample()

@pytest.fixture
def dummy_pixel_env():
    env = DummyEnv(observation_space=gym.spaces.Box(shape=(64, 64, 3), low=0, high=255), action_space=gym.spaces.Discrete(25))
    env = FrameStack(env, n_frames=4)
    env = ActionMeaningsWrapper(env, env_type="flattened")
    return env




def test_segment_creation(dummy_pixel_env):
    # Test that we can build up segments from environment steps

    drlhp_env = HumanPreferencesEnvWrapper(dummy_pixel_env,
                                           reward_predictor_network=net_cnn,
                                           segment_length=10,
                                           mp_context='spawn',
                                           train_reward=False, #Just collect segments for this test
                                           collect_prefs=False,
                                           prefs_dir=None,
                                           log_dir="testing_logs")

    drlhp_env.reset()

    for i in range(30):
        _ = drlhp_env.step(drlhp_env.action_space.sample())
    assert drlhp_env.segments_collected == 3
    drlhp_env.close()


def test_automatic_preference_collection(dummy_pixel_env):
    # Test that we can collect preference and that they're successfully added to the PrefDB

    drlhp_env = HumanPreferencesEnvWrapper(dummy_pixel_env,
                                           reward_predictor_network=net_cnn,
                                           segment_length=10,
                                           mp_context='spawn',
                                           train_reward=False,
                                           collect_prefs=True, # Do collect preferences into a pref_db, but don't train the reward predictor
                                           prefs_dir=None,
                                           log_dir="testing_logs")

    drlhp_env.reset()
    time.sleep(5)
    for i in range(100):
        _ = drlhp_env.step(drlhp_env.action_space.sample())
    time.sleep(5)
    assert drlhp_env.pref_db_size.value > 1
    print(f"Size of pref DB: {drlhp_env.pref_db_size.value}")
    drlhp_env.close()


def test_reward_training(dummy_pixel_env):
    # Test that we can take at least 5 training steps of the reward predictor, and successfully switch to
    # using that for our reward once that many steps are taken

    drlhp_env = HumanPreferencesEnvWrapper(dummy_pixel_env,
                                           reward_predictor_network=net_cnn,
                                           segment_length=5,
                                           mp_context='spawn',
                                           n_initial_training_steps=5,
                                           n_initial_prefs=4,
                                           train_reward=True,
                                           collect_prefs=True,
                                           reward_predictor_log_level=logging.DEBUG,
                                           env_wrapper_log_level=logging.DEBUG,
                                           prefs_dir=None,
                                           log_dir="testing_logs")

    drlhp_env.reset()
    time.sleep(5)
    for i in range(100):
        _ = drlhp_env.step(drlhp_env.action_space.sample())
    time.sleep(10)
    _ = drlhp_env.step(drlhp_env.action_space.sample())
    print(f"Training steps taken: {drlhp_env.reward_training_steps.value}")
    assert drlhp_env.using_reward_from_predictor
    drlhp_env.close()

