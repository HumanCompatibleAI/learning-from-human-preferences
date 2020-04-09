import gym
from drlhp.pref_interface import PrefInterface
from drlhp.reward_predictor_core_network import net_cnn
from stable_baselines.common.atari_wrappers import FrameStack
from HumanPreferencesEnvWrapper import HumanPreferencesEnvWrapper


if __name__ == "__main__":
    nstack = 4

    pi = PrefInterface(synthetic_prefs=True,
                       max_segs=100,
                       log_dir="drlhp_logs",
                       channels=3,
                       zoom=4)
    env = gym.make("CarRacing-v0")
    stacked_env = FrameStack(env, n_frames=nstack)
    import pdb; pdb.set_trace()
    wrapper = HumanPreferencesEnvWrapper(env,
                                         reward_predictor_network=net_cnn,
                                         preference_interface=pi)

    ## Gather Initial Preferences
    ## >> Create a