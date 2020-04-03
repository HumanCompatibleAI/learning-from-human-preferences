from gym import Wrapper

from multiprocessing import Queue


class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self, env, reward_pred_class):
        self.seg_pipe = Queue(maxsize=1)
        self.pref_pipe = Queue(maxsize=1)
        self.recent_obs_stack = [] # rolling list of last 4 observations
        self.reward_predictor = reward_pred_class(env)
        self.train_reward = True # A boolean for whether reward predictor is frozen or actively being trained

    def _pretrain_reward_predictor(self):
        x = 1
        # Sort of unclear if this should live on the env or not, since you need to have some policy for
        # determining how to take the steps used for pretraining, which makes it feel like more the purview of a runner than an EnvWrapper
        # We could have two modes for the env, one for "auto-train" and one for "collect preferences," and in the latter mode,
        # It just builds up a segment buffer, and then when we call "train_reward_predictor" from external

    def step(self, action):
        x = 1
        # Take action in environment
        # Get observations back
        # Add observation to segment pipe (or segment aggregator, since they aren't actually segments yet. Might also involve moving weird stacking logic in here)
        # If (some counter on segment aggregation has passed), run a training step on self.reward_predictor

    def _query_for_reward(self):
        while True:
            x = 1
            # Check if seg_pipe has things in it
            # If so, tell PrefInterface to query for human comparison
            # If not, sleep for some period based on debugging vs not

