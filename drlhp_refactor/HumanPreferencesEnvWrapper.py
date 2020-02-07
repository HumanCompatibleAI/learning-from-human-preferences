from gym import Wrapper

from multiprocessing import Queue


class HumanPreferencesEnvWrapper(Wrapper):
    def __init__(self, env):
        seg_pipe = Queue(maxsize=1)
        pref_pipe = Queue(maxsize=1)