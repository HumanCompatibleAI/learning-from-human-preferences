import queue
import random
import socket
import time

import gym
import numpy as np
import pyglet
import pdb
import sys

from drlhp.deprecated.a2c.common.atari_wrappers import wrap_deepmind
from drlhp.deprecated.a2c.common.misc_util import set_global_seeds
from drlhp.deprecated.a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from scipy.ndimage import zoom
from multiprocessing import Process


# https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S/(self._n - 1)
        else:
            return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


# Based on SimpleImageViewer in OpenAI gym
class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.channels = channels
            self.isopen = True

        assert arr.shape == (self.height, self.width, self.channels), \
            "You passed in an image with the wrong number shape"
        flipped_arr = np.flip(arr, axis=0)
        image = pyglet.image.ImageData(self.width, self.height,
                                       'RGB', flipped_arr.tobytes())
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class VideoRenderer:
    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, vid_queue, mode, fps=12, zoom=1, playback_speed=1, channels=3):
        assert mode == VideoRenderer.restart_on_get_mode or mode == VideoRenderer.play_through_mode
        self.mode = mode
        self.vid_queue = vid_queue
        self.channels = channels
        if self.channels == 1:
            self.zoom_factor = zoom
        else:
            self.zoom_factor = [zoom]*(self.channels-1) + [1]
        self.playback_speed = playback_speed
        self.stop_render = False
        self.current_frames = None
        self.v = None
        self.fps = fps
        self.sleep_time = 1/self.fps

    def stop(self):
        self.stop_render = True

    def render(self, frames):
        v = Im()
        t = 0
        while True:
            start = time.time()
            zoomed_frame = zoom(frames[t], self.zoom_factor, order=1)
            v.imshow(zoomed_frame)
            end = time.time()
            render_time = end - start
            if self.mode == VideoRenderer.play_through_mode:
                # Wait until having finished playing the current
                # set of frames. Then, stop, and get the most
                # recent set of frames.
                t += self.playback_speed
                if t >= len(frames):
                    v.close()
                    return
                else:
                    sleep_time = max(0, self.sleep_time-render_time)
                    time.sleep(sleep_time)
                    continue

    def get_queue_most_recent(self):
        # Make sure we at least get something
        item = self.vid_queue.get(block=True)
        while True:
            try:
                item = self.vid_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                break
        return item


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports


def profile_memory(log_path, pid):
    import memory_profiler
    def profile():
        with open(log_path, 'w') as f:
            # timeout=99999 is necessary because for external processes,
            # memory_usage otherwise defaults to only returning a single sample
            # Note that even with interval=1, because memory_profiler only
            # flushes every 50 lines, we still have to wait 50 seconds before
            # updates.
            memory_profiler.memory_usage(pid, stream=f,
                                         timeout=99999, interval=1)
    p = Process(target=profile, daemon=True)
    p.start()
    return p


def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size



class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin