#!/usr/bin/env python

"""
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from copy import deepcopy
from itertools import combinations
import multiprocessing as mp
from random import shuffle
import sys
from select import select
import easy_tf_log
import numpy as np
from drlhp.utils import VideoRenderer

def handler(signum, frame):
    print("Got no response, replaying segment video")
    raise IOError("No response!")

class PrefInterface:

    def __init__(self, synthetic_prefs, max_segs, log_dir, zoom=4, channels=3,
                 log_level=logging.INFO, min_segments_to_test=2, max_idle_cycles=15, n_pause_frames=4,
                 user_response_timeout=3):
        if not synthetic_prefs:
            self.vid_q = mp.get_context('spawn').Queue()
            self.renderer = VideoRenderer(vid_queue=self.vid_q,
                                          mode=VideoRenderer.play_through_mode,
                                          zoom=zoom,
                                          channels=channels)
        else:
            self.renderer = None
        self.logger = logging.getLogger("PrefInterface")
        self.logger.setLevel(log_level)
        self.min_segments_to_test = min_segments_to_test
        self.synthetic_prefs = synthetic_prefs
        self.zoom = zoom
        self.seg_idx = 0
        self.segments = []
        self.channels = channels
        self.remaining_possible_pairs = 0
        self.tested_pairs = set()  # For O(1) lookup
        self.max_segs = max_segs
        self.max_idle_cycles = max_idle_cycles
        self.n_pause_frames = n_pause_frames
        self.user_response_timeout = user_response_timeout
        easy_tf_log.set_dir(log_dir)

    def stop_renderer(self):
        if self.renderer:
            self.renderer.stop()

    def run(self, seg_pipe, pref_pipe, remaining_pairs, kill_processes):
        self.recv_segments(seg_pipe)
        idle_cycles = 0
        while len(self.segments) < self.min_segments_to_test:
            if kill_processes.value == 1:
                #print("Pref interface got kill signal, exiting")
                self.logger.info("Pref interface got kill signal, exiting")
                return
            print(f"Pref interface only has {len(self.segments)} segments, waiting for {self.min_segments_to_test}, sleeping")
            self.logger.debug(f"Pref interface only has {len(self.segments)} segments, waiting for {self.min_segments_to_test}, sleeping")
            # This sleep time is load bearing, because if you sleep for too long you'll drop more segments on the ground due to
            # not re-querying the segment pipe
            time.sleep(0.05)
            self.recv_segments(seg_pipe)

        self.logger.debug("Preference interface has more than two segments, starting to test")
        while True and kill_processes.value == 0:
            seg_pair = None
            while seg_pair is None:
                if kill_processes.value == 1:
                    self.logger.info("Pref interface got kill signal, exiting")
                    return
                try:
                    seg_pair = self.sample_seg_pair()
                    remaining_pairs.value = self.remaining_possible_pairs
                except IndexError:
                    if idle_cycles > self.max_idle_cycles:
                        self.logger.info("Preference interface has gone idle, exiting")
                        return
                    self.logger.debug("Preference interface ran out of untested segments;"
                          "waiting...")
                    # If we've tested all possible pairs of segments so far,
                    # we'll have to wait for more segments
                    idle_cycles += 1
                    time.sleep(1.0)
                    self.recv_segments(seg_pipe)
            s1, s2 = seg_pair

            self.logger.debug("Querying preference for segments %s and %s",
                          s1.hash, s2.hash)

            if not self.synthetic_prefs:
                pref = self.ask_user(s1, s2)
            else:
                if sum(s1.rewards) > sum(s2.rewards):
                    pref = (1.0, 0.0)
                elif sum(s1.rewards) < sum(s2.rewards):
                    pref = (0.0, 1.0)
                else:
                    pref = (0.5, 0.5)

                time.sleep(0.25)

            if pref is not None:
                # We don't need the rewards from this point on, so just send
                # the frames
                #print("PrefInterface sending preference to pref pipe!")
                pref_pipe.put((s1.frames, s2.frames, pref))
            # If pref is None, the user answered "incomparable" for the segment
            # pair. The pair has been marked as tested; we just drop it.

            self.recv_segments(seg_pipe)

    def recv_segments(self, seg_pipe):
        """
        Receive segments from `seg_pipe` into circular buffer `segments`.
        """
        max_wait_seconds = 0.5
        start_time = time.time()
        n_recvd = 0
        while time.time() - start_time < max_wait_seconds:
            try:
                segment = seg_pipe.get(block=True, timeout=max_wait_seconds)
            except queue.Empty:
                return

            if len(self.segments) < self.max_segs:
                self.segments.append(segment)
            else:
                self.segments[self.seg_idx] = segment
                self.seg_idx = (self.seg_idx + 1) % self.max_segs
            n_recvd += 1
        easy_tf_log.tflog('segment_idx', self.seg_idx)
        easy_tf_log.tflog('n_segments_rcvd', n_recvd)
        easy_tf_log.tflog('n_segments', len(self.segments))

    def sample_seg_pair(self):
        """
        Sample a random pair of segments which hasn't yet been tested.
        """
        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        possible_pairs = combinations(segment_idxs, 2)
        self.remaining_possible_pairs = len(list(deepcopy(possible_pairs))) - len(self.tested_pairs)
        # print(f"Num segments: {len(self.segments)}")
        # print(f"Remaining pairs: {self.remaining_possible_pairs}")
        # print(f"Tested pairs: {len(self.tested_pairs)}")
        for i1, i2 in possible_pairs:
            i1, i2 = min(i1, i2), max(i1, i2)
            # these should now always be in a canonical order
            s1, s2 = self.segments[i1], self.segments[i2]
            if (s1.hash, s2.hash) not in self.tested_pairs:
                self.tested_pairs.add((s1.hash, s2.hash))
                return s1, s2
        raise IndexError("No segment pairs yet untested")

    def ask_user(self, s1, s2):
        vid = []
        seg_len = len(s1)
        frame_shape = s1.frames[0][:, :, -1].shape
        self.logger.debug(f"Creating user-facing video of length {seg_len}")
        for t in range(seg_len):
            border = np.zeros((frame_shape[0], 10, self.channels), dtype=np.uint8)
            # -1 => show only the most recent frame of the 4-frame stack
            frame = np.hstack((s1.frames[t][:, :, -self.channels:],
                               border,
                               s2.frames[t][:, :, -self.channels:]))
            vid.append(frame)
        for _ in range(self.n_pause_frames):
            vid.append(np.copy(vid[-1]))
        print(f"Choose between segments {s1.hash} (L) and {s2.hash} (R). (E) for equal. Video length: {len(vid)}: ")
        while True:
            self.renderer.render(vid)
            user_input, _, _ = select([sys.stdin], [], [], self.user_response_timeout)
            if user_input:
                choice = sys.stdin.readline().lstrip().rstrip()
            else:
                continue
            # L = "I prefer the left segment"
            # R = "I prefer the right segment"
            # E = "I don't have a clear preference between the two segments"
            # "" = "The segments are incomparable"
            if choice == "L" or choice == "R" or choice == "E" or choice == "":
                break
            else:
                self.logger.warning("Invalid choice {}".format(choice))
                continue

        print("Got preference!")
        if choice == "L":
            pref = (1.0, 0.0)
        elif choice == "R":
            pref = (0.0, 1.0)
        elif choice == "E":
            pref = (0.5, 0.5)
        elif choice == "":
            pref = None

        self.vid_q.put([np.zeros(vid[0].shape, dtype=np.uint8)])

        return pref
