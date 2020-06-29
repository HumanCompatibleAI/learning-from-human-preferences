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
import logging
from functools import partial

def handler(signum, frame):
    print("Got no response, replaying segment video")
    raise IOError("No response!")

def fake_log(message, message_level, log_level, log_name):
    if message_level >= log_level:
        level_name = logging.getLevelName(log_level)
        print(f"{level_name} - {log_name} - {message}")

class PrefInterface:

    def __init__(self, synthetic_prefs, max_segs, log_dir, zoom=4, channels=3,
                 min_segments_to_test=10, n_pause_frames=4,
                 user_response_timeout=3):
        if not synthetic_prefs:
            self.vid_q = mp.get_context('spawn').Queue()
            self.renderer = VideoRenderer(vid_queue=self.vid_q,
                                          mode=VideoRenderer.play_through_mode,
                                          zoom=zoom,
                                          channels=channels)
        else:
            self.renderer = None
        self.min_segments_to_test = min_segments_to_test
        self.synthetic_prefs = synthetic_prefs
        self.zoom = zoom
        self.seg_idx = 0
        self.segments = []
        self.channels = channels
        self.max_segs = max_segs
        self.tested_pairs = set()
        self.n_pause_frames = n_pause_frames
        self.user_response_timeout = user_response_timeout
        easy_tf_log.set_dir(log_dir)

    def stop_renderer(self):
        if self.renderer:
            self.renderer.stop()

    def run(self, seg_pipe, pref_pipe, kill_processes, log_level):
        log_name = 'pref_interface.run'
        pref_interface_fake_log = partial(fake_log, log_name=log_name, log_level=log_level)
        self.recv_segments(seg_pipe)
        while len(self.segments) < self.min_segments_to_test:
            if kill_processes.value == 1:
                pref_interface_fake_log(
                    "Pref interface got kill signal before collecting enough segments, exiting",
                    logging.INFO)
                return
            pref_interface_fake_log(f"Pref interface only has {len(self.segments)} segments, waiting for {self.min_segments_to_test}, sleeping", logging.DEBUG)
            # This sleep time is load bearing, because if you sleep for too long you'll drop more segments on the ground due to
            # not re-querying the segment pipe
            time.sleep(0.05)
            self.recv_segments(seg_pipe)

        pref_interface_fake_log(f"Preference interface has at least {self.min_segments_to_test} segments, starting to test", logging.INFO)
        while True and kill_processes.value == 0:
            if kill_processes.value == 1:
                pref_interface_fake_log("Pref interface got kill signal, exiting", logging.INFO)
                return


            seg_pair = self.sample_seg_pair()

            s1, s2 = seg_pair

            pref_interface_fake_log(f"Querying preference for segments {s1.hash} and {s2.hash}", logging.DEBUG)

            if not self.synthetic_prefs:
                pref = self.ask_user(s1, s2, log_func=pref_interface_fake_log)
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
        Sample a random pair of segments (which may already have been tested).
        """
        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        ind1, ind2 = segment_idxs[:2]
        return self.segments[ind1], self.segments[ind2]

    def ask_user(self, s1, s2, log_func):
        vid = []
        seg_len = len(s1)
        frame_shape = s1.frames[0][:, :, -1].shape
        log_func(f"Creating user-facing video of length {seg_len}", logging.DEBUG)
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
                log_func(f"Invalid choice {choice}", logging.WARNING)
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
