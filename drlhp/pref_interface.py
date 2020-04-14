#!/usr/bin/env python

"""
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from copy import deepcopy
from itertools import combinations
from multiprocessing import Queue
from random import shuffle

import easy_tf_log
import numpy as np
import cv2
from drlhp.utils import VideoRenderer
from drlhp.utils import ForkedPdb


class PrefInterface:

    def __init__(self, synthetic_prefs, max_segs, log_dir, zoom, channels):
        self.vid_q = Queue()
        if not synthetic_prefs:
            self.renderer = VideoRenderer(vid_queue=self.vid_q,
                                          mode=VideoRenderer.restart_on_get_mode,
                                          zoom=zoom,
                                          channels=channels)
        else:
            self.renderer = None
        self.synthetic_prefs = synthetic_prefs
        self.zoom = zoom
        self.seg_idx = 0
        self.segments = []
        self.channels = channels
        self.tested_pairs = set()  # For O(1) lookup
        self.max_segs = max_segs
        easy_tf_log.set_dir(log_dir)

    def stop_renderer(self):
        if self.renderer:
            self.renderer.stop()

    def run(self, seg_pipe, pref_pipe):
        self.recv_segments(seg_pipe)

        while len(self.segments) < 2:
            time.sleep(5.0)
            self.recv_segments(seg_pipe)

        print("Preference interface has more than two segments, starting to test")
        while True:
            seg_pair = None
            while seg_pair is None:
                try:
                    seg_pair = self.sample_seg_pair()
                except IndexError:
                    print("Preference interface ran out of untested segments;"
                          "waiting...")
                    # If we've tested all possible pairs of segments so far,
                    # we'll have to wait for more segments
                    time.sleep(5.0)
                    self.recv_segments(seg_pipe)
            s1, s2 = seg_pair

            logging.info("Querying preference for segments %s and %s",
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
                logging.debug("Got segment")
            except queue.Empty:
                logging.debug("Segment queue empty")
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
        logging.debug(f"Num segments: {len(self.segments)}")
        logging.debug(f"Possible pairs: {len(list(deepcopy(possible_pairs)))}")
        logging.debug(f"Tested pairs: {len(self.tested_pairs)}")
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
        for t in range(seg_len):
            border = np.zeros((frame_shape[0], 10, 3), dtype=np.uint8)
            # -1 => show only the most recent frame of the 4-frame stack
            # TODO make this general across channels
            frame = np.hstack((s1.frames[t][:, :, -3:],
                               border,
                               s2.frames[t][:, :, -3:]))
            vid.append(frame)
        #TODO make this a parameter
        n_pause_frames = 12
        for _ in range(n_pause_frames):
            vid.append(np.copy(vid[-1]))
        self.vid_q.put(vid)

        while True:
            print("Choose between segments {} and {}: ".format(s1.hash, s2.hash))
            self.renderer.render()
            choice = input()
            # L = "I prefer the left segment"
            # R = "I prefer the right segment"
            # E = "I don't have a clear preference between the two segments"
            # "" = "The segments are incomparable"
            if choice == "L" or choice == "R" or choice == "E" or choice == "":
                break
            else:
                print("Invalid choice '{}'".format(choice))

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
