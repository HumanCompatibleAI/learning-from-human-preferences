#!/usr/bin/env python3
import unittest
from multiprocessing import Queue

import numpy as np
import termcolor

from drlhp.pref_db import Segment
from drlhp.pref_interface import PrefInterface


def send_segments(n_segments, seg_pipe):
    frame_stack = np.zeros((84, 84, 4))
    for i in range(n_segments):
        segment = Segment()
        for _ in range(25):
            segment.append(frame=frame_stack, reward=0)
        segment.finalise(seg_id=i)
        seg_pipe.put(segment)


class TestPrefInterface(unittest.TestCase):
    def setUp(self):
        self.p = PrefInterface(synthetic_prefs=True, max_segs=1000,
                               log_dir='/tmp')
        termcolor.cprint(self._testMethodName, 'red')

    def test_recv_segments(self):
        """
        Check that segments are stored correctly in the circular buffer.
        """
        pi = PrefInterface(synthetic_prefs=True, max_segs=5, log_dir='/tmp')
        pipe = Queue()
        for i in range(5):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [0, 1, 2, 3, 4])
        for i in range(5, 8):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [5, 6, 7, 3, 4])
        for i in range(8, 11):
            pipe.put(i)
            pi.recv_segments(pipe)
        np.testing.assert_array_equal(pi.segments, [10, 6, 7, 8, 9])


if __name__ == '__main__':
    unittest.main()
