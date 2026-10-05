import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import plan_windows, render_timeline, target_frames


class TimelineTests(unittest.TestCase):
    def test_30_seconds(self):
        windows = plan_windows(target_frames(1000, 16, 30))
        self.assertEqual(len(windows), 8)
        self.assertEqual(windows[-1].end, 481)
        self.assertEqual(windows[-1].padded, 33)
        self.assertTrue(all((w.padded - 1) % 4 == 0 for w in windows))
        self.assertTrue(all(w.padded <= 81 for w in windows))

    def test_identity_renderer_no_dropped_or_repeated_timestamps(self):
        for total in [1, 2, 4, 5, 64, 65, 80, 81, 82, 120, 121, 479, 480, 481]:
            for overlap in [0, 1, 17, 40, 80]:
                for blend in ['crossfade', 'cut']:
                    src = np.arange(total, dtype=np.float32).reshape(total,1,1,1).repeat(3, axis=-1)
                    out, _ = render_timeline(src, total, 81, overlap, blend, lambda x, i: x)
                    np.testing.assert_allclose(out, src, rtol=1e-6)

    def test_padding_repeats_last_frame_and_is_trimmed(self):
        seen = []
        src = np.arange(82, dtype=np.float32).reshape(82,1,1,1).repeat(3,axis=-1)
        def render(x, i):
            seen.append(x.copy()); return x
        out, _ = render_timeline(src,82,81,17,'cut',render)
        self.assertEqual(len(seen[-1]), 21)
        np.testing.assert_array_equal(seen[-1][-4:], np.repeat(src[-1:],4,axis=0))
        np.testing.assert_array_equal(out,src)

    def test_blend_changes_only_overlap(self):
        src = np.zeros((145,1,1,3),dtype=np.float32)
        out,_ = render_timeline(src,145,81,17,'crossfade',lambda x,i: x + i)
        self.assertTrue((out[:64] == 0).all())
        self.assertTrue((out[81:] == 1).all())
        self.assertTrue((np.diff(out[64:81,0,0,0]) > 0).all())

    def test_limits_and_short_sources(self):
        self.assertEqual(target_frames(1000,16,5),81)
        self.assertEqual(target_frames(1000,16,10),161)
        self.assertEqual(target_frames(1000,16,30),481)
        self.assertEqual(target_frames(81,16,30),81)
        self.assertEqual(target_frames(80,16,30),81)
        self.assertEqual(target_frames(1000,16,0),1001)
        self.assertEqual(target_frames(1000,23.976,5),121)
        self.assertEqual(target_frames(1000,16,0.01),5)
        for args in [(0,16,30),(100,0,30),(100,16,-1),(100,float('nan'),30)]:
            with self.assertRaises(ValueError): target_frames(*args)

    def test_short_source_alignment_adds_at_most_three_frames(self):
        for available in range(1, 500):
            total = target_frames(available, 16, 0)
            self.assertEqual((total - 1) % 4, 0)
            self.assertTrue(0 <= total - available <= 3)

    def test_invalid_windows_and_renderer(self):
        for args in [(0,81,17),(10,80,17),(10,81,81),(10,81,-1)]:
            with self.assertRaises(ValueError): plan_windows(*args)
        src=np.zeros((90,1,1,3),dtype=np.float32)
        with self.assertRaises(ValueError):
            render_timeline(src,90,81,17,'crossfade',lambda x,i: x[:1])

    def test_cancel_propagates(self):
        def stop(x,i): raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            render_timeline(np.zeros((90,1,1,3)),90,81,17,'crossfade',stop)


if __name__ == '__main__': unittest.main()
