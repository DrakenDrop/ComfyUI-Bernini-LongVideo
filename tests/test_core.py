import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import plan_windows, render_timeline, target_frames, input_frame_rate, source_frame_indices


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

    def test_fps_conversion_keeps_motion_timestamps(self):
        for source_fps in (8, 23.976, 30, 60):
            available = round(source_fps * 10)
            indices, valid, padding = source_frame_indices(available, source_fps, 16, 10)
            self.assertEqual(len(indices), 161)
            self.assertEqual(indices[16], round(source_fps))
            self.assertEqual(indices[80], round(source_fps * 5))
            source_time = indices[:valid] / source_fps
            target_time = np.arange(valid) / 16
            self.assertLessEqual(np.max(np.abs(source_time - target_time)), 1 / source_fps + 1e-9)
            self.assertTrue(0 <= padding <= 3)
            np.testing.assert_array_equal(indices[valid:], np.repeat(indices[valid - 1], padding))

    def test_duration_cap_applies_after_resampling(self):
        indices, valid, padding = source_frame_indices(1800, 60, 16, 5)
        self.assertEqual(len(indices), 81)
        self.assertEqual(indices[-1], 300)
        self.assertEqual((valid, padding), (81, 0))

    def test_short_native_source_does_not_expand_to_requested_duration(self):
        indices, valid, padding = source_frame_indices(120, 30, 16, 10)
        self.assertEqual((len(indices), valid, padding), (65, 64, 1))
        self.assertEqual(indices[16], 30)

    def test_metadata_uses_loaded_fps_and_overrides_manual(self):
        self.assertEqual(input_frame_rate({'source_fps': 60, 'loaded_fps': 16}, 30, 16),
                         (16, 'video_info.loaded_fps'))
        self.assertEqual(input_frame_rate(None, 30, 16), (30, 'input_fps'))
        self.assertEqual(input_frame_rate(None, 0, 16), (16, 'assumed_from_output_fps'))
        indices, _, _ = source_frame_indices(81, 16, 16, 5)
        np.testing.assert_array_equal(indices, np.arange(81))

    def test_invalid_rate_metadata_is_not_silently_assumed(self):
        for info in ({'source_fps': 30}, {'loaded_fps': 0}, {'loaded_fps': 'bad'},
                     {'loaded_fps': float('nan')}, []):
            with self.assertRaises(ValueError): input_frame_rate(info, 30, 16)
        for rate in (-1, float('inf'), 'bad'):
            with self.assertRaises(ValueError): input_frame_rate(None, rate, 16)
        for args in ((10, 0, 16, 5), (10, 30, float('nan'), 5), (0, 30, 16, 5)):
            with self.assertRaises(ValueError): source_frame_indices(*args)

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
