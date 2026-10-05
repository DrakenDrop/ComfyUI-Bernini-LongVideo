import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolution import output_geometry, prepare_frames, remove_padding
from core import render_timeline


class ResolutionTests(unittest.TestCase):
    def test_presets_landscape_portrait_square(self):
        for w, h, preset, expected in [(1920, 1080, '480p', (864, 486)),
                                      (1920, 1080, '720p', (1280, 720)),
                                      (1080, 1920, '720p', (720, 1280)),
                                      (1440, 1080, '480p', (640, 480)),
                                      (1024, 1024, '480p', (480, 480)),
                                      (1920, 1080, '1080p', (1920, 1080))]:
            g = output_geometry(w, h, preset)
            self.assertEqual((g.width, g.height), expected)
            self.assertEqual(g.width * h, g.height * w)
            self.assertEqual(g.width % 2, 0)
            self.assertEqual(g.height % 2, 0)
            self.assertEqual(g.model_width % 16, 0)
            self.assertEqual(g.model_height % 16, 0)

    def test_unusual_ratios_remain_exact(self):
        for w, h in [(853, 480), (1000, 333), (17, 31), (4096, 2160)]:
            g = output_geometry(w, h, '480p')
            self.assertEqual(g.width * h, g.height * w)

    def test_padding_round_trip_preserves_every_original_pixel(self):
        source = np.random.default_rng(3).random((2, 13, 21, 3), dtype=np.float32)
        g = output_geometry(21, 13, 'source')
        padded = prepare_frames(source, g)
        self.assertEqual(padded.shape, (2, 16, 32, 3))
        np.testing.assert_array_equal(remove_padding(padded, g), source)
        np.testing.assert_array_equal(padded[:, 0, 0], source[:, 0, 0])
        np.testing.assert_array_equal(padded[:, -1, -1], source[:, -1, -1])

    def test_resize_keeps_full_frame_and_orientation(self):
        source = np.zeros((1, 90, 160, 3), dtype=np.float32)
        source[:, :20, :20, 0] = 1
        source[:, -20:, -20:, 2] = 1
        g = output_geometry(160, 90, '360p')
        result = remove_padding(prepare_frames(source, g), g)
        self.assertEqual(result.shape, (1, 360, 640, 3))
        np.testing.assert_array_equal(result[0, 0, 0], [1, 0, 0])
        np.testing.assert_array_equal(result[0, -1, -1], [0, 0, 1])
        np.testing.assert_array_equal(result[0, 180, 320], [0, 0, 0])

    def test_timeline_overlap_has_output_dimensions_not_canvas_dimensions(self):
        source = np.random.default_rng(4).random((12, 13, 21, 3), dtype=np.float32)
        g = output_geometry(21, 13, 'source')
        output, _ = render_timeline(source, 12, 9, 3, 'crossfade',
                                    lambda clip, i: remove_padding(prepare_frames(clip, g), g))
        np.testing.assert_allclose(output, source, atol=1e-7)

    def test_cancel_propagates_during_resize(self):
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            prepare_frames(np.zeros((1, 9, 16, 3)), output_geometry(16, 9, '480p'), cancel)

    def test_legacy_custom_and_invalid_geometry(self):
        g = output_geometry(1920, 1080, 'custom', 480, 832)
        self.assertEqual((g.width, g.height, g.left, g.top), (480, 832, 0, 0))
        for args in [(0, 1080, '480p'), (1920, 1080, 'invalid'),
                     (1920, 1080, 'custom', 481, 832), (9999, 1, '720p')]:
            with self.assertRaises(ValueError): output_geometry(*args)


if __name__ == '__main__': unittest.main()
