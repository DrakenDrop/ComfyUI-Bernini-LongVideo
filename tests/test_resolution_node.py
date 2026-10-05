"""Verify the real node's resize/pad/decode wiring with CPU inference doubles."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, array): self.array = array
    @property
    def shape(self): return self.array.shape
    def __len__(self): return len(self.array)
    def __getitem__(self, key): return Tensor(self.array[key])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class ResolutionNodeTests(unittest.TestCase):
    def test_node_pads_before_conditioning_and_unpads_before_assembly(self):
        self.check_node(12, 16, 30, 9, 3, 13, 2)

    def test_short_source_retains_81_frames_and_aligned_comparison(self):
        self.check_node(80, 16, 10, 81, 17, 81, 1)

    def test_duration_cap_keeps_endpoint_frame(self):
        self.check_node(160, 16, 5, 81, 17, 81, 1)

    def test_multichunk_returns_161_frames_with_matching_metadata(self):
        self.check_node(200, 16, 10, 81, 17, 161, 3)

    def test_unlimited_length_preserves_fractional_fps(self):
        self.check_node(82, 23.976, 0, 81, 17, 85, 2)

    def test_30_seconds_retains_481_frames(self):
        self.check_node(500, 16, 30, 81, 17, 481, 8)

    def test_aligned_source_is_not_extended(self):
        self.check_node(81, 16, 5, 81, 17, 81, 1)

    def test_single_frame_is_not_extended_to_requested_duration(self):
        self.check_node(1, 16, 30, 81, 17, 1, 1)

    def check_node(self, available, fps, seconds, chunk, overlap, expected, expected_chunks):
        source = np.random.default_rng(12).random((available, 13, 21, 3), dtype=np.float32)
        calls = []

        class Conditioning:
            @classmethod
            def execute(cls, pos, neg, vae, width, height, length, batch, source_video, **kwargs):
                calls.append((width, height, length))
                self.assertEqual(source_video.shape[1:3], (16, 32))
                return pos, neg, source_video

        class Sampler:
            def sample(self, *args): return (args[-1],)

        class Decoder:
            def decode(self, vae, latent, *args): return (latent,)

        package = types.ModuleType('bernini_resize_test')
        package.__path__ = [str(ROOT)]
        comfy = types.ModuleType('comfy')
        comfy.__path__ = []
        management = types.ModuleType('comfy.model_management')
        management.throw_exception_if_processing_interrupted = lambda: None
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = MagicMock()
        comfy.model_management, comfy.utils = management, utils
        bernini = types.SimpleNamespace(BerniniConditioning=Conditioning)
        modules = {'bernini_resize_test': package, 'torch': types.SimpleNamespace(from_numpy=Tensor, isclose=np.isclose),
                   'comfy': comfy, 'comfy.model_management': management, 'comfy.utils': utils,
                   'comfy_extras': types.ModuleType('comfy_extras'), 'comfy_extras.nodes_bernini': bernini,
                   'comfy_extras.nodes_custom_sampler': types.SimpleNamespace(SamplerCustom=Sampler),
                   'nodes': types.SimpleNamespace(VAEDecode=Decoder, VAEDecodeTiled=Decoder)}
        spec = importlib.util.spec_from_file_location('bernini_resize_test.nodes', ROOT/'nodes.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            output, report, frame_count, output_fps, comparison = module.BerniniLongV2V().run(
                model_high=None, model_low=None, positive=[], negative=[], vae=None,
                source_video=Tensor(source), sampler=None, sigmas_high=np.array([1., .5]),
                sigmas_low=np.array([.5, 0.]), width=480, height=832, fps=fps, max_seconds=seconds,
                chunk_frames=chunk, overlap=overlap, blend_mode='crossfade', seed=1, seed_mode='fixed',
                cfg_high=1, cfg_low=1, tiled_encode=False, tiled_decode=False, tile_size=512,
                ref_max_size=512, resolution='source')
        expected_source = source[np.minimum(np.arange(expected), available - 1)]
        np.testing.assert_allclose(output.numpy(), expected_source, atol=1e-7)
        np.testing.assert_array_equal(comparison.numpy(), expected_source)
        self.assertEqual(len(calls), expected_chunks)
        self.assertEqual(frame_count, expected)
        self.assertEqual(frame_count, len(output))
        self.assertEqual(frame_count, len(comparison))
        self.assertIsInstance(frame_count, int)
        self.assertEqual(output_fps, fps)
        self.assertIsInstance(output_fps, float)
        self.assertTrue(all((length - 1) % 4 == 0 for _, _, length in calls))
        report = json.loads(report)
        self.assertEqual(report['frames'], frame_count)
        self.assertEqual(report['fps'], output_fps)
        self.assertEqual(report['input_frames'], available)
        self.assertEqual(report['max_seconds'], seconds)
        self.assertEqual(report['seconds'], expected / fps)
        self.assertEqual(report['frame_span_seconds'], (expected - 1) / fps)
        self.assertEqual(report['source_frames_used'], min(available, expected))
        self.assertEqual(report['tail_padding_frames'], max(0, expected - available))
        self.assertEqual(report['output_size'], [21, 13])
        self.assertEqual(report['model_canvas'], [32, 16])


if __name__ == '__main__': unittest.main()
