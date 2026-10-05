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
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class ResolutionNodeTests(unittest.TestCase):
    single_model = False
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

    def test_native_30fps_source_is_resampled_before_conditioning(self):
        expected_indices = np.minimum(np.floor(np.minimum(np.arange(161), 159) * 30 / 16 + .5), 299).astype(int)
        self.check_node(300, 16, 10, 81, 17, 161, 3,
                        video_info={'loaded_fps': 30, 'source_fps': 30},
                        expected_indices=expected_indices, expected_tail=1, expected_input_fps=30)

    def test_loader_force_rate_is_not_applied_twice(self):
        self.check_node(81, 16, 5, 81, 17, 81, 1,
                        video_info={'loaded_fps': 16, 'source_fps': 60}, input_fps=60)

    def test_manual_input_fps_preserves_a_full_five_seconds(self):
        self.check_node(600, 16, 5, 81, 17, 81, 1, input_fps=60,
                        expected_indices=np.floor(np.arange(81) * 60 / 16 + .5).astype(int),
                        expected_tail=0, expected_input_fps=60)

    def check_node(self, available, fps, seconds, chunk, overlap, expected, expected_chunks,
                   video_info=None, input_fps=0, expected_indices=None, expected_tail=None,
                   expected_input_fps=None, bad_config=None, bad_sigmas=None):
        source = np.random.default_rng(12).random((available, 13, 21, 3), dtype=np.float32)
        calls = []
        sampler_calls = []
        refs = Tensor(np.zeros((1, 13, 21, 3), dtype=np.float32))

        class Conditioning:
            @classmethod
            def execute(cls, pos, neg, vae, width, height, length, batch, source_video, **kwargs):
                calls.append((width, height, length))
                self.assertEqual(source_video.shape[1:3], (16, 32))
                self.assertIs(kwargs['reference_images']['reference_image_0'], refs)
                return pos, neg, source_video

        class Sampler:
            def sample(self, *args):
                sampler_calls.append(args)
                return (args[-1],)

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
            args = dict(positive=[], negative=[], vae=None,
                source_video=Tensor(source), sampler=None, width=480, height=832, fps=fps, max_seconds=seconds,
                chunk_frames=chunk, overlap=overlap, blend_mode='crossfade', seed=1, seed_mode='fixed',
                tiled_encode=False, tiled_decode=False, tile_size=512, reference_images=refs,
                ref_max_size=512, resolution='source', video_info=video_info, input_fps=input_fps)
            if self.single_model:
                config = {'image_model': 'wan2.1', 'model_type': 't2v', 'dim': 1536, 'in_dim': 16, 'out_dim': 16}
                if bad_config is not None: config.update(bad_config)
                model = types.SimpleNamespace(model=types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config=config)))
                sigmas = Tensor(np.array([1., .7, .3, 0.] if bad_sigmas is None else bad_sigmas))
                args.update(model=model, sigmas=sigmas, cfg=4)
                node = module.BerniniLongV2V13B()
            else:
                args.update(model_high='high', model_low='low', sigmas_high=np.array([1., .5]),
                            sigmas_low=np.array([.5, 0.]), cfg_high=1, cfg_low=1)
                node = module.BerniniLongV2V()
            if bad_config is not None or bad_sigmas is not None:
                with self.assertRaises(ValueError): node.run(**args)
                self.assertEqual(calls, [])
                self.assertEqual(sampler_calls, [])
                return
            output, report, frame_count, output_fps, comparison = node.run(**args)
        self.assertEqual(len(sampler_calls), expected_chunks * (1 if self.single_model else 2))
        for i, call in enumerate(sampler_calls):
            if self.single_model:
                self.assertIs(call[0], model)
                self.assertIs(call[7], sigmas)
                self.assertEqual(call[1:4], (True, 1, 4))
            else:
                self.assertEqual(call[:4], ('high' if i % 2 == 0 else 'low', i % 2 == 0, 1, 1))
        if expected_indices is None:
            expected_indices = np.minimum(np.arange(expected), available - 1)
        expected_source = source[expected_indices]
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
        self.assertEqual(report['sampling_mode'], 'single_model_1.3b' if self.single_model else 'high_low')
        self.assertEqual(report['frames'], frame_count)
        self.assertEqual(report['fps'], output_fps)
        self.assertEqual(report['input_frames'], available)
        self.assertEqual(report['max_seconds'], seconds)
        self.assertEqual(report['seconds'], expected / fps)
        self.assertEqual(report['frame_span_seconds'], (expected - 1) / fps)
        self.assertEqual(report['source_frames_used'], int(expected_indices[-1]) + 1)
        self.assertEqual(report['tail_padding_frames'], max(0, expected - available) if expected_tail is None else expected_tail)
        self.assertEqual(report['input_fps'], fps if expected_input_fps is None else expected_input_fps)
        self.assertEqual(report['timing_warning'] is None, video_info is not None or input_fps > 0)
        self.assertEqual(report['output_size'], [21, 13])
        self.assertEqual(report['model_canvas'], [32, 16])


class SingleModelNodeTests(ResolutionNodeTests):
    single_model = True

    def test_ten_seconds_can_use_one_sampling_window(self):
        self.check_node(161, 16, 10, 161, 0, 161, 1)

    def test_rejects_wrong_architecture_before_conditioning(self):
        for bad in ({'dim': 5120}, {'in_dim': 36}, {'model_type': 'vace'}, {'image_model': 'other'}):
            with self.subTest(config=bad):
                self.check_node(81, 16, 5, 81, 17, 81, 1, bad_config=bad)

    def test_rejects_incomplete_or_invalid_schedule_before_conditioning(self):
        for bad in ([], [0], [0, 0], [1, .5], [1, .3, .5, 0], [1, -1, 0], [float('nan'), 0], [float('inf'), 0], [[1, 0]]):
            with self.subTest(sigmas=bad):
                self.check_node(81, 16, 5, 81, 17, 81, 1, bad_sigmas=bad)


if __name__ == '__main__': unittest.main()
