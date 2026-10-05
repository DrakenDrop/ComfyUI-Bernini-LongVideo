"""Exercise vision input selection and local projector validation without GPU."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, array): self.array = array
    def __len__(self): return len(self.array)
    def __getitem__(self, index): return Tensor(self.array[index])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class EnhancerNodeTests(unittest.TestCase):
    def setUp(self):
        package = types.ModuleType('bernini_enhancer_test')
        package.__path__ = [str(ROOT)]
        modules = {'bernini_enhancer_test': package, 'torch': types.ModuleType('torch')}
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        spec = importlib.util.spec_from_file_location('bernini_enhancer_test.nodes', ROOT / 'nodes.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.config = patch.object(self.module, 'load_config', return_value={}).start()
        self.client = patch.object(self.module, 'enhance', return_value='Edited prompt').start()
        self.addCleanup(patch.stopall)
        self.args = dict(instruction='Match reference clothes', enabled=True, scene_description='',
                         base_url='http://localhost:8000', model='auto', api_key_env='',
                         sample_frames=1, temperature=0.2, max_tokens=512, timeout=10)
        self.source = np.arange(5*2*3*3,dtype=np.float32).reshape(5,2,3,3)
        self.reference = np.ones((2,4,5,3),dtype=np.float32)

    def call(self, **changes):
        return self.module.BerniniPromptEnhancerVLLM().run(**(self.args | changes))

    def test_one_source_and_first_reference_are_passed_separately(self):
        self.call(source_frames=Tensor(self.source), reference_images=Tensor(self.reference))
        np.testing.assert_array_equal(self.client.call_args.args[-2], self.source[:1])
        np.testing.assert_array_equal(self.client.call_args.kwargs['reference_images'], self.reference[:1])

    def test_source_sampling_stays_uniform_when_reference_is_added(self):
        self.call(source_frames=Tensor(self.source), sample_frames=3, reference_images=Tensor(self.reference))
        np.testing.assert_array_equal(self.client.call_args.args[-2], self.source[[0,2,4]])

    def test_reference_only_requires_local_projector(self):
        with patch.object(self.module, 'resolve', return_value=('model.gguf', None)):
            with self.assertRaisesRegex(ValueError, 'mmproj'):
                self.call(llm_model='local.gguf', sample_frames=0, reference_images=Tensor(self.reference))
        self.client.assert_not_called()

    def test_explicit_text_only_mode_omits_both_image_inputs(self):
        mm = types.ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = MagicMock()
        mm.unload_all_models = MagicMock()
        mm.soft_empty_cache = MagicMock()
        comfy = types.ModuleType('comfy')
        comfy.model_management = mm
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': mm}), \
             patch.object(self.module, 'resolve', return_value=('model.gguf', None)), \
             patch.object(self.module, 'local_server') as server:
            server.return_value.__enter__.return_value = 'http://localhost:8091/v1'
            self.call(llm_model='local.gguf', mmproj=self.module.MMPROJ_NONE,
                      source_frames=Tensor(self.source), reference_images=Tensor(self.reference))
        self.assertIsNone(self.client.call_args.args[-2])
        self.assertIsNone(self.client.call_args.kwargs['reference_images'])

    def test_disabled_node_does_not_inspect_reference_or_call_server(self):
        self.assertEqual(self.call(enabled=False, reference_images=object()), ('Match reference clothes',))
        self.config.assert_not_called()
        self.client.assert_not_called()

    def test_empty_reference_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, 'referensi'):
            self.call(reference_images=Tensor(self.reference[:0]))
        self.client.assert_not_called()

    def test_existing_text_only_call_still_works(self):
        self.assertEqual(self.call(sample_frames=0), ('Edited prompt',))
        self.assertIsNone(self.client.call_args.kwargs['reference_images'])


if __name__ == '__main__': unittest.main()
