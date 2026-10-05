import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server_paths as paths


class ExecutableDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.comfy = self.root / 'ComfyUI'
        self.package = self.comfy / 'custom_nodes' / 'ComfyUI-Bernini-LongVideo'
        self.package.mkdir(parents=True)
        self.sibling = self.package.parent / 'ComfyUI-MiniMaxH3-Prompter'
        self.sibling.mkdir()
        folder_paths = types.SimpleNamespace(base_path=str(self.comfy), get_folder_paths=lambda key: [str(self.package.parent)])
        for patcher in (patch.object(paths, 'PACKAGE_DIR', self.package),
                        patch.object(paths.shutil, 'which', return_value=None),
                        patch.object(paths.Path, 'home', return_value=self.root / 'home'),
                        patch.dict(sys.modules, {'folder_paths': folder_paths}),
                        patch.dict(os.environ, {'SystemDrive': str(self.root / 'drive')}) ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def binary(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / paths.EXE_NAME
        binary.touch()
        binary.chmod(0o755)
        return str(binary.resolve())

    def test_explicit_quoted_file_and_directory_override_path(self):
        binary = self.binary(self.root / 'manual')
        with patch.object(paths.shutil, 'which', return_value='other'):
            self.assertEqual(paths.find_llama_server({'llama_server_path': f'"{binary}"'}), binary)
            self.assertEqual(paths.find_llama_server({'llama_server_path': str(Path(binary).parent)}), binary)

    def test_path_lookup(self):
        with patch.object(paths.shutil, 'which', return_value='/existing/llama-server'):
            self.assertEqual(paths.find_llama_server({}), '/existing/llama-server')

    def test_minimax_config_reuses_existing_binary(self):
        binary = self.binary(self.root / 'shared' / 'bin')
        (self.sibling / 'config.json').write_text(json.dumps({'llama_server_path': binary}), encoding='utf-8')
        self.assertEqual(paths.find_llama_server({}), binary)

    def test_minimax_relative_config_is_relative_to_plugin(self):
        binary = self.binary(self.sibling / 'runtime' / 'bin')
        (self.sibling / 'config.json').write_text(json.dumps({'llama_server_path': 'runtime'}), encoding='utf-8')
        self.assertEqual(paths.find_llama_server({}), binary)

    def test_minimax_bundled_build_found_without_config(self):
        binary = self.binary(self.sibling / 'llama.cpp' / 'build' / 'bin' / 'Release')
        self.assertEqual(paths.find_llama_server({}), binary)

    def test_invalid_minimax_config_falls_back(self):
        (self.sibling / 'config.json').write_text('{broken', encoding='utf-8')
        binary = self.binary(self.package / 'llama.cpp' / 'bin')
        with self.assertLogs(paths.log, level='WARNING'):
            self.assertEqual(paths.find_llama_server({}), binary)

    def test_comfy_portable_and_home_layouts(self):
        for directory in (self.comfy / 'llama.cpp', self.root / 'llama.cpp' / 'build' / 'bin',
                          self.root / 'home' / 'llama.cpp'):
            binary = self.binary(directory)
            self.assertEqual(paths.find_llama_server({}), binary)
            Path(binary).unlink()

    def test_explicit_bad_path_does_not_silently_select_another_binary(self):
        self.binary(self.package / 'llama.cpp')
        with self.assertRaisesRegex(FileNotFoundError, 'kosongkan untuk auto-detect'):
            paths.find_llama_server({'llama_server_path': 'not-found'})

    def test_missing_executable_reports_search_scope(self):
        with patch.object(paths, '_locations', return_value=([], [self.root / 'missing'])):
            with self.assertRaisesRegex(FileNotFoundError, 'MiniMaxH3'):
                paths.find_llama_server({})


if __name__ == '__main__': unittest.main()
