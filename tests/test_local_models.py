import sys
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_models import scan, choices, resolve, SERVER_DEFAULT, MMPROJ_NONE
import managed_server as server


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'extra_model_dirs': [str(self.root)]}

    def add(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_nested_shards_and_projectors(self):
        self.add('Qwen/model-Q4-00001-of-00002.gguf')
        self.add('Qwen/model-Q4-00002-of-00002.gguf')
        self.add('Qwen/mmproj-F16.GGUF')
        self.add('.hidden/private.gguf')
        models, projectors = scan(self.config)
        self.assertEqual(list(models), ['Qwen/model-Q4-00001-of-00002.gguf'])
        self.assertEqual(list(projectors), ['Qwen/mmproj-F16.GGUF'])
        self.assertEqual(choices(self.config)[0][0], SERVER_DEFAULT)

    def test_matching_family_and_text_only(self):
        model = self.add('Qwen2.5-VL-7B-Q4_K_M.gguf')
        projector = self.add('mmproj-Qwen2.5-VL-7B-F16.gguf')
        self.add('mmproj-Qwen2.5-VL-3B-F16.gguf')
        self.assertEqual(resolve(Path(model).name, 'auto', self.config), (model, projector))
        self.assertEqual(resolve(Path(model).name, MMPROJ_NONE, self.config), (model, None))

    def test_generic_projector_only_for_single_local_family(self):
        self.add('model-Q4.gguf')
        projector = self.add('mmproj-F16.gguf')
        self.assertEqual(resolve('model-Q4.gguf', 'auto', self.config)[1], projector)
        self.add('other-Q4.gguf')
        self.assertIsNone(resolve('model-Q4.gguf', 'auto', self.config)[1])

    def test_unrelated_or_ambiguous_projector_not_guessed(self):
        self.add('one/model-Q4.gguf')
        self.add('two/mmproj-other-F16.gguf')
        self.assertIsNone(resolve('one/model-Q4.gguf', 'auto', self.config)[1])
        self.add('one/mmproj-model-F16.gguf')
        self.add('one/mmproj-model-BF16.gguf')
        self.assertIsNone(resolve('one/model-Q4.gguf', 'auto', self.config)[1])

    def test_duplicate_roots_and_labels(self):
        roots = [self.root / str(i) / 'LLM' for i in range(3)]
        for root in roots:
            root.mkdir(parents=True)
            (root / 'same.gguf').touch()
        models, _ = scan({'extra_model_dirs': [str(r) for r in [*roots, roots[0]]]})
        self.assertEqual(len(models), 3)
        self.assertEqual(len(set(models.values())), 3)

    def test_stale_selection_rejected(self):
        with self.assertRaises(FileNotFoundError): resolve('missing.gguf', 'auto', self.config)


class ManagedServerTests(unittest.TestCase):
    def setUp(self):
        server._process = None
        server._signature = None
        server._port = None
        self.addCleanup(server.stop_owned)

    @patch('managed_server.executable', return_value='llama-server')
    @patch('managed_server.choose_port', return_value=8091)
    @patch('managed_server.subprocess.Popen')
    @patch('managed_server.build_opener')
    def test_owned_launch_arguments_reuse_and_unload(self, opener, popen, sock, exe):
        process = popen.return_value
        process.poll.return_value = None
        opener.return_value.open.return_value.__enter__.return_value.status = 200
        with server.local_server('a.gguf', 'mm.gguf', 8192, {}, unload=False) as url:
            self.assertEqual(url, 'http://127.0.0.1:8091/v1')
        process.terminate.assert_not_called()
        with server.local_server('a.gguf', 'mm.gguf', 8192, {}, unload=True):
            pass
        self.assertEqual(popen.call_count, 1)
        args = popen.call_args.args[0]
        self.assertIn('--mmproj', args)
        self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')
        self.assertFalse(popen.call_args.kwargs.get('shell', False))
        process.terminate.assert_called_once()
        self.assertIsNone(server._process)

    @patch('managed_server.executable', return_value='llama-server')
    @patch('managed_server.choose_port', return_value=53211)
    @patch('managed_server.subprocess.Popen')
    @patch('managed_server.build_opener')
    def test_fallback_port_is_used_for_launch_health_and_reuse(self, opener, popen, choose, exe):
        popen.return_value.poll.return_value = None
        opener.return_value.open.return_value.__enter__.return_value.status = 200
        with server.local_server('a.gguf', None, 8192, {}, unload=False) as url:
            self.assertEqual(url, 'http://127.0.0.1:53211/v1')
        with server.local_server('a.gguf', None, 8192, {}, unload=False) as reused:
            self.assertEqual(reused, url)
        choose.assert_called_once_with(8091)
        popen.assert_called_once()
        args = popen.call_args.args[0]
        self.assertEqual(args[args.index('--port') + 1], '53211')
        self.assertEqual(opener.return_value.open.call_args.args[0], 'http://127.0.0.1:53211/health')
        popen.return_value.terminate.assert_not_called()

    def test_real_occupied_port_selects_another_without_touching_listener(self):
        with socket.socket() as external:
            external.bind(('127.0.0.1', 0))
            external.listen(1)
            occupied = external.getsockname()[1]
            selected = server.choose_port(occupied)
            self.assertNotEqual(selected, occupied)
            self.assertGreater(selected, 0)
            self.assertEqual(external.getsockname()[1], occupied)

    def test_zero_selects_an_available_port(self):
        selected = server.choose_port(0)
        self.assertTrue(1 <= selected <= 65535)

    @patch('managed_server.socket.socket')
    def test_failure_to_allocate_port_has_actionable_error(self, sock):
        sock.return_value.__enter__.return_value.bind.side_effect = OSError('denied')
        with self.assertRaisesRegex(RuntimeError, 'port localhost'):
            server.choose_port(8091)

    @patch('managed_server.launch')
    def test_cancellation_and_request_failure_clean_up_even_when_keep_loaded(self, launch):
        for failure in (RuntimeError('failed'), KeyboardInterrupt()):
            process = MagicMock()
            process.poll.return_value = None
            server._process = process
            with self.assertRaises(type(failure)):
                with server.local_server('a', None, 8192, {}, unload=False):
                    raise failure
            process.terminate.assert_called_once()


if __name__ == '__main__': unittest.main()
