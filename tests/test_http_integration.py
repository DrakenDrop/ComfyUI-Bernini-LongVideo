"""Exercise actual local HTTP transport without downloading an LLM."""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancer import enhance


class HTTPIntegrationTests(unittest.TestCase):
    def test_discovery_and_completion_over_http(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass

            def reply(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                requests.append((self.command, self.path, None))
                self.reply({'data': [{'id': 'local-fixture'}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.command, self.path, payload))
                self.reply({'choices': [{'message': {'content': 'Make the shirt blue.'}, 'finish_reason': 'stop'}]})

        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = enhance('Baju biru', '', f'http://127.0.0.1:{server.server_port}',
                                 'auto', '', 0.2, 512, 5)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertEqual(result, 'Make the shirt blue.')
        self.assertEqual([(r[0], r[1]) for r in requests], [('GET', '/v1/models'), ('POST', '/v1/chat/completions')])
        self.assertEqual(requests[1][2]['model'], 'local-fixture')


if __name__ == '__main__': unittest.main()
