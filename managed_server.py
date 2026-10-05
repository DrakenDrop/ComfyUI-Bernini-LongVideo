"""Own one llama-server child process; never stop an external server."""
import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from urllib.request import build_opener, ProxyHandler, HTTPRedirectHandler

_lock = threading.RLock()
_process = None
_signature = None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def stop_owned():
    global _process, _signature
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=10)
    _process = None
    _signature = None


def executable(config):
    if __package__:
        from .server_paths import find_llama_server
    else:  # Standalone CPU tests.
        from server_paths import find_llama_server
    return find_llama_server(config)


def launch(model, projector, context_size, config, check_cancel):
    global _process, _signature
    port = int(config.get("managed_port", 8091))
    wait = float(config.get("startup_wait_seconds", 900))
    if not 1 <= port <= 65535 or not 1 <= wait <= 3600:
        raise ValueError("managed_port harus 1..65535 dan startup_wait_seconds 1..3600.")
    args = [executable(config), "-m", model, "--host", "127.0.0.1", "--port", str(port),
            "--alias", "bernini-local", "-c", str(context_size), "-np", "1", "--jinja",
            "-ngl", str(int(config.get("gpu_layers", 999)))]
    if projector:
        args.extend(["--mmproj", projector])
    signature = tuple(args)
    if _signature == signature and _process is not None and _process.poll() is None:
        return f"http://127.0.0.1:{port}/v1"
    stop_owned()
    # Refuse occupied ports, including other nodes' managed servers.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} sudah dipakai. Ganti managed_port; server lain tidak dihentikan.") from None
    log_path = Path(tempfile.gettempdir()) / f"bernini-llama-{os.getpid()}.log"
    with log_path.open("wb") as log:
        _process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    client = build_opener(ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        check_cancel()
        if _process.poll() is not None:
            raise RuntimeError(f"llama-server berhenti saat loading. Periksa {log_path}")
        try:
            with client.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    _signature = signature
                    return f"http://127.0.0.1:{port}/v1"
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Loading llama-server timeout. Periksa {log_path}")


@contextmanager
def local_server(model, projector, context_size, config, unload=True, check_cancel=lambda: None):
    with _lock:
        try:
            yield launch(model, projector, context_size, config, check_cancel)
        except BaseException:
            stop_owned()
            raise
        finally:
            if unload:
                stop_owned()


atexit.register(stop_owned)
