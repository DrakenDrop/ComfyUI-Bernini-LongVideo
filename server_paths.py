"""Locate an existing llama.cpp installation without importing other custom nodes."""
import json
import logging
import os
from pathlib import Path
import shutil

log = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent
EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
LAYOUTS = ("", "bin", "build/bin", "build/bin/Release", "build/bin/Debug")


def _path(value, base):
    path = Path(os.path.expandvars(value.strip().strip('"'))).expanduser()
    return path if path.is_absolute() else base / path


def _candidate(value, base):
    if not isinstance(value, str) or not value.strip():
        return None
    path = _path(value, base)
    candidates = [path / layout / EXE_NAME for layout in LAYOUTS] if path.is_dir() else [path]
    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate.resolve())
    return None


def _locations():
    comfy_roots, custom_roots = [], [PACKAGE_DIR.parent]
    if PACKAGE_DIR.parent.name.lower() == "custom_nodes":
        comfy_roots.append(PACKAGE_DIR.parent.parent)
    try:
        import folder_paths
        if getattr(folder_paths, "base_path", None):
            comfy_roots.append(Path(folder_paths.base_path))
        try:
            custom_roots.extend(Path(p) for p in folder_paths.get_folder_paths("custom_nodes"))
        except KeyError:
            pass
    except ImportError:
        pass
    siblings = []
    for parent in dict.fromkeys(custom_roots):
        try:
            siblings.extend(p for p in parent.iterdir() if p.is_dir()
                            and p.name.casefold() == "comfyui-minimaxh3-prompter")
        except OSError:
            continue
    roots = [PACKAGE_DIR / "llama.cpp"]
    roots.extend(p / "llama.cpp" for p in siblings)
    for root in dict.fromkeys(comfy_roots):
        roots.extend((root / "llama.cpp", root.parent / "llama.cpp"))
    roots.append(Path.home() / "llama.cpp")
    if os.name == "nt":
        roots.append(Path(os.environ.get("SystemDrive", "C:" ) + "/") / "llama.cpp")
        roots.append(Path("C:/llama.cpp"))
    return list(dict.fromkeys(siblings)), list(dict.fromkeys(roots))


def find_llama_server(config):
    configured = config.get("llama_server_path")
    if configured and not isinstance(configured, str):
        raise ValueError("llama_server_path harus berupa string path file atau folder.")
    if configured and configured.strip():
        found = _candidate(configured, PACKAGE_DIR) or shutil.which(configured.strip().strip('"'))
        if found:
            return found
        raise FileNotFoundError("llama_server_path di bernini_config.json tidak valid. Perbaiki path atau kosongkan untuk auto-detect.")
    found = shutil.which(EXE_NAME)
    if found:
        return found
    siblings, roots = _locations()
    # Reuse only the executable setting; never import another node or its API keys.
    for sibling in siblings:
        try:
            config_path = sibling / "config.json"
            if not config_path.is_file():
                continue
            settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if isinstance(settings, dict):
                found = _candidate(settings.get("llama_server_path"), sibling)
                if found:
                    log.info("Bernini: llama-server detected from MiniMaxH3 configuration: %s", found)
                    return found
        except (OSError, ValueError):
            log.warning("Bernini: unable to read MiniMaxH3 config; continuing llama-server discovery.")
    for root in roots:
        found = _candidate(str(root), PACKAGE_DIR)
        if found:
            log.info("Bernini: llama-server detected: %s", found)
            return found
    raise FileNotFoundError(
        "llama-server belum ditemukan setelah memeriksa PATH, konfigurasi/folder MiniMaxH3, "
        "serta folder llama.cpp di node, ComfyUI, portable root, dan home. "
        "Jika MiniMaxH3 bekerja di instalasi ComfyUI yang sama, pastikan folder plugin itu masih tersedia. "
        "Alternatif: isi llama_server_path di bernini_config.json dengan path executable atau folder llama.cpp.")
