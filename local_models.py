"""Discover local prompt models; never download weights automatically."""
import json
import os
import re
from pathlib import Path

SERVER_DEFAULT = "server (vLLM / OpenAI-compatible)"
MMPROJ_AUTO = "auto"
MMPROJ_NONE = "none (text only)"
CONFIG = Path(__file__).with_name("bernini_config.json")
SHARD = re.compile(r"-([0-9]{5})-of-([0-9]{5})$", re.I)


def load_config():
    if not CONFIG.exists():
        return {}
    value = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("bernini_config.json harus berupa object JSON.")
    return value


def model_roots(config):
    candidates = []
    try:
        import folder_paths
        candidates.append(Path(folder_paths.models_dir) / "LLM")
        for name in ("LLM", "llm"):
            try:
                candidates.extend(folder_paths.get_folder_paths(name))
            except KeyError:
                pass
    except ImportError:
        pass
    extra = config.get("extra_model_dirs", [])
    if not isinstance(extra, list):
        raise ValueError("extra_model_dirs harus berupa array path.")
    candidates.extend(extra)
    roots = []
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def scan(config):
    models, projectors, seen = {}, {}, set()
    for root in model_roots(config):
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix.lower() != ".gguf" or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                shard = SHARD.search(path.stem)
                if shard and shard[1] != "00001":
                    continue
                target = projectors if "mmproj" in name.lower() else models
                label = path.relative_to(root).as_posix()
                if label in target:
                    label += f" [{root.as_posix()}]"
                target[label] = str(path)
    return dict(sorted(models.items())), dict(sorted(projectors.items()))


def family(path):
    stem = SHARD.sub("", Path(path).stem.lower()).replace("mmproj", "")
    stem = re.sub(r"[-_.](?:[iu]?q\d.*|bf16|f16|f32)$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem)


def auto_projector(model, models, projectors):
    # Exact normalized family names only; similar prefixes can be different sizes.
    matching = [p for p in projectors.values() if family(p) == family(model)]
    local = [p for p in matching if Path(p).parent == Path(model).parent]
    candidates = local or matching
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return None  # Explicit choice required when multiple precisions are present.
    local_models = {family(p) for p in models.values() if Path(p).parent == Path(model).parent}
    generic = [p for p in projectors.values()
               if Path(p).parent == Path(model).parent and not family(p)]
    return generic[0] if len(local_models) == 1 and len(generic) == 1 else None


def choices(config):
    models, projectors = scan(config)
    return [SERVER_DEFAULT, *models], [MMPROJ_AUTO, MMPROJ_NONE, *projectors]


def resolve(model_label, projector_label, config):
    models, projectors = scan(config)
    if model_label not in models:
        raise FileNotFoundError("Model GGUF tidak ditemukan. Refresh daftar model dan pilih ulang.")
    model = models[model_label]
    if projector_label == MMPROJ_NONE:
        return model, None
    if projector_label == MMPROJ_AUTO:
        return model, auto_projector(model, models, projectors)
    if projector_label not in projectors:
        raise FileNotFoundError("mmproj tidak ditemukan. Refresh daftar model dan pilih ulang.")
    return model, projectors[projector_label]
