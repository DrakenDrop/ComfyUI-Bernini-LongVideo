"""CPU timeline assembly; independent of ComfyUI and model weights."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    padded: int


def plan_windows(total, chunk_frames=81, overlap=17):
    if total < 1:
        raise ValueError("Video input kosong.")
    if chunk_frames < 5 or (chunk_frames - 1) % 4:
        raise ValueError("chunk_frames harus 4n+1, misalnya 49, 65, 81, 121.")
    if not 0 <= overlap < chunk_frames:
        raise ValueError("overlap harus >= 0 dan lebih kecil dari chunk_frames.")
    result = []
    start = 0
    while True:
        end = min(start + chunk_frames, total)
        valid = end - start
        result.append(Window(start, end, 1 + 4 * math.ceil((valid - 1) / 4)))
        if end == total:
            return result
        start += chunk_frames - overlap


def target_frames(available, fps, seconds):
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("fps harus positif; seconds harus >= 0 (0 = seluruh input).")
    if available < 1:
        raise ValueError("Video input kosong.")
    if seconds == 0:
        return available
    return min(available, max(1, math.floor(fps * seconds + 0.5)))


def render_timeline(source, total, chunk_frames, overlap, blend_mode, render, progress=None):
    """render receives padded source frames and the window index, returns NHWC.

    Output is preallocated once in CPU RAM. The renderer owns GPU work for only
    one window at a time. Blending overlaps never inserts or removes timestamps.
    """
    if total > len(source):
        raise ValueError("Jumlah target melebihi frame input.")
    if blend_mode not in ("crossfade", "cut"):
        raise ValueError("blend_mode harus crossfade atau cut.")
    windows = plan_windows(total, chunk_frames, overlap)
    output = None
    previous_end = 0
    for index, window in enumerate(windows):
        valid = window.end - window.start
        clip = source[window.start:window.end]
        if window.padded > valid:
            clip = np.concatenate((clip, np.repeat(clip[-1:], window.padded - valid, axis=0)))
        decoded = np.asarray(render(clip, index), dtype=np.float32)
        if decoded.ndim != 4 or len(decoded) < valid or decoded.shape[-1] != 3:
            raise ValueError("Renderer harus mengembalikan frame RGB NHWC yang cukup.")
        decoded = decoded[:valid]
        if output is None:
            output = np.empty((total, *decoded.shape[1:]), dtype=np.float32)
        elif decoded.shape[1:] != output.shape[1:]:
            raise ValueError("Resolusi renderer berubah antar chunk.")
        shared = max(0, previous_end - window.start)
        if shared:
            if blend_mode == "crossfade":
                weight = np.linspace(0, 1, shared + 2, dtype=np.float32)[1:-1, None, None, None]
                old = output[window.start:window.start + shared]
                output[window.start:window.start + shared] = old * (1 - weight) + decoded[:shared] * weight
            else:
                middle = shared // 2
                output[window.start + middle:window.start + shared] = decoded[middle:shared]
        output[window.start + shared:window.end] = decoded[shared:]
        previous_end = window.end
        del decoded, clip
        if progress:
            progress(index + 1, len(windows))
    return output, windows
