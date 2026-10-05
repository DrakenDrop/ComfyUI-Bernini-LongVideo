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
    """Include the endpoint and round up to Wan's 4n+1 frame grid.

    Short sources need at most three repeated tail frames, not an extension to
    the requested duration. The caller must retain this tail in the output.
    """
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("fps harus positif; seconds harus >= 0 (0 = seluruh input).")
    if available < 1:
        raise ValueError("Video input kosong.")
    count = available
    if seconds > 0:
        intervals = fps * seconds
        if not math.isfinite(intervals):
            raise ValueError("fps * seconds harus finite.")
        count = min(count, 1 + 4 * math.ceil(intervals / 4))
    return 1 + 4 * ((count - 1 + 3) // 4)


def input_frame_rate(video_info, input_fps, output_fps):
    """VHS loaded_fps describes the actual IMAGE batch, unlike source_fps."""
    if video_info is not None:
        if not isinstance(video_info, dict) or 'loaded_fps' not in video_info:
            raise ValueError("video_info harus berasal dari VHS Load Video dan memiliki loaded_fps.")
        value, origin = video_info['loaded_fps'], 'video_info.loaded_fps'
    else:
        try:
            manual = float(input_fps)
        except (TypeError, ValueError):
            raise ValueError("input_fps harus angka >= 0.") from None
        if not math.isfinite(manual) or manual < 0:
            raise ValueError("input_fps harus angka finite >= 0.")
        value, origin = (manual, 'input_fps') if manual else (output_fps, 'assumed_from_output_fps')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("FPS input tidak valid.") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("FPS input harus positif dan finite.")
    return value, origin


def source_frame_indices(available, input_fps, output_fps, seconds):
    """Sample source timestamps at output_fps, then retain 4n+1 tail alignment.

    Nearest-frame selection changes sampling density, not motion speed. There is
    no interpolation or optical flow. The final 0-3 alignment frames repeat the
    last selected source frame, never rescale the whole clip's time axis.
    """
    for rate in (input_fps, output_fps):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("FPS harus positif dan finite.")
    if available < 1:
        raise ValueError("Video input kosong.")
    count_float = available * output_fps / input_fps
    if not math.isfinite(count_float):
        raise ValueError("Jumlah frame hasil konversi FPS terlalu besar.")
    # Avoid one spurious sample from floating-point noise at an integer boundary.
    nearest = round(count_float)
    if abs(count_float - nearest) < 1e-9:
        count_float = nearest
    resampled_available = max(1, math.ceil(count_float))
    total = target_frames(resampled_available, output_fps, seconds)
    valid = min(total, resampled_available)
    times = np.minimum(np.arange(total, dtype=np.float64), valid - 1) / output_fps
    indices = np.minimum(np.floor(times * input_fps + 0.5), available - 1).astype(np.int64)
    return indices, valid, total - valid


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
