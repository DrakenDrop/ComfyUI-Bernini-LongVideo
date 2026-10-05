"""Exact-ratio output geometry with temporary model-alignment padding."""
from dataclasses import dataclass
from math import gcd

import numpy as np
from PIL import Image

RESOLUTIONS = ["480p", "720p", "1080p", "360p", "source", "custom"]


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    model_width: int
    model_height: int
    left: int
    top: int


def output_geometry(input_width, input_height, resolution, width=480, height=832):
    if input_width < 1 or input_height < 1:
        raise ValueError("Dimensi video input harus positif.")
    if resolution not in RESOLUTIONS:
        raise ValueError("Preset resolution tidak dikenal.")
    if resolution == "custom":
        if width < 16 or height < 16 or width % 16 or height % 16:
            raise ValueError("Pada mode custom, width dan height harus kelipatan 16.")
    elif resolution == "source":
        width, height = input_width, input_height
    else:
        divisor = gcd(input_width, input_height)
        ratio_w, ratio_h = input_width // divisor, input_height // divisor
        # A reduced ratio has at least one odd component. An even multiplier
        # therefore yields even dimensions for ordinary yuv420p video encoders.
        unit = 2 * min(ratio_w, ratio_h)
        target = int(resolution[:-1])
        multiplier = 2 * max(1, (target + unit // 2) // unit)
        width, height = ratio_w * multiplier, ratio_h * multiplier
    model_width = ((width + 15) // 16) * 16
    model_height = ((height + 15) // 16) * 16
    if model_width > 8192 or model_height > 8192:
        raise ValueError("Ukuran dengan rasio persis melebihi 8192 px. Pilih preset lebih kecil bila memungkinkan; rasio input tidak dibulatkan diam-diam.")
    return Geometry(width, height, model_width, model_height,
                    (model_width - width) // 2, (model_height - height) // 2)


def prepare_frames(frames, geometry, check_cancel=lambda: None):
    """Resize RGB float frames without quantization, pad edges, one frame at a time."""
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("Video harus berupa batch gambar NHWC RGB.")
    g = geometry
    output = np.empty((len(frames), g.model_height, g.model_width, 3), dtype=np.float32)
    bottom, right = g.top + g.height, g.left + g.width
    for i, frame in enumerate(frames):
        check_cancel()
        if frame.shape[:2] == (g.height, g.width):
            output[i, g.top:bottom, g.left:right] = frame[:, :, :3]
        else:
            for channel in range(3):
                plane = Image.fromarray(np.asarray(frame[:, :, channel], dtype=np.float32))
                resized = np.asarray(plane.resize((g.width, g.height), Image.Resampling.LANCZOS))
                output[i, g.top:bottom, g.left:right, channel] = np.clip(resized, 0, 1)
        # Replicate border pixels; remove these pixels again after VAE decoding.
        output[i, :g.top, g.left:right] = output[i, g.top:g.top + 1, g.left:right]
        output[i, bottom:, g.left:right] = output[i, bottom - 1:bottom, g.left:right]
        output[i, :, :g.left] = output[i, :, g.left:g.left + 1]
        output[i, :, right:] = output[i, :, right - 1:right]
    return output


def remove_padding(frames, geometry):
    g = geometry
    if frames.ndim != 4 or frames.shape[1:3] != (g.model_height, g.model_width):
        raise ValueError("Ukuran hasil VAE tidak sesuai canvas Bernini.")
    return frames[:, g.top:g.top + g.height, g.left:g.left + g.width, :3]
