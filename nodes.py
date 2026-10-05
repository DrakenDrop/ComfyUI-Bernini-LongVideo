import json
import logging

import numpy as np
import torch

from .core import target_frames, render_timeline, plan_windows
from .enhancer import enhance

log = logging.getLogger(__name__)


class BerniniPromptEnhancerVLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "instruction": ("STRING", {"multiline": True, "default": ""}),
            "enabled": ("BOOLEAN", {"default": False}),
            "scene_description": ("STRING", {"multiline": True, "default": ""}),
            "base_url": ("STRING", {"default": "http://127.0.0.1:8000/v1"}),
            "model": ("STRING", {"default": ""}),
            "api_key_env": ("STRING", {"default": "VLLM_API_KEY"}),
            "sample_frames": ("INT", {"default": 0, "min": 0, "max": 8}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0, "max": 2}),
            "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
            "timeout": ("INT", {"default": 90, "min": 5, "max": 600}),
        }, "optional": {"source_frames": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "run"
    CATEGORY = "Bernini/Long Video"
    DESCRIPTION = "Optional text/VLM prompt rewriting through your vLLM server. sample_frames > 0 sends sampled images to that server."

    def run(self, instruction, enabled, scene_description, base_url, model, api_key_env,
            sample_frames, temperature, max_tokens, timeout, source_frames=None):
        if not enabled:
            return (instruction,)
        frames = None
        if sample_frames and source_frames is not None:
            if len(source_frames) == 0:
                raise ValueError("Frame untuk enhancer kosong.")
            # Copy only requested frames from GPU, not the entire video.
            indices = np.linspace(0, len(source_frames) - 1, min(sample_frames, len(source_frames)), dtype=int).tolist()
            frames = source_frames[indices].detach().cpu().numpy()
        return (enhance(instruction, scene_description, base_url, model, api_key_env,
                        temperature, max_tokens, timeout, frames, sample_frames),)


class TiledEncodeVAE:
    """Only the encode interface used by native BerniniConditioning."""
    def __init__(self, vae, tile_size):
        self.vae = vae
        self.tile_size = tile_size

    def encode(self, pixels):
        return self.vae.encode_tiled(pixels, tile_x=self.tile_size, tile_y=self.tile_size,
                                     overlap=min(64, self.tile_size // 4), tile_t=64, overlap_t=8)


class BerniniLongV2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_high": ("MODEL",), "model_low": ("MODEL",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "vae": ("VAE",), "source_video": ("IMAGE",),
            "sampler": ("SAMPLER",), "sigmas_high": ("SIGMAS",), "sigmas_low": ("SIGMAS",),
            "width": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
            "fps": ("FLOAT", {"default": 16, "min": 1, "max": 120}),
            "max_seconds": ("FLOAT", {"default": 30, "min": 0, "max": 3600}),
            "chunk_frames": ("INT", {"default": 81, "min": 5, "max": 513, "step": 4}),
            "overlap": ("INT", {"default": 17, "min": 0, "max": 512}),
            "blend_mode": (["crossfade", "cut"],),
            "seed": ("INT", {"default": 413532359003415, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
            "seed_mode": (["fixed", "increment"],),
            "cfg_high": ("FLOAT", {"default": 1, "min": 0, "max": 30}),
            "cfg_low": ("FLOAT", {"default": 1, "min": 0, "max": 30}),
            "tiled_encode": ("BOOLEAN", {"default": True}),
            "tiled_decode": ("BOOLEAN", {"default": True}),
            "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
            "ref_max_size": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
        }, "optional": {"reference_images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "Bernini/Long Video"
    DESCRIPTION = "Sequential native Bernini V2V windows. Set the video loader and saver to the same fps. Overlap blending is not a temporal consistency guarantee."

    def run(self, model_high, model_low, positive, negative, vae, source_video, sampler,
            sigmas_high, sigmas_low, width, height, fps, max_seconds, chunk_frames,
            overlap, blend_mode, seed, seed_mode, cfg_high, cfg_low, tiled_encode,
            tiled_decode, tile_size, ref_max_size, reference_images=None):
        # Imports here keep the standalone prompt enhancer usable without Bernini core support.
        try:
            from comfy_extras.nodes_bernini import BerniniConditioning
        except ImportError:
            raise RuntimeError("Update ComfyUI ke versi yang memiliki comfy_extras/nodes_bernini.py.") from None
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        from nodes import VAEDecode, VAEDecodeTiled
        import comfy.model_management as mm
        import comfy.utils

        if width % 16 or height % 16:
            raise ValueError("width dan height harus kelipatan 16.")
        total = target_frames(len(source_video), fps, max_seconds)
        windows = plan_windows(total, chunk_frames, overlap)
        if len(sigmas_high) < 2 or len(sigmas_low) < 2:
            raise ValueError("Masing-masing tahap high/low harus memiliki minimal satu langkah sampling.")
        if not torch.isclose(sigmas_high[-1], sigmas_low[0]).item() or sigmas_low[-1].item() != 0:
            raise ValueError("Sigma high/low harus bersambung dan tahap low harus berakhir di 0. Gunakan SplitSigmas.")
        source = source_video[:total, :, :, :3].detach().cpu().numpy()
        refs = {"reference_image_0": reference_images} if reference_images is not None else None
        encoder = TiledEncodeVAE(vae, tile_size) if tiled_encode else vae
        progress = comfy.utils.ProgressBar(len(windows))

        def render(clip, index):
            mm.throw_exception_if_processing_interrupted()
            chunk_seed = (seed + (index if seed_mode == "increment" else 0)) % (1 << 64)
            log.info("Bernini Long V2V: chunk %d/%d, %d frames", index + 1, len(windows), len(clip))
            conditioned = BerniniConditioning.execute(
                positive, negative, encoder, width, height, len(clip), 1,
                source_video=torch.from_numpy(clip), reference_images=refs, ref_max_size=ref_max_size)
            pos, neg, latent = conditioned[0], conditioned[1], conditioned[2]
            high_result = SamplerCustom().sample(model_high, True, chunk_seed, cfg_high,
                                                 pos, neg, sampler, sigmas_high, latent)
            high = high_result[0]
            del high_result, latent
            low_result = SamplerCustom().sample(model_low, False, chunk_seed, cfg_low,
                                                pos, neg, sampler, sigmas_low, high)
            low = low_result[0]
            del low_result, high, pos, neg, conditioned
            mm.throw_exception_if_processing_interrupted()
            if tiled_decode:
                images = VAEDecodeTiled().decode(vae, low, tile_size, 64, 64, 8)[0]
            else:
                images = VAEDecode().decode(vae, low)[0]
            return images.detach().cpu().numpy()

        output, windows = render_timeline(source, total, chunk_frames, overlap, blend_mode, render,
                                         lambda done, count: progress.update_absolute(done, count))
        report = {"frames": total, "fps": fps, "seconds": total / fps, "chunks": len(windows),
                  "windows": [{"start": w.start, "end_exclusive": w.end, "sampled_frames": w.padded} for w in windows],
                  "blend": blend_mode, "output_ram_gib": output.nbytes / 1024 ** 3,
                  "note": "Independent sampling windows + pixel overlap. No latent continuity lock; inspect seams and identity drift."}
        return (torch.from_numpy(output), json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {"BerniniLongV2V": BerniniLongV2V,
                       "BerniniPromptEnhancerVLLM": BerniniPromptEnhancerVLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"BerniniLongV2V": "Bernini · Long V2V",
                              "BerniniPromptEnhancerVLLM": "Bernini · Prompt Enhancer (vLLM)"}
