"""Factory for embedder backbones — single entry point for all callers."""
from typing import Optional
from .base import BaseEmbedder

_SUPPORTED = {"clip", "openclip_b32", "openclip_l14", "blip"}
_instances: dict = {}


def get_embedder(backbone: str = "clip") -> BaseEmbedder:
    """
    Return a lazy-loaded singleton embedder for the requested backbone.

    Parameters
    ----------
    backbone : str
        One of: 'clip' (HF openai/clip-vit-base-patch32),
                'openclip_b32' (OpenCLIP ViT-B/32, LAION-2B),
                'openclip_l14' (OpenCLIP ViT-L/14, LAION-2B),
                'blip' (Salesforce/blip-itm-base — added later).
    """
    if backbone not in _SUPPORTED:
        raise ValueError(f"Unknown backbone {backbone!r}. Supported: {_SUPPORTED}")

    if backbone in _instances:
        return _instances[backbone]

    if backbone == "clip":
        from .clip_hf import CLIPEmbedder
        _instances[backbone] = CLIPEmbedder()
    elif backbone == "openclip_b32":
        from .open_clip import OpenCLIPEmbedder
        _instances[backbone] = OpenCLIPEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    elif backbone == "openclip_l14":
        from .open_clip import OpenCLIPEmbedder
        _instances[backbone] = OpenCLIPEmbedder(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k")
    elif backbone == "blip":
        from .blip import BLIPEmbedder
        _instances[backbone] = BLIPEmbedder()    

    return _instances[backbone]