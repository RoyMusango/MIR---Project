"""
backbone_manager.py
─────────────────────────────────────────────────────────────────────
Group 07 — I-ILIA-014 MIR Project
Phase 3: Centralised, multi-backbone resource manager.

Replaces the global FAISS / embedder singletons in ``app.py`` with a
thread-safe cache that:

    • lazy-loads embedders and FAISS indices on first use,
    • caches them as long as RAM allows,
    • evicts least-recently-used entries when the cache exceeds
      ``MIR_MAX_BACKBONES`` (env var, default 1).

This makes the live backbone selector in the UI cheap to swap on a
laptop (one backbone resident at a time) while still letting HF Spaces
CPU-Basic (16 GB RAM) keep all four resident for instant switching.

Public API:

    from backbone_manager import manager

    embedder    = manager.get_embedder("clip")
    image_idx   = manager.get_image_index("clip")        # FAISS
    caption_idx = manager.get_caption_index("clip")      # FAISS
    img_files   = manager.get_image_files("clip")        # list[str]
    captions    = manager.get_captions("clip")           # list[str]

    manager.list_backbones()  → list of {"key", "backbone_id",
                                          "display_name", "loaded": bool}

    manager.is_known("clip")  → bool
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import faiss  # type: ignore

log = logging.getLogger("backbone_manager")


# ─────────────────────────────────────────────────────────────────────
#  Registry — single source of truth, must match embedders/__init__.py
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BackboneSpec:
    key:          str
    backbone_id:  str
    display_name: str


BACKBONES: dict[str, BackboneSpec] = {
    "clip":         BackboneSpec("clip",         "clip_openai_b32",
                                 "CLIP ViT-B/32 (OpenAI)"),
    "openclip_b32": BackboneSpec("openclip_b32", "openclip_laion2b_vitb32",
                                 "OpenCLIP ViT-B/32 (LAION-2B)"),
    "openclip_l14": BackboneSpec("openclip_l14", "openclip_laion2b_vitl14",
                                 "OpenCLIP ViT-L/14 (LAION-2B)"),
    "blip":         BackboneSpec("blip",         "blip_itm_base",
                                 "BLIP ITM-base (COCO)"),
}

DEFAULT_BACKBONE = os.environ.get("MIR_DEFAULT_BACKBONE", "clip")
MAX_BACKBONES    = max(1, int(os.environ.get("MIR_MAX_BACKBONES", "1")))
H5_DIR           = Path(os.environ.get("MIR_H5_DIR", "descriptors/CLIP_Flickr8k"))


# ─────────────────────────────────────────────────────────────────────
#  Cached payload (embedder + FAISS indices + identifier lists) — one
#  per backbone key. We bundle them together so the LRU evicts as a
#  unit, since they're useless without each other.
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _BackboneAssets:
    embedder:       object              # whatever get_embedder returns
    image_index:    faiss.Index
    caption_index:  faiss.Index
    image_files:    list[str]
    captions:       list[str]
    caption_filenames: list[str]   # filenames corresponding to captions built 
    embedding_dim:  int


# ─────────────────────────────────────────────────────────────────────
#  H5 loader (mirrors compare_backbones.py — same defensive key probing)
# ─────────────────────────────────────────────────────────────────────
def _load_h5(path: Path) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        emb_key = next((k for k in ("embeddings", "features", "vectors") if k in keys), None)
        if emb_key is None:
            raise KeyError(f"No embeddings dataset in {path} (keys={keys})")
        embeddings = f[emb_key][:]
        id_key = next((k for k in ("filenames", "captions", "ids", "labels") if k in keys), None)
        if id_key is None:
            raise KeyError(f"No identifier dataset in {path} (keys={keys})")
        raw = f[id_key][:]
        identifiers = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]
    return embeddings, identifiers


def _build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vectors = vectors.astype(np.float32, copy=False)
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx


# ─────────────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────────────
class BackboneManager:
    """Thread-safe LRU cache for VLM backbones + their FAISS indices."""

    def __init__(self, max_backbones: int = MAX_BACKBONES,
                 h5_dir: Path = H5_DIR) -> None:
        self._max         = max_backbones
        self._h5_dir      = h5_dir
        self._cache: "OrderedDict[str, _BackboneAssets]" = OrderedDict()
        self._lock        = threading.RLock()
        log.info("BackboneManager initialised · MAX_BACKBONES=%d · h5_dir=%s",
                 max_backbones, h5_dir)

    # ───── Public ─────
    def is_known(self, key: str) -> bool:
        return key in BACKBONES

    def list_backbones(self) -> list[dict]:
        """For the /backbones API endpoint that powers the dropdown."""
        with self._lock:
            return [
                {
                    "key":          spec.key,
                    "backbone_id":  spec.backbone_id,
                    "display_name": spec.display_name,
                    "loaded":       spec.key in self._cache,
                    "active":       False,  # the route fills this from session
                }
                for spec in BACKBONES.values()
            ]
    def get_caption_filenames(self, key: str) -> list[str]:
        return self._get_assets(key).caption_filenames
        
    def get_embedder(self, key: str):
        return self._get_assets(key).embedder

    def get_image_index(self, key: str) -> faiss.Index:
        return self._get_assets(key).image_index

    def get_caption_index(self, key: str) -> faiss.Index:
        return self._get_assets(key).caption_index

    def get_image_files(self, key: str) -> list[str]:
        return self._get_assets(key).image_files

    def get_captions(self, key: str) -> list[str]:
        return self._get_assets(key).captions

    def get_embedding_dim(self, key: str) -> int:
        return self._get_assets(key).embedding_dim

    def preload(self, key: str) -> None:
        """Eagerly load a backbone (useful for warmup in HF Spaces)."""
        self._get_assets(key)

    def unload_all(self) -> None:
        with self._lock:
            self._cache.clear()
            gc.collect()

    # ───── Internals ─────
    def _get_assets(self, key: str) -> _BackboneAssets:
        if key not in BACKBONES:
            raise KeyError(
                f"Unknown backbone '{key}'. Known: {list(BACKBONES.keys())}"
            )

        with self._lock:
            assets = self._cache.get(key)
            if assets is not None:
                self._cache.move_to_end(key)  # touch — most recently used
                log.debug("Cache hit: %s", key)
                return assets

            # Cache miss — load.
            log.info("Loading backbone '%s' …", key)
            assets = self._load(key)

            self._cache[key] = assets
            self._cache.move_to_end(key)

            # Evict LRU entries beyond the cap.
            while len(self._cache) > self._max:
                evicted_key, _ = self._cache.popitem(last=False)
                log.info("Evicted LRU backbone '%s' (cache size > %d)",
                         evicted_key, self._max)
            gc.collect()
            return assets

    def _load(self, key: str) -> _BackboneAssets:
        spec = BACKBONES[key]

        img_h5  = self._h5_dir / f"embeddings_{spec.backbone_id}.h5"
        capt_h5 = self._h5_dir / f"captions_{spec.backbone_id}.h5"

        img_embeds,  img_files  = _load_h5(img_h5)
        with h5py.File(capt_h5, "r") as f:
            capt_embeds = f["embeddings"][:]
            captions = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in f["captions"][:]
            ]
            caption_filenames = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in f["filenames"][:]
            ]
        log.info("  %s · %d images · %d captions · dim=%d",
                 spec.backbone_id, len(img_files), len(captions),
                 img_embeds.shape[1])

        image_index   = _build_faiss_index(img_embeds.copy())
        caption_index = _build_faiss_index(capt_embeds.copy())

        # Local import — keep torch off the import-time hot path.
        from embedders import get_embedder
        embedder = get_embedder(key)

        return _BackboneAssets(
            embedder      = embedder,
            image_index   = image_index,
            caption_index = caption_index,
            image_files   = img_files,
            captions      = captions,
            embedding_dim = int(img_embeds.shape[1]),
            caption_filenames = caption_filenames,
        )


# ─────────────────────────────────────────────────────────────────────
#  Module-level singleton — import this from app.py
# ─────────────────────────────────────────────────────────────────────
manager = BackboneManager()


__all__ = ["manager", "BackboneManager", "BackboneSpec",
           "BACKBONES", "DEFAULT_BACKBONE", "MAX_BACKBONES"]
