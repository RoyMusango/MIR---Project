"""OpenCLIP embedder — LAION-trained CLIP variants via the open_clip library.

Implements the BaseEmbedder contract so it's drop-in compatible with the
HF CLIP backbone for FAISS indexing and the Flask retrieval routes.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
import open_clip

from .base import BaseEmbedder


class OpenCLIPEmbedder(BaseEmbedder):
    """
    Wraps an OpenCLIP dual-encoder.

    Default config matches the standard LAION-2B ViT-B/32 release, which
    is the closest 1:1 comparison to openai/clip-vit-base-patch32 in terms
    of architecture (same dim=512) but different training data.
    """

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.backbone_id = f"openclip_{pretrained.split('_')[0]}_{model_name.lower().replace('-', '')}"

        print(f"[OpenCLIP] Loading {model_name} ({pretrained}) ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[OpenCLIP] Using device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Probe dim with a dummy forward — robust across model variants.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

        print(f"[OpenCLIP] Model ready · dim={self.embedding_dim}")

    # ── single-item encoders ────────────────────────────────────────────
    def encode_image(self, image) -> np.ndarray:
        pil = self._to_pil(image)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().flatten().astype(np.float32)

    # ── batch encoders ──────────────────────────────────────────────────
    def encode_images_batch(self, image_paths: list, batch_size: int = 32) -> list:
        results = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors, valid_paths = [], []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[OpenCLIP] Skipping {p}: {e}")

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings = feats.cpu().numpy().astype(np.float32)

            for path, emb in zip(valid_paths, embeddings):
                results.append((os.path.basename(path), emb))
        return results

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            import cv2
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")