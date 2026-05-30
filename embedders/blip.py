"""BLIP embedder — Salesforce BLIP image-text retrieval model.

Uses the ITC (Image-Text Contrastive) heads from BlipForImageTextRetrieval,
which produce a shared embedding space comparable to CLIP's dual-encoder.
Drop-in replacement via the BaseEmbedder contract.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

from .base import BaseEmbedder


class BLIPEmbedder(BaseEmbedder):
    """
    Wraps Salesforce/blip-itm-base-coco for dense image-text retrieval.

    Produces 256-dimensional L2-normalized feature vectors.
    Note: embedding_dim=256, unlike CLIP's 512 — FAISS indices built
    with this backbone are not interchangeable with CLIP indices.
    """

    MODEL_NAME = "Salesforce/blip-itm-base-coco"
    backbone_id = "blip_itm_base"

    def __init__(self):
        print(f"[BLIP] Loading model: {self.MODEL_NAME} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BLIP] Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        # Probe embedding dim via vision config
        self.embedding_dim = self.model.config.projection_dim  # 256
        print(f"[BLIP] Model ready · dim={self.embedding_dim}")

    # ── single-item encoders ────────────────────────────────────────────
    def encode_image(self, image) -> np.ndarray:
        pil = self._to_pil(image)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            feats = self.model.vision_proj(vision_out.last_hidden_state[:, 0, :])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_out = self.model.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            feats = self.model.text_proj(text_out.last_hidden_state[:, 0, :])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().flatten().astype(np.float32)

    # ── batch encoders ──────────────────────────────────────────────────
    def encode_images_batch(self, image_paths: list, batch_size: int = 32) -> list:
        results = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            pil_images, valid_paths = [], []
            for p in batch_paths:
                try:
                    pil_images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[BLIP] Skipping {p}: {e}")

            if not pil_images:
                continue

            inputs = self.processor(
                images=pil_images, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                feats = self.model.vision_proj(vision_out.last_hidden_state[:, 0, :])
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings = feats.cpu().numpy().astype(np.float32)

            for path, emb in zip(valid_paths, embeddings):
                results.append((os.path.basename(path), emb))
        return results

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            text_out = self.model.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            feats = self.model.text_proj(text_out.last_hidden_state[:, 0, :])
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