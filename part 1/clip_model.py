"""
CLIP Embedder — Provides image embedding extraction using OpenAI's CLIP model.
Used as a deep-learning descriptor for image-to-image retrieval in the MIR project.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Singleton CLIPEmbedder ──────────────────────────────────────────────────
_embedder_instance = None


def get_embedder():
    """Get or create the global CLIPEmbedder instance (lazy singleton)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = CLIPEmbedder()
    return _embedder_instance


class CLIPEmbedder:
    """
    Wraps HuggingFace's openai/clip-vit-base-patch32 model for image embedding.
    
    Produces 512-dimensional L2-normalized feature vectors.
    Automatically uses CUDA if available, otherwise CPU.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        print(f"[CLIP] Loading model: {self.MODEL_NAME} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        print("[CLIP] Model loaded successfully.")

    def encode_image(self, image) -> np.ndarray:
        """
        Encode a single image into a 512-dim normalized feature vector.
        
        Parameters
        ----------
        image : np.ndarray or PIL.Image or str
            If np.ndarray (BGR from OpenCV), it will be converted to RGB PIL Image.
            If str, it is treated as a file path.
            If PIL.Image, used directly.
            
        Returns
        -------
        np.ndarray
            A 1-D float32 array of shape (512,), L2-normalized.
        """
        pil_image = self._to_pil(image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=pixel_values)
        
        # Handle newer transformers that return BaseModelOutputWithPooling
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into a 512-dim normalized feature vector.
        
        Parameters
        ----------
        text : str
            The text query to encode.
            
        Returns
        -------
        np.ndarray
            A 1-D float32 array of shape (512,), L2-normalized.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten().astype(np.float32)

    def encode_images_batch(self, image_paths: list, batch_size: int = 32) -> list:
        """
        Encode a batch of images into 512-dim normalized feature vectors.
        
        Parameters
        ----------
        image_paths : list of str
            List of file paths to images.
        batch_size : int
            Number of images to process at once.
            
        Returns
        -------
        list of (filename, np.ndarray)
            Each entry is (basename, 512-dim vector).
        """
        import os
        results = []
        
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            pil_images = []
            valid_paths = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[CLIP] Skipping {p}: {e}")
            
            if not pil_images:
                continue
                
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=pixel_values)
            
            # Handle newer transformers that return BaseModelOutputWithPooling
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings = image_features.cpu().numpy().astype(np.float32)
            
            for path, emb in zip(valid_paths, embeddings):
                results.append((os.path.basename(path), emb))
        
        return results

    @staticmethod
    def _to_pil(image):
        """Convert various image formats to PIL RGB Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
