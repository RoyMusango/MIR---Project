"""Abstract base class for multimodal embedders.

All embedders produce L2-normalized float32 vectors with the same interface,
so they can be swapped via the factory without changing downstream code
(FAISS indexing, Flask routes, evaluation script).
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Contract for any image/text dual-encoder used in the retrieval engine."""

    #: Embedding dimensionality — must be set by subclasses after model load.
    embedding_dim: int

    #: Short identifier used in H5 filenames and CSV reports (e.g. 'clip_openai_b32').
    backbone_id: str

    @abstractmethod
    def encode_image(self, image) -> np.ndarray:
        """Encode a single image to a 1-D L2-normalized float32 vector."""

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query to a 1-D L2-normalized float32 vector."""

    @abstractmethod
    def encode_images_batch(self, image_paths: list, batch_size: int = 32) -> list:
        """Encode many images. Returns list of (basename, vector) tuples."""

    @abstractmethod
    def encode_texts_batch(self, texts: list) -> np.ndarray:
        """Encode many texts. Returns a 2-D array of shape (N, embedding_dim)."""