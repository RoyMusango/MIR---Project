"""Backward-compat shim — use `from embedders import get_embedder` in new code."""
from embedders import get_embedder

__all__ = ["get_embedder"]