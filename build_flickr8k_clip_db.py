"""
Build CLIP Embeddings Database for Flickr8k
============================================
Pre-computes CLIP image embeddings for all images in the 
Flickr8k_Dataset/Flicker8k_Dataset/ directory and saves them as an HDF5 file
at descriptors/CLIP_Flickr8k/clip_flickr8k.h5.

Usage:
    python build_flickr8k_clip_db.py
"""

import os
import sys
import time
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Flickr8k_Dataset", "Flicker8k_Dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")
CLIP_FLICKR8K_DIR = os.path.join(DESCRIPTORS_PATH, "CLIP_Flickr8k")


def main():
    from tqdm import tqdm
    from clip_model import CLIPEmbedder

    print("=" * 60)
    print("  CLIP Flickr8k Embeddings Database Builder")
    print("=" * 60)

    # ── Validate dataset ─────────────────────────────────────────────────
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        sys.exit(1)

    # Collect all images
    all_files = sorted(os.listdir(DATASET_PATH))
    image_files = [
        f for f in all_files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"[INFO] Found {len(image_files)} images in Flickr8k dataset")

    if len(image_files) == 0:
        print("[ERROR] No images found. Check the dataset path.")
        sys.exit(1)

    # ── Create output directory ──────────────────────────────────────────
    os.makedirs(CLIP_FLICKR8K_DIR, exist_ok=True)

    # ── Load CLIP model ──────────────────────────────────────────────────
    embedder = CLIPEmbedder()

    # ── Extract embeddings in batches ────────────────────────────────────
    batch_size = 32
    image_paths = [os.path.join(DATASET_PATH, f) for f in image_files]

    all_filenames = []
    all_embeddings = []

    start_time = time.time()

    print(f"\n[INFO] Extracting CLIP embeddings (batch_size={batch_size})...")

    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Batches"):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        results = embedder.encode_images_batch(batch_paths, batch_size=batch_size)

        for filename, embedding in results:
            all_filenames.append(filename)
            all_embeddings.append(embedding)

    elapsed = time.time() - start_time

    print(f"\n[INFO] Extracted {len(all_embeddings)} embeddings in {elapsed:.1f}s")

    # ── Save HDF5 database ───────────────────────────────────────────────
    try:
        import h5py

        h5_path = os.path.join(CLIP_FLICKR8K_DIR, "clip_flickr8k.h5")
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings_matrix)
            dt = h5py.special_dtype(vlen=str)
            filenames_ds = f.create_dataset("filenames", (len(all_filenames),), dtype=dt)
            for i, name in enumerate(all_filenames):
                filenames_ds[i] = name

        print(f"[INFO] Saved HDF5 database to: {h5_path}")
        print(f"[INFO] Shape: {embeddings_matrix.shape}")
    except ImportError:
        print("[ERROR] h5py is required. Install with: pip install h5py")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(all_embeddings)} Flickr8k CLIP embeddings saved.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
