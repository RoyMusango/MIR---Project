"""
Build CLIP Embeddings Database
===============================
Pre-computes CLIP image embeddings for all even-first-digit images in the 
dataset_voitures/dataset/ directory and saves them as individual .txt files
in descriptors/CLIP/ (compatible with the existing search pipeline).

Also creates an HDF5 database at descriptors/CLIP/clip_embeddings.h5 for
fast batch operations.

Usage:
    python build_clip_database.py
"""

import os
import sys
import time
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")
CLIP_DIR = os.path.join(DESCRIPTORS_PATH, "CLIP")


def main():
    # Delayed imports so --help is fast
    from tqdm import tqdm
    from clip_model import CLIPEmbedder

    print("=" * 60)
    print("  CLIP Embeddings Database Builder")
    print("=" * 60)

    # ── Validate dataset ─────────────────────────────────────────────────
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        sys.exit(1)

    # Collect even-first-digit images (matching existing descriptor convention)
    all_files = sorted(os.listdir(DATASET_PATH))
    image_files = [
        f for f in all_files
        if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and f[0].isdigit() and int(f[0]) % 2 == 0
    ]

    print(f"[INFO] Found {len(image_files)} even-first-digit images in dataset")

    if len(image_files) == 0:
        print("[ERROR] No matching images found. Check the dataset path.")
        sys.exit(1)

    # ── Create output directory ──────────────────────────────────────────
    os.makedirs(CLIP_DIR, exist_ok=True)

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
            
            # Save individual .txt file (compatible with existing search pipeline)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(CLIP_DIR, f"{base_name}_CLIP.txt")
            np.savetxt(txt_path, embedding)

    elapsed = time.time() - start_time
    
    print(f"\n[INFO] Extracted {len(all_embeddings)} embeddings in {elapsed:.1f}s")
    print(f"[INFO] Saved individual .txt files to: {CLIP_DIR}")

    # ── Save HDF5 database ───────────────────────────────────────────────
    try:
        import h5py
        
        h5_path = os.path.join(CLIP_DIR, "clip_embeddings.h5")
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings_matrix)
            # Store filenames as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            filenames_ds = f.create_dataset("filenames", (len(all_filenames),), dtype=dt)
            for i, name in enumerate(all_filenames):
                filenames_ds[i] = name
        
        print(f"[INFO] Saved HDF5 database to: {h5_path}")
        print(f"[INFO] Shape: {embeddings_matrix.shape}")
    except ImportError:
        print("[WARN] h5py not installed, skipping HDF5 database creation.")

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(all_embeddings)} CLIP descriptors saved.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
