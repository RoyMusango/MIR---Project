import os
import sys
import numpy as np
import h5py
import argparse
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from embedders import get_embedder

# Paths
CAPTIONS_FILE = os.path.join(PROJECT_ROOT, "Flickr8k_text", "Flickr8k.token.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "descriptors", "CLIP_Flickr8k")
# OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clip_flickr8k_captions.h5")

BATCH_SIZE = 64

def load_captions(captions_file):
    """Parse Flickr8k.token.txt -> list of (image_filename, caption)"""
    pairs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "image.jpg#0\tcaption text"
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            img_id, caption = parts
            img_filename = img_id.split("#")[0]
            pairs.append((img_filename, caption))
    return pairs

def main(args): # args is the backbone name
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading captions...")
    pairs = load_captions(CAPTIONS_FILE)
    print(f"Found {len(pairs)} captions")

    embedder = get_embedder(args.backbone)
    output_path = os.path.join(OUTPUT_DIR, f"captions_{embedder.backbone_id}.h5")

    all_embeddings = []
    all_captions = []
    all_filenames = []

    # Process in batches
    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="Encoding captions"):
        batch = pairs[i:i + BATCH_SIZE]
        batch_captions = [p[1] for p in batch]
        batch_filenames = [p[0] for p in batch]

        embeddings = embedder.encode_texts_batch(batch_captions)
        all_embeddings.append(embeddings)
        all_captions.extend(batch_captions)
        all_filenames.extend(batch_filenames)

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)

    print(f"Saving {len(all_captions)} caption embeddings to {output_path}...")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("embeddings", data=all_embeddings)
        hf.create_dataset("captions",   data=np.array(all_captions,   dtype=h5py.string_dtype()))
        hf.create_dataset("filenames",  data=np.array(all_filenames,  dtype=h5py.string_dtype()))

    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Flickr8k CLIP caption embeddings.")
    parser.add_argument("--backbone", default="clip",
                    choices=["clip", "openclip_b32", "openclip_l14", "blip"],
                    help="Embedder backbone to use.")
    args = parser.parse_args()
    main(args)