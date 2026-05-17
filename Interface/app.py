import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

# make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import functions as f

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flickr8k paths
FLICKR8K_DATASET_PATH = os.path.join(PROJECT_ROOT, "Flickr8k_Dataset", "Flicker8k_Dataset")
FLICKR8K_H5_PATH = os.path.join(DESCRIPTORS_PATH, "CLIP_Flickr8k", "clip_flickr8k.h5")

# descriptor name -> (folder, function, algo_id) — CLIP removed
DESCRIPTOR_MAP = {
    "Color Histogram": ("Hist_Col", f.generateHistogramme_Color, 1),
    "HSV Histogram":   ("HSV",      f.generateHistogramme_HSV,   2),
    "GLCM (Texture)":  ("GLCM",     f.generateGLCM,             5),
    "LBP (Texture)":   ("LBP",      f.generateLBP,              6),
    "HOG (Shape)":     ("HOG",      f.generateHOG,              7),
    "SIFT (Keypoints)":("SIFT",     f.generateSIFT,             3),
    "ORB (Keypoints)": ("ORB",      f.generateORB,              4),
}

VECTOR_DESCRIPTORS = ["Color Histogram", "HSV Histogram", "GLCM (Texture)", "LBP (Texture)", "HOG (Shape)"]
KEYPOINT_DESCRIPTORS = ["SIFT (Keypoints)", "ORB (Keypoints)"]

VECTOR_DISTANCES = ["Euclidienne", "Chi carre", "Correlation", "Intersection", "Bhattacharyya"]
KEYPOINT_DISTANCES = ["Brute force", "Flann"]

HIGHER_IS_BETTER = {"Correlation", "Intersection", "Brute force", "Flann"}

# ── Lazy-loaded Flickr8k CLIP data ──────────────────────────────────────
_flickr8k_data = None


def load_flickr8k_embeddings():
    """Load pre-computed Flickr8k CLIP embeddings from HDF5 (lazy singleton)."""
    global _flickr8k_data
    if _flickr8k_data is not None:
        return _flickr8k_data

    import h5py
    if not os.path.exists(FLICKR8K_H5_PATH):
        raise FileNotFoundError(
            f"Flickr8k CLIP embeddings not found at {FLICKR8K_H5_PATH}. "
            "Run 'python build_flickr8k_clip_db.py' first."
        )

    print(f"[CLIP] Loading Flickr8k embeddings from {FLICKR8K_H5_PATH}...")
    with h5py.File(FLICKR8K_H5_PATH, "r") as hf:
        embeddings = hf["embeddings"][:]
        filenames = [fn.decode("utf-8") if isinstance(fn, bytes) else fn for fn in hf["filenames"][:]]

    _flickr8k_data = (filenames, embeddings)
    print(f"[CLIP] Loaded {len(filenames)} Flickr8k embeddings ({embeddings.shape})")
    return _flickr8k_data


def get_image_class(filename):
    """get class from filename like '0_0_BMW_Serie3_100.jpg' -> '0_0'"""
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def run_search(query_path, descriptor_keys, distance_name, k):
    """main search function, returns results + metrics"""
    start = time.time()
    query_class = get_image_class(query_path)

    img = cv2.imread(query_path)
    if img is None:
        return None, "Could not read the query image."

    num_descriptors = len(descriptor_keys)
    is_keypoint = descriptor_keys[0] in KEYPOINT_DESCRIPTORS

    all_distances_per_descriptor = []
    total_relevant = 0

    for d_idx, desc_key in enumerate(descriptor_keys):
        desc_folder, desc_func, algo_id = DESCRIPTOR_MAP[desc_key]
        desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)

        # extract query features
        query_features = desc_func(img)
        if query_features is None:
            return None, f"Could not extract {desc_key} features."

        # load precomputed descriptors (even-first-digit only)
        all_files = [ff for ff in os.listdir(desc_dir) if ff.endswith(".txt")]
        desc_files = [ff for ff in all_files if ff[0].isdigit() and int(ff[0]) % 2 == 0]
        total = len(desc_files)
        if total == 0:
            return None, f"No descriptors found in {desc_dir}"

        # count relevant images (only once)
        if d_idx == 0:
            total_relevant = sum(
                1 for ff in desc_files
                if get_image_class(ff.replace(f'_{desc_folder}.txt', '.jpg')) == query_class
            )

        results = []
        for fname in desc_files:
            fpath = os.path.join(desc_dir, fname)
            db_features = np.loadtxt(fpath)

            if is_keypoint:
                query_f = np.float32(query_features)
                db_f = np.float32(db_features)
                if query_f.ndim == 1:
                    query_f = query_f.reshape(1, -1)
                if db_f.ndim == 1:
                    db_f = db_f.reshape(1, -1)
                try:
                    dist = f.distance_f(query_f, db_f, distance_name)
                except Exception:
                    dist = 0
            else:
                dist = f.distance_f(query_features, db_features, distance_name)

            base = fname.replace(f"_{desc_folder}.txt", "")
            img_file = base + ".jpg"
            results.append((img_file, float(dist)))

        all_distances_per_descriptor.append(results)

    # single descriptor -> simple sort
    if num_descriptors == 1:
        results = all_distances_per_descriptor[0]
        if distance_name in HIGHER_IS_BETTER:
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results.sort(key=lambda x: x[1], reverse=False)

        elapsed = time.time() - start
        return build_response(results, query_class, total_relevant, elapsed, k, descriptor_keys), None

    # multiple descriptors -> normalize and average
    higher_is_better = distance_name in HIGHER_IS_BETTER

    combined = {}
    for d_idx, results in enumerate(all_distances_per_descriptor):
        for fname, dist in results:
            if fname not in combined:
                combined[fname] = [None] * num_descriptors
            combined[fname][d_idx] = dist

    combined = {k: v for k, v in combined.items() if all(x is not None for x in v)}
    if not combined:
        return None, "No common images found across selected descriptors."

    # min-max normalization per descriptor
    for d_idx in range(num_descriptors):
        vals = [combined[k][d_idx] for k in combined]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v + 1e-10
        for k in combined:
            normalized = (combined[k][d_idx] - min_v) / rng
            if higher_is_better:
                normalized = 1.0 - normalized
            combined[k][d_idx] = normalized

    final_results = []
    for fname, dists in combined.items():
        avg_dist = sum(dists) / len(dists)
        final_results.append((fname, avg_dist))

    final_results.sort(key=lambda x: x[1])

    elapsed = time.time() - start
    return build_response(final_results, query_class, total_relevant, elapsed, k, descriptor_keys), None


def build_response(all_results, query_class, total_relevant, elapsed, k, descriptor_keys):
    """format results into a dict for the frontend"""

    def compute_metrics(results_slice, total_rel):
        if not results_slice:
            return 0.0, 0.0
        relevant_found = sum(1 for fname, _ in results_slice if get_image_class(fname) == query_class)
        precision = relevant_found / len(results_slice)
        recall = relevant_found / total_rel if total_rel > 0 else 0
        return precision, recall

    p50, r50 = compute_metrics(all_results[:50], total_relevant)
    p100, r100 = compute_metrics(all_results[:100], total_relevant)

    top_k = all_results[:k]
    results_list = []
    for i, (fname, dist) in enumerate(top_k):
        results_list.append({
            "filename": fname,
            "distance": round(dist, 6),
            "rank": i + 1,
            "image_url": url_for('serve_dataset_image', filename=fname),
            "is_relevant": get_image_class(fname) == query_class
        })

    return {
        "results": results_list,
        "elapsed": round(elapsed, 2),
        "query_class": query_class,
        "total_relevant": total_relevant,
        "total_results": len(all_results),
        "k": k,
        "descriptors_used": ", ".join(descriptor_keys),
        "metrics": {
            "p50": round(p50 * 100, 1),
            "r50": round(r50 * 100, 1),
            "p100": round(p100 * 100, 1),
            "r100": round(r100 * 100, 1)
        }
    }


# -- Routes --

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset/<filename>')
def serve_dataset_image(filename):
    return send_from_directory(DATASET_PATH, filename)


@app.route('/flickr8k/<filename>')
def serve_flickr8k_image(filename):
    return send_from_directory(FLICKR8K_DATASET_PATH, filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/get_distances', methods=['POST'])
def get_distances():
    """return the right distance list depending on descriptor group"""
    data = request.get_json()
    descriptors = data.get('descriptors', [])

    if any(d in KEYPOINT_DESCRIPTORS for d in descriptors):
        return jsonify(KEYPOINT_DISTANCES)
    else:
        return jsonify(VECTOR_DISTANCES)


@app.route('/search', methods=['POST'])
def search():
    # get the uploaded file
    if 'query_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['query_image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # get search params
    descriptors = request.form.getlist('descriptors')
    distance = request.form.get('distance', 'Euclidienne')
    k = int(request.form.get('k', 10))

    if not descriptors:
        return jsonify({"error": "Select at least one descriptor"}), 400

    # run the actual search
    result, error = run_search(filepath, descriptors, distance, k)
    if error:
        return jsonify({"error": error}), 400

    # add the query image url so we can show it
    result["query_image_url"] = url_for('serve_upload', filename=filename)

    return jsonify(result)


@app.route('/search_text', methods=['POST'])
def search_text():
    """Multimodal text-to-image search using CLIP on Flickr8k."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    query = data.get('query', '').strip()
    k = int(data.get('k', 20))

    if not query:
        return jsonify({"error": "Please enter a text query"}), 400

    start = time.time()

    try:
        filenames, embeddings = load_flickr8k_embeddings()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Encode text query
    from clip_model import get_embedder
    embedder = get_embedder()
    text_embedding = embedder.encode_text(query)

    # Compute cosine similarity (embeddings are already L2-normalized)
    similarities = embeddings @ text_embedding

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]

    elapsed = time.time() - start

    results_list = []
    for rank, idx in enumerate(top_indices):
        results_list.append({
            "filename": filenames[idx],
            "similarity": round(float(similarities[idx]), 6),
            "rank": rank + 1,
            "image_url": url_for('serve_flickr8k_image', filename=filenames[idx]),
        })

    return jsonify({
        "results": results_list,
        "elapsed": round(elapsed, 2),
        "total_results": len(filenames),
        "query": query,
        "k": k,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
