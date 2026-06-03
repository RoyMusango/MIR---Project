import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# make project root importable — must come before any local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("MIR_H5_DIR", os.path.join(PROJECT_ROOT, "descriptors", "CLIP_Flickr8k"))

import torch
import math
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import faiss
from pathlib import Path
from PIL import Image
from backbone_manager import manager, BACKBONES, DEFAULT_BACKBONE
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from flask import redirect

import functions as f

from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get("MIR_SECRET_KEY", "mir-static-fallback-2026")
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
csrf = CSRFProtect(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = ""


class _User(UserMixin):
    def __init__(self, uid: str):
        self.id = uid


@login_manager.user_loader
def _load_user(uid: str) -> _User | None:
    return _User(uid) if uid == os.environ.get("MIR_USERNAME", "admin") else None

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flickr8k — images live inside the Flicker8k_Dataset subdirectory (original Project 1 layout)
FLICKR8K_DATASET_PATH = os.path.join(PROJECT_ROOT, "Flickr8k_Dataset", "Flicker8k_Dataset")



DESCRIPTOR_MAP = {
    "Color Histogram":          ("Hist_Col", f.generateHistogramme_Color, 1),
    "HSV Histogram":            ("HSV",      f.generateHistogramme_HSV,   2),
    "GLCM (Texture)":           ("GLCM",     f.generateGLCM,              5),
    "LBP (Texture)":            ("LBP",      f.generateLBP,               6),
    "HOG (Shape)":              ("HOG",      f.generateHOG,               7),
    "SIFT (Keypoints)":         ("SIFT",     f.generateSIFT,              3),
    "ORB (Keypoints)":          ("ORB",      f.generateORB,               4),
    "ViT (Deep Learning)":      ("ViT",      f.generateViT,               8),
    "ResNet50 (Deep Learning)": ("ResNet",   f.generateResNet,            9),
    "CLIP (OpenAI)":            ("CLIP",     f.generateCLIP,             10),
}

VECTOR_DESCRIPTORS   = ["Color Histogram", "HSV Histogram", "GLCM (Texture)", "LBP (Texture)", "HOG (Shape)"]
KEYPOINT_DESCRIPTORS = ["SIFT (Keypoints)", "ORB (Keypoints)"]
DL_DESCRIPTORS       = ["ViT (Deep Learning)", "ResNet50 (Deep Learning)", "CLIP (OpenAI)"]

VECTOR_DISTANCES   = ["Euclidienne", "Chi carre", "Correlation", "Intersection", "Bhattacharyya"]
KEYPOINT_DISTANCES = ["Brute force", "Flann"]
DL_DISTANCES       = ["Cosine Similarity", "Euclidienne"]

HIGHER_IS_BETTER = {"Correlation", "Intersection", "Brute force", "Flann", "Cosine Similarity"}




def get_image_class(filename, manual_class=None):
    """Extract class from filename like '0_1_BMW_X3_207.jpg' -> '0_1'."""
    if manual_class:
        return manual_class.strip()
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def run_search(query_path, descriptor_keys, distance_name, k, manual_class=None):
    """Main search function — returns (result_dict, error_string)."""
    global_start = time.time()
    query_class = get_image_class(query_path, manual_class)

    img = cv2.imread(query_path)
    if img is None:
        return None, "Could not read the query image."

    num_descriptors = len(descriptor_keys)
    all_distances_per_descriptor = []
    higher_is_better_per_desc = []
    total_relevant = 0

    for d_idx, desc_key in enumerate(descriptor_keys):
        desc_folder, desc_func, algo_id = DESCRIPTOR_MAP[desc_key]
        desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)

        query_features = desc_func(img)
        if query_features is None:
            return None, f"Could not extract {desc_key} features."

        all_files = [ff for ff in os.listdir(desc_dir) if ff.endswith(".txt")]
        desc_files = [ff for ff in all_files if ff[0].isdigit() and int(ff[0]) % 2 == 0]
        if not desc_files:
            return None, f"No descriptors found in {desc_dir}"

        if d_idx == 0:
            total_relevant = sum(
                1 for ff in desc_files
                if get_image_class(ff.replace(f'_{desc_folder}.txt', '.jpg')) == query_class
            )

        is_keypoint_desc = desc_key in KEYPOINT_DESCRIPTORS
        is_binary = desc_key == "ORB (Keypoints)"

        if is_keypoint_desc:
            effective_distance = distance_name if distance_name in ["Brute force", "Flann"] else "Brute force"
            higher_is_better_per_desc.append(True)
        else:
            effective_distance = distance_name
            higher_is_better_per_desc.append(distance_name in HIGHER_IS_BETTER)

        results = []
        for fname in desc_files:
            fpath = os.path.join(desc_dir, fname)
            db_features = np.loadtxt(fpath)
            if is_keypoint_desc:
                query_f = np.float32(query_features)
                db_f = np.float32(db_features)
                if query_f.ndim == 1: query_f = query_f.reshape(1, -1)
                if db_f.ndim == 1:   db_f = db_f.reshape(1, -1)
                try:
                    dist = f.distance_f(query_f, db_f, effective_distance, is_binary=is_binary)
                except Exception:
                    dist = 0
            else:
                try:
                    dist = f.distance_f(query_features, db_features, distance_name)
                except Exception:
                    dist = 0
            base = fname.replace(f"_{desc_folder}.txt", "")
            results.append((base + ".jpg", float(dist)))
        all_distances_per_descriptor.append(results)

    if num_descriptors == 1:
        results = all_distances_per_descriptor[0]
        results.sort(key=lambda x: x[1], reverse=higher_is_better_per_desc[0])
        elapsed = time.time() - global_start
        return build_response(results, query_class, total_relevant, elapsed, k, descriptor_keys), None

    combined = {}
    for d_idx, results in enumerate(all_distances_per_descriptor):
        for fname, dist in results:
            if fname not in combined:
                combined[fname] = [None] * num_descriptors
            combined[fname][d_idx] = dist

    combined = {img_key: v for img_key, v in combined.items() if all(x is not None for x in v)}
    if not combined:
        return None, "No common images found across selected descriptors."

    for d_idx in range(num_descriptors):
        vals = [combined[img_key][d_idx] for img_key in combined]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v + 1e-10
        for img_key in combined:
            normalized = (combined[img_key][d_idx] - min_v) / rng
            if higher_is_better_per_desc[d_idx]:
                normalized = 1.0 - normalized
            combined[img_key][d_idx] = normalized

    final_results = [(fname, sum(dists) / len(dists)) for fname, dists in combined.items()]
    final_results.sort(key=lambda x: x[1])
    elapsed = time.time() - global_start
    return build_response(final_results, query_class, total_relevant, elapsed, k, descriptor_keys), None


def build_response(all_results, query_class, total_relevant, elapsed, k, descriptor_keys):
    """Format results + compute metrics + build P-R curve for the frontend."""

    def compute_metrics(results_slice, total_rel):
        if not results_slice:
            return 0.0, 0.0, 0.0, 0.0
        relevant_found = sum(1 for fname, _ in results_slice if get_image_class(fname) == query_class)
        prec = relevant_found / len(results_slice)
        rec  = relevant_found / total_rel if total_rel > 0 else 0.0
        hits, s = 0, 0.0
        for i, (fname, _) in enumerate(results_slice, 1):
            if get_image_class(fname) == query_class:
                hits += 1
                s += hits / i
        ap = s / total_rel if total_rel > 0 else 0.0
        r = min(total_rel, len(results_slice))
        r_prec_hits = sum(1 for fname, _ in results_slice[:r] if get_image_class(fname) == query_class)
        r_precision = r_prec_hits / r if r > 0 else 0.0
        return prec, rec, ap, r_precision

    p50,  r50,  ap50,  rprec50  = compute_metrics(all_results[:50],  total_relevant)
    p100, r100, ap100, rprec100 = compute_metrics(all_results[:100], total_relevant)

    # mAP = AP on the full ranked list (single-query scenario)
    map_score = 0.0
    if total_relevant > 0 and all_results:
        hits, s = 0, 0.0
        for i, (fname, _) in enumerate(all_results, 1):
            if get_image_class(fname) == query_class:
                hits += 1
                s += hits / i
        map_score = s / total_relevant

    # Precision-Recall curve (sampled at every rank up to 300)
    pr_curve = []
    max_k_curve = min(len(all_results), 300)
    cumulative_relevant = 0
    for rank_i in range(1, max_k_curve + 1):
        fname, _ = all_results[rank_i - 1]
        if get_image_class(fname) == query_class:
            cumulative_relevant += 1
        if rank_i <= 50 or (rank_i <= 200 and rank_i % 5 == 0) or rank_i % 10 == 0:
            p = cumulative_relevant / rank_i
            r = cumulative_relevant / total_relevant if total_relevant > 0 else 0
            pr_curve.append({"k": rank_i, "precision": round(p * 100, 1), "recall": round(r * 100, 1)})

    top_k = all_results[:k]
    results_list = []
    for i, (fname, dist) in enumerate(top_k):
        results_list.append({
            "filename":   fname,
            "distance":   round(dist, 6),
            "rank":       i + 1,
            "image_url":  url_for('serve_dataset_image', filename=fname),
            "is_relevant": get_image_class(fname) == query_class,
        })

    return {
        "results":        results_list,
        "elapsed":        round(elapsed, 2),
        "query_class":    query_class,
        "total_relevant": total_relevant,
        "total_results":  len(all_results),
        "k":              k,
        "descriptors_used": ", ".join(descriptor_keys),
        "metrics": {
            "p50":      round(p50  * 100, 1),
            "r50":      round(r50  * 100, 1),
            "ap50":     round(ap50 * 100, 1),
            "rprec50":  round(rprec50  * 100, 1),
            "p100":     round(p100  * 100, 1),
            "r100":     round(r100  * 100, 1),
            "ap100":    round(ap100 * 100, 1),
            "rprec100": round(rprec100 * 100, 1),
            "map":      round(map_score * 100, 1),
        },
        "pr_curve": pr_curve,
    }


# -- Routes --

@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/dataset/<path:filename>')
@login_required
def serve_dataset_image(filename):
    safe = os.path.abspath(os.path.join(DATASET_PATH, filename))
    if not safe.startswith(os.path.abspath(DATASET_PATH) + os.sep):
        return "Forbidden", 403
    return send_from_directory(DATASET_PATH, filename)


@app.route('/api/dataset_images')
@login_required
def get_dataset_images():
    page = max(1, request.args.get('page', 1, type=int))
    per_page = 40
    try:
        files = []
        for root, _, fs in os.walk(DATASET_PATH):
            for f_ in fs:
                if allowed_file(f_):
                    rel = os.path.relpath(os.path.join(root, f_), DATASET_PATH)
                    files.append(rel.replace(os.sep, '/'))
        files.sort()
        total = len(files)
        pages = max(1, math.ceil(total / per_page))
        start = (page - 1) * per_page
        return jsonify({"files": files[start:start+per_page], "page": page,
                        "total_pages": pages, "total_files": total})
    except Exception as e:
        app.logger.error("dataset_images: %s", e)
        return jsonify({"error": "Could not read dataset."}), 500



@login_required
@app.route('/flickr8k_image/<path:filename>')
def serve_flickr8k_image(filename):
    import os
    target_dir = os.path.join(PROJECT_ROOT, "Flickr8k_Dataset", "Flicker8k_Dataset")
    return send_from_directory(target_dir, filename)


@app.route('/api/flickr8k_images')
@login_required
def get_flickr8k_images():
    page = max(1, request.args.get('page', 1, type=int))
    per_page = 40
    try:
        files = sorted([f_ for f_ in os.listdir(FLICKR8K_DATASET_PATH)
                        if allowed_file(f_) and os.path.isfile(os.path.join(FLICKR8K_DATASET_PATH, f_))])
        total = len(files)
        pages = max(1, math.ceil(total / per_page))
        start = (page - 1) * per_page
        return jsonify({"files": files[start:start+per_page], "page": page,
                        "total_pages": pages, "total_files": total})
    except Exception as e:
        app.logger.error("flickr8k_images: %s", e)
        return jsonify({"error": "Could not read Flickr8k."}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/get_distances', methods=['POST'])
@login_required
@csrf.exempt
def get_distances():
    """Return the compatible distance list for the selected descriptor group."""
    data = request.get_json()
    descriptors = data.get('descriptors', [])

    if any(d in DL_DESCRIPTORS for d in descriptors):
        return jsonify(DL_DISTANCES)
    elif any(d in KEYPOINT_DESCRIPTORS for d in descriptors):
        return jsonify(KEYPOINT_DISTANCES)
    else:
        return jsonify(VECTOR_DISTANCES)


@app.route('/search', methods=['POST'])
@login_required
@csrf.exempt
def search():
    # get the uploaded file
    dataset_filename = request.form.get('dataset_filename')
    if 'query_image' in request.files and request.files['query_image'].filename != '':
        file = request.files['query_image']
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed."}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        query_url = url_for('serve_upload', filename=filename)
    elif dataset_filename:
        clean = dataset_filename.replace('\\', '/').lstrip('/')
        if '..' in clean.split('/'):
            return jsonify({"error": "Invalid file path."}), 400
        filepath = os.path.abspath(os.path.join(DATASET_PATH, clean))
        if not filepath.startswith(os.path.abspath(DATASET_PATH) + os.sep) or not os.path.isfile(filepath):
            return jsonify({"error": "Dataset image not found."}), 404
        query_url = url_for('serve_dataset_image', filename=clean)
    else:
        return jsonify({"error": "No image selected"}), 400

    # get search params
    descriptors = request.form.getlist('descriptors')
    distance = request.form.get('distance', 'Euclidienne')
    k = int(request.form.get('k', 10))
    manual_class = request.form.get('manual_class', '').strip() or None

    if not descriptors:
        return jsonify({"error": "Select at least one descriptor"}), 400

    result, error = run_search(filepath, descriptors, distance, k, manual_class=manual_class)
    if error:
        return jsonify({"error": error}), 400

    # add the query image url so we can show it
    result["query_image_url"] = url_for('serve_upload', filename=filename)

    return jsonify(result)



@app.route('/search_text', methods=['POST'])
@login_required
@csrf.exempt
def search_text():
    t0 = time.perf_counter()
    try:
        backbone_key = _resolve_backbone(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    k     = int(body.get("k") or 20)
    if not query:
        return jsonify({"error": "Empty query"}), 400

    embedder    = manager.get_embedder(backbone_key)
    image_index = manager.get_image_index(backbone_key)
    image_files = manager.get_image_files(backbone_key)

    qvec = embedder.encode_text(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(qvec)
    sims, idxs = image_index.search(qvec, min(k, len(image_files)))

    results = [
        {"rank": r + 1,
         "filename":   image_files[int(i)],
         "similarity": float(s),
         "image_url": url_for("serve_flickr8k_image", filename=image_files[int(i)])}
        for r, (i, s) in enumerate(zip(idxs[0], sims[0]))
    ]
    return jsonify({
        "results":          results,
        "total_results":    len(results),
        "elapsed":          round(time.perf_counter() - t0, 3),
        "backbone":         backbone_key,
        "backbone_display": BACKBONES[backbone_key].display_name,
    })

@app.route('/search_image_to_text', methods=['POST'])
@login_required
@csrf.exempt
def search_image_to_text():
    t0 = time.perf_counter()
    try:
        backbone_key = _resolve_backbone(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    k = int(request.form.get("k") or 20)
    dataset_filename = request.form.get('dataset_filename')
    file = request.files.get("query_image")
    if file is not None and file.filename != "":
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed."}), 400
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)
    elif dataset_filename:
        clean = secure_filename(dataset_filename)
        filepath = os.path.abspath(os.path.join(FLICKR8K_DATASET_PATH, clean))
        if not filepath.startswith(os.path.abspath(FLICKR8K_DATASET_PATH) + os.sep) or not os.path.isfile(filepath):
            return jsonify({"error": "Dataset image not found."}), 404
    else:
        return jsonify({"error": "No image uploaded"}), 400

    embedder      = manager.get_embedder(backbone_key)
    caption_index = manager.get_caption_index(backbone_key)
    captions      = manager.get_captions(backbone_key)

    pil  = Image.open(filepath).convert("RGB")
    qvec = embedder.encode_image(pil).astype("float32").reshape(1, -1)
    faiss.normalize_L2(qvec)
    sims, idxs = caption_index.search(qvec, min(k, len(captions)))

    caption_filenames = manager.get_caption_filenames(backbone_key)

    results = [
        {"rank": r + 1,
        "caption":      captions[int(i)],
        "similarity":   float(s),
        "source_image": caption_filenames[int(i)],
        "image_url":    url_for("serve_flickr8k_image", filename=caption_filenames[int(i)])}
        for r, (i, s) in enumerate(zip(idxs[0], sims[0]))
    ]
    return jsonify({
        "results":          results,
        "elapsed":          round(time.perf_counter() - t0, 3),
        "backbone":         backbone_key,
        "backbone_display": BACKBONES[backbone_key].display_name,
    })


@app.route("/backbones", methods=["GET"])
@login_required
@csrf.exempt
def list_backbones():
    return jsonify({
        "backbones":    manager.list_backbones(),
        "default":      DEFAULT_BACKBONE,
        "max_resident": int(os.environ.get("MIR_MAX_BACKBONES", "1")),
    })


def _resolve_backbone(req) -> str:
    key = None
    if req.is_json:
        key = (req.get_json(silent=True) or {}).get("backbone")
    key = key or req.form.get("backbone") or req.args.get("backbone") or DEFAULT_BACKBONE
    if not manager.is_known(key):
        raise ValueError(f"Unknown backbone '{key}'")
    return key


try:
    manager.preload(DEFAULT_BACKBONE)
    app.logger.info("Default backbone '%s' preloaded.", DEFAULT_BACKBONE)
except Exception as e:
    app.logger.warning("Default backbone preload failed: %s", e)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/")
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        expected_user = os.environ.get("MIR_USERNAME", "admin")
        expected_hash = os.environ.get("MIR_PASSWORD_HASH", "")
        if username == expected_user and check_password_hash(expected_hash, password):
            login_user(_User(username), remember=True)
            return redirect(request.args.get("next") or "/")
        error = "Invalid credentials — access denied."
    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

if __name__ == '__main__':
    app.run(port=5000, debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
