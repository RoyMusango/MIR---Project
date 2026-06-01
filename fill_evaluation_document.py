import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functions as f


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DESCRIPTOR_MAP = {
    "Color Histogram": f.generateHistogramme_Color,
    "HSV Histogram":   f.generateHistogramme_HSV,
    "GLCM":            f.generateGLCM,
    "LBP":             f.generateLBP,
    "HOG":             f.generateHOG,
    "SIFT":            f.generateSIFT,
    "ORB":             f.generateORB,
    "ViT":             f.generateViT,
    "ResNet":          f.generateResNet,
    "CLIP":            f.generateCLIP,
}

_KEYPOINT_DESCRIPTORS = {"SIFT", "ORB"}
_HIGHER_IS_BETTER = {"Correlation", "Intersection", "Brute force", "Flann", "Cosine Similarity"}

# Maps descriptor name → sub-folder inside descriptors/ and file suffix
# Must match the naming convention used by descriptors_making.ipynb / build_vit_resnet_descriptors.py
_DESCRIPTOR_FOLDERS = {
    "Color Histogram": "Hist_Col",
    "HSV Histogram":   "HSV",
    "GLCM":            "GLCM",
    "LBP":             "LBP",
    "HOG":             "HOG",
    "SIFT":            "SIFT",
    "ORB":             "ORB",
    "ViT":             "ViT",
    "ResNet":          "ResNet",
}

_DESCRIPTORS_ROOT = os.path.join(PROJECT_ROOT, "descriptors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_float(val, precision=4):
    return f"{val:.{precision}f}".replace('.', ',')


def get_image_class(filename):
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _load_descriptor(img_stem, desc_name):
    """
    Try to load a pre-computed descriptor from descriptors/<folder>/<stem>_<folder>.txt.
    Returns a numpy array on success, None if the file does not exist.
    """
    folder = _DESCRIPTOR_FOLDERS.get(desc_name)
    if folder is None:
        return None
    path = os.path.join(_DESCRIPTORS_ROOT, folder, f"{img_stem}_{folder}.txt")
    if not os.path.exists(path):
        return None
    return np.loadtxt(path)


def _effective_distance(desc_name, comparator):
    """Return (effective_comparator, higher_is_better) for a descriptor."""
    if desc_name in _KEYPOINT_DESCRIPTORS:
        eff = comparator if comparator in ("Brute force", "Flann") else "Brute force"
        return eff, True
    return comparator, comparator in _HIGHER_IS_BETTER


def _compute_single_distance(query_desc, db_desc, desc_name, comparator):
    """Compute one distance value, handling keypoint vs vector descriptors."""
    eff_dist, _ = _effective_distance(desc_name, comparator)
    is_binary = desc_name == "ORB"
    if desc_name in _KEYPOINT_DESCRIPTORS:
        q = np.float32(query_desc)
        d = np.float32(db_desc)
        if q.ndim == 1: q = q.reshape(1, -1)
        if d.ndim == 1: d = d.reshape(1, -1)
        try:
            return float(f.distance_f(q, d, eff_dist, is_binary=is_binary))
        except Exception:
            return 0.0
    else:
        try:
            return float(f.distance_f(query_desc, db_desc, eff_dist))
        except Exception:
            return 0.0


def _fuse_and_rank(raw_distances, descriptor_names, comparator):
    """
    Fuse per-descriptor raw distances and return a ranked list.

    For a single descriptor: sorts by raw score directly.
    For two descriptors: applies per-descriptor min-max normalisation,
    flips higher-is-better scores so that 0 always means "most similar",
    then averages the two normalised scores.

    Parameters
    ----------
    raw_distances : list of (img_name, [dist_desc0, dist_desc1, ...])
    descriptor_names : list of str
    comparator : str

    Returns
    -------
    list of (img_name, fused_score)   — sorted ascending (0 = most similar)
    """
    hib_list = [_effective_distance(d, comparator)[1] for d in descriptor_names]
    num_desc = len(descriptor_names)
    names = [name for name, _ in raw_distances]

    # Collect per-descriptor arrays for normalisation
    dist_matrix = np.array(
        [[dists[i] for i in range(num_desc)] for _, dists in raw_distances],
        dtype=float
    )  # shape (N, num_desc)

    normalised = np.zeros_like(dist_matrix)
    for i in range(num_desc):
        col = dist_matrix[:, i]
        min_v, max_v = col.min(), col.max()
        rng = max_v - min_v if max_v > min_v else 1e-10
        norm_col = (col - min_v) / rng
        if hib_list[i]:
            norm_col = 1.0 - norm_col
        normalised[:, i] = norm_col

    fused = normalised.mean(axis=1)
    return sorted(zip(names, fused.tolist()), key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fill_evaluation_document(
    dataset_test_path,
    descriptor_names,
    comparator,
    output_filename="evaluation_results.xls",
    db_dataset_path="dataset_voitures/dataset",
    k_values=[50, 100]
):
    """
    Generate an evaluation document for one or two combined descriptors.

    Parameters
    ----------
    dataset_test_path : str
        Path to the folder with 15 query images.
    descriptor_names : str or list of str
        One or two descriptor names.
        Accepted names: 'Color Histogram', 'HSV Histogram', 'GLCM', 'LBP',
        'HOG', 'SIFT', 'ORB', 'ViT', 'ResNet', 'CLIP'.
        A '+'-separated string is also accepted ('ViT+ResNet').
    comparator : str
        Distance/similarity measure.
    output_filename : str
        Output file name (.xls / .txt).
    db_dataset_path : str
        Path to the reference dataset (even-indexed images).
    k_values : list
        Cut-off values for Recall@k and Precision@k. Default [50, 100].

    Returns
    -------
    dict  Summary of results.
    """
    # --- Normalise descriptor_names to a list ---
    if isinstance(descriptor_names, str):
        descriptor_names = [d.strip() for d in descriptor_names.split("+")]
    descriptor_names = list(descriptor_names)

    if not (1 <= len(descriptor_names) <= 2):
        raise ValueError("1 or 2 descriptors must be provided.")
    for d in descriptor_names:
        if d not in _DESCRIPTOR_MAP:
            raise ValueError(f"Unknown descriptor: {d!r}. "
                             f"Available: {list(_DESCRIPTOR_MAP.keys())}")

    desc_funcs = [_DESCRIPTOR_MAP[d] for d in descriptor_names]
    combined_label = " + ".join(descriptor_names)

    # --- Test images ---
    test_images = sorted([
        fn for fn in os.listdir(dataset_test_path)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if len(test_images) != 15:
        print(f"WARN: {len(test_images)} query images found (expected 15).")

    # --- DB images (even first-digit only) ---
    db_images = sorted([
        fn for fn in os.listdir(db_dataset_path)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
        and fn[0].isdigit() and int(fn[0]) % 2 == 0
    ])
    if not db_images:
        raise ValueError(f"No images found in {db_dataset_path}")

    # --- Load DB descriptors (pre-computed files when available) ---
    has_precomputed = {
        d: os.path.isdir(os.path.join(_DESCRIPTORS_ROOT, _DESCRIPTOR_FOLDERS.get(d, "")))
        for d in descriptor_names
    }
    for d, avail in has_precomputed.items():
        src = "pre-computed files" if avail else "on-the-fly computation (slow!)"
        print(f"  [{d}] source: {src}")
    print(f"Loading DB descriptors  [{combined_label}]  ({len(db_images)} images)...")

    db_descriptors = []
    computed_count = 0

    for idx, img_name in enumerate(db_images):
        img_stem = img_name.rsplit('.', 1)[0]
        descs = []
        img = None

        for d_name, d_func in zip(descriptor_names, desc_funcs):
            loaded = _load_descriptor(img_stem, d_name)
            if loaded is not None:
                descs.append(loaded)
            else:
                if img is None:
                    img = cv2.imread(os.path.join(db_dataset_path, img_name))
                if img is None:
                    break
                try:
                    descs.append(d_func(img))
                    computed_count += 1
                except Exception as e:
                    print(f"  WARN: {img_name} [{d_name}]: {e}")
                    break

        if len(descs) == len(descriptor_names):
            db_descriptors.append((img_name, descs))

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(db_images)}")

    if computed_count:
        print(f"  NOTE: {computed_count} descriptors were computed from raw images "
              f"(pre-computed files missing).")
    print(f"OK {len(db_descriptors)} DB descriptors loaded.\n")

    # --- Evaluate queries ---
    result_rows = []
    indexing_times = []
    descriptor_sizes = []

    print(f"Evaluating {len(test_images)} queries...")
    for q_idx, query_name in enumerate(test_images):

        try:
            query_stem = query_name.rsplit('.', 1)[0]
            query_descs = []
            query_img = None
            for d_name, d_func in zip(descriptor_names, desc_funcs):
                loaded = _load_descriptor(query_stem, d_name)
                if loaded is not None:
                    query_descs.append(loaded)
                else:
                    if query_img is None:
                        query_img = cv2.imread(os.path.join(dataset_test_path, query_name))
                    if query_img is None:
                        raise RuntimeError(f"Cannot read {query_name}")
                    query_descs.append(d_func(query_img))

            t0 = time.time()
            raw = []
            for db_name, db_descs in db_descriptors:
                dists = [
                    _compute_single_distance(
                        query_descs[i], db_descs[i], descriptor_names[i], comparator
                    )
                    for i in range(len(descriptor_names))
                ]
                raw.append((db_name, dists))
            elapsed = time.time() - t0
            indexing_times.append(elapsed)

            total_bytes = sum(np.asarray(qd).nbytes for qd in query_descs)
            descriptor_sizes.append(total_bytes / (1024 * 1024))

            ranked = _fuse_and_rank(raw, descriptor_names, comparator)

            query_class = get_image_class(query_name)
            total_relevant = sum(
                1 for db_name, _ in db_descriptors
                if get_image_class(db_name) == query_class
            )

            row = {"label": f"R{q_idx + 1}", "indexing_time": elapsed}

            for k in k_values:
                top_k = ranked[:k]
                hits = sum(1 for n, _ in top_k if get_image_class(n) == query_class)
                row[f"recall_{k}"]    = hits / total_relevant if total_relevant else 0.0
                row[f"precision_{k}"] = hits / k
                h, ap_sum = 0, 0.0
                for rank, (n, _) in enumerate(top_k, 1):
                    if get_image_class(n) == query_class:
                        h += 1
                        ap_sum += h / rank
                row[f"ap_{k}"] = ap_sum / total_relevant if total_relevant else 0.0

            k_max = max(total_relevant, 1)
            top_max = ranked[:k_max]
            hits_max = sum(1 for n, _ in top_max if get_image_class(n) == query_class)
            row["recall_max"]    = hits_max / total_relevant if total_relevant else 0.0
            row["precision_max"] = hits_max / k_max
            h_m, ap_m = 0, 0.0
            for rank, (n, _) in enumerate(top_max, 1):
                if get_image_class(n) == query_class:
                    h_m += 1
                    ap_m += h_m / rank
            row["ap_max"] = ap_m / total_relevant if total_relevant else 0.0
            row["k_max"] = k_max

            result_rows.append(row)
            print(f"  OK  {row['label']:>3}  {query_name.rsplit('.', 1)[0]}  (Top Max k={k_max})")

        except Exception as e:
            print(f"  ERROR {query_name}: {e}")
            import traceback; traceback.print_exc()

    # --- Write output file ---
    print("\nWriting output file...")
    lines = []

    lines.append(f"Descriptor(s):\t{combined_label}\tComparator:\t{comparator}")
    lines.append("")

    lines.append(
        "Query\tR (Recall)\t\t\tP (Precision)\t\t\tAP\t\t\t"
        "Indexing Time\tDescriptor Size\tAvg Search Time/image"
    )
    lines.append(
        "Index\tTop 50\tTop 100\tTop Max\tTop 50\tTop 100\tTop Max\t"
        "Top 50\tTop 100\tTop Max"
    )

    for i, row in enumerate(result_rows):
        idx_time  = row["indexing_time"]
        desc_size = descriptor_sizes[i]
        avg_time  = idx_time / len(db_descriptors) if db_descriptors else 0.0
        lines.append(
            f"{row['label']}\t"
            f"{format_float(row['recall_50'], 4)}\t"
            f"{format_float(row['recall_100'], 4)}\t"
            f"{format_float(row['recall_max'], 4)}\t"
            f"{format_float(row['precision_50'], 4)}\t"
            f"{format_float(row['precision_100'], 4)}\t"
            f"{format_float(row['precision_max'], 4)}\t"
            f"{format_float(row['ap_50'], 4)}\t"
            f"{format_float(row['ap_100'], 4)}\t"
            f"{format_float(row['ap_max'], 4)}\t"
            f"{format_float(idx_time, 6)}\t"
            f"{format_float(desc_size, 6)}\t"
            f"{format_float(avg_time, 9)}"
        )

    lines.append("")

    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

    map_50_ap    = _mean([r["ap_50"]         for r in result_rows])
    map_100_ap   = _mean([r["ap_100"]        for r in result_rows])
    map_max_ap   = _mean([r["ap_max"]        for r in result_rows])
    map_50_rec   = _mean([r["recall_50"]     for r in result_rows])
    map_100_rec  = _mean([r["recall_100"]    for r in result_rows])
    map_max_rec  = _mean([r["recall_max"]    for r in result_rows])
    map_50_prec  = _mean([r["precision_50"]  for r in result_rows])
    map_100_prec = _mean([r["precision_100"] for r in result_rows])
    map_max_prec = _mean([r["precision_max"] for r in result_rows])

    lines.append("\t\tmAP")
    lines.append(f"\t\tTop 50\t{format_float(map_50_ap, 4)}")
    lines.append(f"\t\tTop 100\t{format_float(map_100_ap, 4)}")
    lines.append(f"\t\tTop Max\t{format_float(map_max_ap, 4)}")
    lines.append(f"\tRecall Top 50\t{format_float(map_50_rec, 4)}")
    lines.append(f"\tRecall Top 100\t{format_float(map_100_rec, 4)}")
    lines.append(f"\tRecall Top Max\t{format_float(map_max_rec, 4)}")
    lines.append(f"\tPrecision Top 50\t{format_float(map_50_prec, 4)}")
    lines.append(f"\tPrecision Top 100\t{format_float(map_100_prec, 4)}")
    lines.append(f"\tPrecision Top Max\t{format_float(map_max_prec, 4)}")

    output_path = os.path.join(PROJECT_ROOT, output_filename)
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(lines))
    print(f"OK Written: {output_path}")

    return {
        "output_file":        output_path,
        "descriptors":        combined_label,
        "comparator":         comparator,
        "num_queries":        len(result_rows),
        "num_db_images":      len(db_descriptors),
        "map_50_ap":          map_50_ap,
        "map_100_ap":         map_100_ap,
        "map_max_ap":         map_max_ap,
        "avg_indexing_time":  _mean(indexing_times),
    }


# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_test = "dataset_test"
    db_path = "dataset_voitures/dataset"

    # Two descriptors combined
    print("\n" + "=" * 60)
    print("Example — combined descriptors: ViT + ResNet + Cosine Similarity")
    print("=" * 60)
    result = fill_evaluation_document(
        dataset_test_path=dataset_test,
        descriptor_names=["ViT", "ResNet"],
        comparator="Cosine Similarity",
        output_filename="eval_ViT_ResNet_Cosine Similarity.xls",
        db_dataset_path=db_path,
    )
    for k, v in result.items():
        print(f"  {k}: {format_float(v, 6) if isinstance(v, float) else v}")
