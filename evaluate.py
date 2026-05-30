"""
evaluate.py — Batch evaluation for I-ILIA-014 MIR Project
Group 7 (odd-numbered) → Table 1 queries (classes 0, 2, 4, 6, 8)
Database: even-prefix images (filter % 2 == 0)

Generates:
  - results/table3_indexing_performance.csv  (indexing time, descriptor size, search time)
  - results/table4_query_metrics.csv         (R, P, AP, R-Prec @50/@100 + mAP)
"""

import os
import sys
import time
import csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functions as f

DATASET_PATH     = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")

# ── 15 mandatory queries — TABLE 1 (Group 7 = odd-numbered) ────────────────
# Classes: 0=BMW, 2=Volkswagen, 4=Opel, 6=Hyundai, 8=Ford
QUERIES = [
    ("0_1_BMW_X3_207.jpg",                "0"),
    ("0_0_BMW_Serie3Berline_74.jpg",      "0"),
    ("0_2_BMW_i8_299.jpg",                "0"),

    ("2_0_Volkswagen_Touareg_2822.jpg",   "2"),
    ("2_4_Volkswagen_Polo_3463.jpg",      "2"),
    ("2_9_Volkswagen_T-Roc_4209.jpg",     "2"),

    ("4_2_Opel_vivarofourgon_5999.jpg",   "4"),
    ("4_4_Opel_Insignatourer_6353.jpg",   "4"),
    ("4_9_Opel_zafiralife_6887.jpg",      "4"),

    ("6_0_Hyundai_Nexo_8282.jpg",         "6"),
    ("6_3_Hyundai_i10_8837.jpg",          "6"),
    ("6_5_Hyundai_i30_9125.jpg",          "6"),

    ("8_1_Ford_Puma_11276.jpg",           "8"),
    ("8_5_Ford_Explorer_11897.jpg",       "8"),
    ("8_6_Ford_Focus_11951.jpg",          "8"),
]

# ── Top-3 descriptors (must include ≥1 CNN + ≥1 ViT per énoncé Note 1) ────
# Tuple: (display_name, folder_name, descriptor_fn, default_distance)
TOP_3_DESCRIPTORS = [
    ("ResNet50", "ResNet50", f.generateResNet50, "Euclidienne"),
    ("CLIP_ViT", "CLIP",     f.generateCLIP,     "Cosine Similarity"),
    ("HOG",      "HOG",      f.generateHOG,      "Euclidienne"),
]


def get_image_class(filename):
    """First digit of the filename is the class (brand).
    e.g. '0_1_BMW_X3_207.jpg' -> '0'  (BMW)
    The second digit is a model sub-index — NOT part of the class.
    """
    return os.path.basename(filename).split("_")[0]


def compute_all_metrics(ranked_filenames, query_class, total_relevant, k):
    """Returns (precision, recall, ap, r_precision) at cutoff k."""
    top_k = ranked_filenames[:k]
    relevant_found = sum(1 for fn in top_k if get_image_class(fn) == query_class)

    precision = relevant_found / k if k > 0 else 0.0
    recall    = relevant_found / total_relevant if total_relevant > 0 else 0.0

    # Average Precision calculation
    hits, sum_precisions = 0, 0.0
    for i, fn in enumerate(top_k, 1):
        if get_image_class(fn) == query_class:
            hits += 1
            sum_precisions += hits / i
            
    # Standard global AP — academically correct denominator (all relevant in corpus)
    ap = sum_precisions / total_relevant if total_relevant > 0 else 0.0

    # AP@K — valid IR metric; denominator capped at k, yields higher values
    ap_at_k = sum_precisions / min(total_relevant, k) if total_relevant > 0 else 0.0
    

    # R-Precision (precision at rank R, where R = total relevant)
    r = min(total_relevant, len(ranked_filenames))
    r_hits = sum(1 for fn in ranked_filenames[:r] if get_image_class(fn) == query_class)
    r_prec = r_hits / r if r > 0 else 0.0

    return precision, recall, ap, ap_at_k, r_prec


def load_descriptors(desc_folder, desc_name):
    """Load all even-prefix descriptor files (matches the indexed DB)."""
    desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)
    if not os.path.isdir(desc_dir):
        print(f"[WARN] Descriptor folder not found: {desc_dir}")
        return []

    files = [
        ff for ff in os.listdir(desc_dir)
        if ff.endswith(".txt")
        and ff[0].isdigit()
        and int(ff[0]) % 2 == 0    # even-prefix → classes 0, 2, 4, 6, 8 (Table 1)
    ]

    data = []
    for fname in files:
        fpath = os.path.join(desc_dir, fname)
        try:
            feat = np.loadtxt(fpath)
            img_file = fname.replace(f"_{desc_folder}.txt", ".jpg")
            data.append((img_file, feat))
        except Exception as e:
            print(f"[WARN] could not load {fname}: {e}")
    return data


def run_query(query_img_path, query_class, desc_data, desc_func, distance_name):
    """Run one query: extract feature, compute distance to every DB entry, return ranked filenames."""
    img = cv2.imread(query_img_path)
    if img is None:
        print(f"[ERROR] could not read {query_img_path}")
        return [], 0

    query_feat = desc_func(img)
    total_relevant = sum(1 for img_f, _ in desc_data if get_image_class(img_f) == query_class)

    results = []
    for img_file, db_feat in desc_data:
        try:
            dist = f.distance_f(query_feat, db_feat, distance_name)
            results.append((img_file, dist))
        except Exception:
            continue

    # similarity metrics sort descending; distance metrics sort ascending
    higher_is_better = distance_name in {
        "Correlation", "Cosine Similarity", "Intersection", "Brute force", "Flann"
    }
    results.sort(key=lambda x: x[1], reverse=higher_is_better)

    return [r[0] for r in results], total_relevant


def benchmark_indexing():
    """Table 3: indexing time, descriptor size, avg search time."""
    rows = []
    sample_img_path = os.path.join(DATASET_PATH, QUERIES[0][0])
    sample_img = cv2.imread(sample_img_path)

    if sample_img is None:
        print(f"[FATAL] sample image not found: {sample_img_path}")
        print("Check your dataset path and filenames.")
        return

    for desc_name, desc_folder, desc_func, distance_name in TOP_3_DESCRIPTORS:
        desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)

        # indexing time = 1 image feature extraction
        start = time.time()
        desc_func(sample_img)
        index_time = round(time.time() - start, 4)

        # descriptor folder size in MB
        total_size = 0
        if os.path.exists(desc_dir):
            for ff in os.listdir(desc_dir):
                total_size += os.path.getsize(os.path.join(desc_dir, ff))
        size_mb = round(total_size / (1024 * 1024), 2)

        # avg search time over 50 comparisons
        desc_data = load_descriptors(desc_folder, desc_name)
        if not desc_data:
            print(f"[WARN] no descriptors loaded for {desc_name} — skipping search timing")
            avg_search = 0.0
        else:
            sample_feat = desc_func(sample_img)
            n = min(50, len(desc_data))
            start = time.time()
            for img_file, db_feat in desc_data[:n]:
                f.distance_f(sample_feat, db_feat, distance_name)
            avg_search = round((time.time() - start) / n, 4)

        rows.append([desc_name, index_time, size_mb, avg_search])
        print(f"[Table 3] {desc_name}: index={index_time}s · size={size_mb}MB · search={avg_search}s")

    os.makedirs("results", exist_ok=True)
    with open("results/table3_indexing_performance.csv", "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Descriptor", "Indexing Time (s)", "Size (MB)", "Avg Search Time (s)"])
        writer.writerows(rows)
    print("Saved: results/table3_indexing_performance.csv")


def benchmark_queries():
    """Table 4: R, P, AP, R-Prec @50 and @100 + mAP for all 15 queries × top-3 descriptors."""
    os.makedirs("results", exist_ok=True)
    rows = []
    map_summary = []

    for desc_name, desc_folder, desc_func, distance_name in TOP_3_DESCRIPTORS:
        print(f"\n[Table 4] Running queries for descriptor: {desc_name}")
        desc_data = load_descriptors(desc_folder, desc_name)
        if not desc_data:
            print(f"[ERROR] no descriptors loaded for {desc_name} — skipping")
            continue

        ap_list_50, ap_list_100 = [], []
        ap_k_list_50, ap_k_list_100 = [], []

        for q_idx, (q_file, q_class) in enumerate(QUERIES, 1):
            q_path = os.path.join(DATASET_PATH, q_file)
            ranked, total_rel = run_query(q_path, q_class, desc_data, desc_func, distance_name)

            p50,  r50,  ap50,  ap_k50,  rp50  = compute_all_metrics(ranked, q_class, total_rel, 50)
            p100, r100, ap100, ap_k100, rp100 = compute_all_metrics(ranked, q_class, total_rel, 100)

            

            # Append metrics to lists
            ap_list_50.append(ap50)
            ap_list_100.append(ap100)
            ap_k_list_50.append(ap_k50)
            ap_k_list_100.append(ap_k100)
            

            rows.append([
                f"R{q_idx}", q_file, desc_name,
                round(r50   * 100, 1), round(p50   * 100, 1),
                round(ap50  * 100, 1), round(ap_k50  * 100, 1), round(rp50  * 100, 1),
                round(r100  * 100, 1), round(p100  * 100, 1),
                round(ap100 * 100, 1), round(ap_k100 * 100, 1), round(rp100 * 100, 1),
            ])
            print(f"  R{q_idx:>2} {q_file[:42]:42} P@50={p50*100:5.1f}% R@50={r50*100:5.1f}% AP@50={ap50*100:5.1f}%")

        map50      = round(np.mean(ap_list_50)    * 100, 1)
        map100     = round(np.mean(ap_list_100)   * 100, 1)
        map_k50    = round(np.mean(ap_k_list_50)  * 100, 1)
        map_k100   = round(np.mean(ap_k_list_100) * 100, 1)
        map_summary.append((desc_name, map50, map100, map_k50, map_k100))
        print(f"  → mAP@50={map50}%  mAP@K50={map_k50}%  |  mAP@100={map100}%  mAP@K100={map_k100}%")

    with open("results/table4_query_metrics.csv", "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "Query", "Image", "Descriptor",
            "R@50", "P@50", "AP@50", "AP@K50", "R-Prec@50",
            "R@100", "P@100", "AP@100", "AP@K100", "R-Prec@100",
        ])
        writer.writerows(rows)
    print("\nSaved: results/table4_query_metrics.csv")

    print("\n" + "=" * 60)
    print("FINAL mAP SUMMARY (Group 7 → Table 1 queries)")
    print("=" * 60)
    print(f"{'Descriptor':<14} {'mAP@50':>10} {'mAP@100':>10} {'mAP@K50':>10} {'mAP@K100':>10}")
    print("-" * 50)
    for name, m50, m100, mk50, mk100 in map_summary:
        print(f"{name:<14} {m50:>9.1f}% {m100:>9.1f}% {mk50:>9.1f}% {mk100:>9.1f}%")
    
    # ── Unimodal rankings dump — consumed by plot_pr_curves.py ──────────
    import json as _json
    _dump = {}
    for desc_name, desc_folder, desc_func, distance_name in TOP_3_DESCRIPTORS:
        _dump[desc_name] = {"queries": []}
        desc_data = load_descriptors(desc_folder, desc_name)
        for q_file, q_class in QUERIES:
            q_path = os.path.join(DATASET_PATH, q_file)
            ranked, total_rel = run_query(q_path, q_class, desc_data, desc_func, distance_name)
            _dump[desc_name]["queries"].append({
                "query_id":   q_file,
                "n_relevant": total_rel,
                "ranked": [
                    {"rank": r + 1,
                    "filename": fn,
                    "relevant": get_image_class(fn) == q_class}
                    for r, fn in enumerate(ranked)
                ]
            })
    os.makedirs("results", exist_ok=True)
    _json_path = os.path.join("results", "unimodal_rankings.json")
    with open(_json_path, "w", encoding="utf-8") as _f:
        _json.dump(_dump, _f)
    print(f"Saved: {_json_path}")


if __name__ == "__main__":
    import cv2  # delayed so the script can show a help banner without the import
    print("=" * 60)
    print("TABLE 3 — Indexing Performance")
    print("=" * 60)
    benchmark_indexing()

    print("\n" + "=" * 60)
    print("TABLE 4 — Query Metrics")
    print("=" * 60)
    benchmark_queries()