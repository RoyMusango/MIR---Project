import os
import sys
import time
import numpy as np
import csv

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functions as f

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")

# ── 15 mandatory queries for even-numbered groups ────────────────────────────
QUERIES = [
    ("1_4_Kia_stinger_1990.jpg",       "1_4"),
    ("1_2_Kia_sorento_1675.jpg",        "1_2"),
    ("1_9_Kia_stonic_2677.jpg",         "1_9"),
    ("3_1_Renault_Twingo_4487.jpg",     "3_1"),
    ("3_0_Renault_grandscenic_4372.jpg","3_0"),
    ("3_5_Renault_clio_5101.jpg",       "3_5"),
    ("5_0_Mercedes_ClasseCLS_7059.jpg", "5_0"),
    ("5_4_Mercedes_GLEcoupe_7428.jpg",  "5_4"),
    ("5_8_Mercedes_CLA_7992.jpg",       "5_8"),
    ("7_0_Peugeot_508break_9591.jpg",   "7_0"),
    ("7_3_Peugeot_Rifter_10091.jpg",    "7_3"),
    ("7_6_Peugeot_3008_10530.jpg",      "7_6"),
    ("9_0_Audi_A6_12268.jpg",           "9_0"),
    ("9_3_Audi_Q7_12722.jpg",           "9_3"),
    ("9_4_Audi_A1_12910.jpg",           "9_4"),
]

# ── Top-3 descriptors to benchmark (must include 1 CNN + 1 ViT) ─────────────
TOP_3_DESCRIPTORS = [
    ("ResNet50", "ResNet50", f.generateResNet50, "Euclidienne"),
    ("CLIP_ViT",  "CLIP",    f.generateCLIP,     "Cosine Similarity"),
    ("HOG",       "HOG",     f.generateHOG,      "Euclidienne"),
]

def get_image_class(filename):
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]

def compute_all_metrics(ranked_filenames, query_class, total_relevant, k):
    top_k = ranked_filenames[:k]
    relevant_found = sum(1 for f in top_k if get_image_class(f) == query_class)
    precision = relevant_found / k if k > 0 else 0.0
    recall = relevant_found / total_relevant if total_relevant > 0 else 0.0

    # AP
    hits, ap = 0, 0.0
    for i, fname in enumerate(top_k, 1):
        if get_image_class(fname) == query_class:
            hits += 1
            ap += hits / i
    ap = ap / total_relevant if total_relevant > 0 else 0.0

    # R-Precision
    r = min(total_relevant, len(ranked_filenames))
    r_hits = sum(1 for fname in ranked_filenames[:r] if get_image_class(fname) == query_class)
    r_prec = r_hits / r if r > 0 else 0.0

    return precision, recall, ap, r_prec

def load_descriptors(desc_folder, desc_name):
    desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)
    files = [ff for ff in os.listdir(desc_dir)
             if ff.endswith(".txt") and ff[0].isdigit() and int(ff[0]) % 2 == 0]
    data = []
    for fname in files:
        fpath = os.path.join(desc_dir, fname)
        feat = np.loadtxt(fpath)
        img_file = fname.replace(f"_{desc_folder}.txt", ".jpg")
        data.append((img_file, feat))
    return data

def run_query(query_img_path, query_class, desc_data, distance_name):
    import cv2
    img = cv2.imread(query_img_path)
    if img is None:
        print(f"Could not read {query_img_path}")
        return [], 0

    # pick generate function from TOP_3
    desc_func = None
    for name, folder, func, dist in TOP_3_DESCRIPTORS:
        if dist == distance_name:
            desc_func = func
            break

    # get query features
    query_feat = None
    for name, folder, func, dist in TOP_3_DESCRIPTORS:
        if distance_name == dist:
            query_feat = func(img)
            break

    total_relevant = sum(1 for img_f, _ in desc_data if get_image_class(img_f) == query_class)

    results = []
    for img_file, db_feat in desc_data:
        try:
            dist = f.distance_f(query_feat, db_feat, distance_name)
            results.append((img_file, dist))
        except Exception as e:
            continue

    higher = distance_name in {"Correlation", "Cosine Similarity", "Intersection", "Brute force", "Flann"}
    results.sort(key=lambda x: x[1], reverse=higher)
    return [r[0] for r in results], total_relevant


def benchmark_indexing():
    """Table 3: indexing time, descriptor size, avg search time."""
    import cv2
    rows = []
    sample_img_path = os.path.join(DATASET_PATH, QUERIES[0][0])
    sample_img = cv2.imread(sample_img_path)

    for desc_name, desc_folder, desc_func, distance_name in TOP_3_DESCRIPTORS:
        desc_dir = os.path.join(DESCRIPTORS_PATH, desc_folder)

        # indexing time
        start = time.time()
        desc_func(sample_img)
        index_time = round(time.time() - start, 4)

        # descriptor folder size in MB
        total_size = 0
        if os.path.exists(desc_dir):
            for ff in os.listdir(desc_dir):
                total_size += os.path.getsize(os.path.join(desc_dir, ff))
        size_mb = round(total_size / (1024 * 1024), 2)

        # avg search time (load + compare one image)
        desc_data = load_descriptors(desc_folder, desc_name)
        start = time.time()
        for img_file, db_feat in desc_data[:50]:
            f.distance_f(desc_func(sample_img), db_feat, distance_name)
        avg_search = round((time.time() - start) / 50, 4)

        rows.append([desc_name, index_time, size_mb, avg_search])
        print(f"[Table 3] {desc_name}: index={index_time}s, size={size_mb}MB, search={avg_search}s")

    os.makedirs("results", exist_ok=True)
    with open("results/table3_indexing_performance.csv", "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Descriptor", "Indexing Time (s)", "Size (MB)", "Avg Search Time (s)"])
        writer.writerows(rows)
    print("Saved: results/table3_indexing_performance.csv")


def benchmark_queries():
    """Table 4: R, P, AP, mAP for all 15 queries x top-3 descriptors."""
    os.makedirs("results", exist_ok=True)
    rows = []

    for desc_name, desc_folder, desc_func, distance_name in TOP_3_DESCRIPTORS:
        print(f"\n[Table 4] Running queries for descriptor: {desc_name}")
        desc_data = load_descriptors(desc_folder, desc_name)
        ap_list_50, ap_list_100 = [], []

        for q_idx, (q_file, q_class) in enumerate(QUERIES):
            q_path = os.path.join(DATASET_PATH, q_file)
            ranked, total_rel = run_query(q_path, q_class, desc_data, distance_name)

            p50,  r50,  ap50,  rp50  = compute_all_metrics(ranked, q_class, total_rel, 50)
            p100, r100, ap100, rp100 = compute_all_metrics(ranked, q_class, total_rel, 100)

            ap_list_50.append(ap50)
            ap_list_100.append(ap100)

            rows.append([
                f"R{q_idx+1}", q_file, desc_name,
                round(r50*100,1),  round(p50*100,1),  round(ap50*100,1),  round(rp50*100,1),
                round(r100*100,1), round(p100*100,1), round(ap100*100,1), round(rp100*100,1),
            ])
            print(f"  R{q_idx+1} {q_file}: P@50={p50:.2f} R@50={r50:.2f} AP@50={ap50:.2f}")

        map50  = round(np.mean(ap_list_50)*100, 1)
        map100 = round(np.mean(ap_list_100)*100, 1)
        print(f"  mAP@50={map50}%  mAP@100={map100}%")

    with open("results/table4_query_metrics.csv", "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "Query", "Image", "Descriptor",
            "R@50", "P@50", "AP@50", "R-Prec@50",
            "R@100", "P@100", "AP@100", "R-Prec@100"
        ])
        writer.writerows(rows)
    print("\nSaved: results/table4_query_metrics.csv")


if __name__ == "__main__":
    print("=" * 50)
    print("TABLE 3 — Indexing Performance")
    print("=" * 50)
    benchmark_indexing()

    print("\n" + "=" * 50)
    print("TABLE 4 — Query Metrics")
    print("=" * 50)
    benchmark_queries()