"""
compare_backbones.py
─────────────────────────────────────────────────────────────────────
Group 07 — I-ILIA-014 MIR Project
Phase 1: Comparative benchmarking of all registered VLM backbones.

Reuses the canonical query pairs from ``evaluate_multimodal.py`` so the
numbers in Table 5 are directly comparable to the existing multimodal
evaluation. For each backbone we measure:

    • Retrieval quality  →  Recall@1 / @5 / @10  and  MRR
                            for both Text→Image and Image→Text modes.
    • Engineering cost   →  Per-query encode latency (ms)
                            Per-query FAISS search latency (ms)
                            One-time model load time (s)
                            Resident memory delta (MB)

Outputs (under ``results/``):

    table5_backbone_comparison.csv   - machine-readable table.
    backbone_rankings.json           - per-query ranked lists; consumed
                                        by ``plot_pr_curves.py``.
    failure_modes.md                 - qualitative diff: which queries
                                        succeed on backbone A but fail
                                        on backbone B.

Run:
    python compare_backbones.py
    python compare_backbones.py --backbones clip openclip_l14
    python compare_backbones.py --k 20 --skip-i2t

Assumptions documented in MIR_Project___Final_State.md:
    • Factory signature   :  ``get_embedder(key) -> BaseEmbedder``
    • Embedder interface  :  ``encode_image, encode_text,
                              encode_images_batch, encode_texts_batch``
    • H5 file layout      :  ``descriptors/CLIP_Flickr8k/
                              embeddings_<backbone_id>.h5``
                              ``descriptors/CLIP_Flickr8k/
                              captions_<backbone_id>.h5``
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

# FAISS is the same library used by app.py — keep behaviour identical.
import faiss  # type: ignore

# ─────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s · %(levelname)-7s · %(message)s",
    datefmt="%H:%M:%S",
    level=os.environ.get("MIR_LOG_LEVEL", "INFO").upper(),
)
log = logging.getLogger("compare_backbones")


# ─────────────────────────────────────────────────────────────────────
#  Project-wide constants — single source of truth for backbone IDs.
#  Must agree with embedders/__init__.py.
# ─────────────────────────────────────────────────────────────────────
BACKBONES: dict[str, dict[str, str]] = {
    "clip":         {"backbone_id": "clip_openai_b32",
                     "display":     "CLIP ViT-B/32 (OpenAI)"},
    "openclip_b32": {"backbone_id": "openclip_laion2b_vitb32",
                     "display":     "OpenCLIP ViT-B/32 (LAION-2B)"},
    "openclip_l14": {"backbone_id": "openclip_laion2b_vitl14",
                     "display":     "OpenCLIP ViT-L/14 (LAION-2B)"},
    "blip":         {"backbone_id": "blip_itm_base",
                     "display":     "BLIP ITM-base (COCO)"},
}

H5_DIR        = Path("descriptors/CLIP_Flickr8k")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
#  Test pairs — import the canonical set; fall back to a documented
#  default so the script never silently uses something different.
# ─────────────────────────────────────────────────────────────────────
def load_test_pairs() -> list[tuple[str, str]]:
    try:
        from evaluate_multimodal import test_pairs
        result = []
        for p in test_pairs:
            if isinstance(p, dict):
                result.append((p["text"], p["image"]))
            else:
                result.append(tuple(p))
        log.info("Loaded %d test pairs from evaluate_multimodal.py", len(result))
        return result
    except Exception as e:
        log.warning(
            "Could not import test_pairs from evaluate_multimodal (%s). "
            "Falling back to documented defaults — verify these match your report!",
            e,
        )
    fallback = [
        ("a dog running on the beach",           "2090545563_a4e66ec76b.jpg"),
        ("a man riding a bicycle in the snow",   "2532302501_44b96e1ad6.jpg"),
        ("two children playing with a ball",     "3637013_c675de7705.jpg"),
    ]
    return fallback


# ─────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────
@dataclass
class BackboneReport:
    key:                  str
    backbone_id:          str
    display_name:         str
    n_queries:            int

    # Retrieval quality — Text → Image
    t2i_recall_at_1:      float = 0.0
    t2i_recall_at_5:      float = 0.0
    t2i_recall_at_10:     float = 0.0
    t2i_mrr:              float = 0.0

    # Retrieval quality — Image → Text
    i2t_recall_at_1:      float = 0.0
    i2t_recall_at_5:      float = 0.0
    i2t_recall_at_10:     float = 0.0
    i2t_mrr:              float = 0.0

    # Engineering cost
    model_load_seconds:   float = 0.0
    memory_mb_delta:      float = 0.0
    encode_ms_mean:       float = 0.0
    faiss_ms_mean:        float = 0.0

    # Index sizes (sanity-check rows + dimensionality)
    n_images_indexed:     int   = 0
    n_captions_indexed:   int   = 0
    embedding_dim:        int   = 0

    # Per-query rankings — saved separately to JSON, not inline in CSV.
    rankings: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
#  FAISS helpers — match the convention used by app.py:
#  L2-normalize embeddings and use IndexFlatIP so inner product == cosine.
# ─────────────────────────────────────────────────────────────────────
def _build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build a cosine-similarity index from a stack of vectors."""
    vectors = vectors.astype(np.float32, copy=False)
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx


def _query_index(index: faiss.IndexFlatIP, query_vec: np.ndarray, k: int):
    """Single-query search returning (similarities, indices)."""
    query_vec = query_vec.astype(np.float32, copy=False).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    sims, idxs = index.search(query_vec, k)
    return sims[0], idxs[0]


# ─────────────────────────────────────────────────────────────────────
#  H5 loading — defensive: try several common key names.
# ─────────────────────────────────────────────────────────────────────
def _load_h5(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Return (embeddings, identifiers).
    ``identifiers`` are image filenames for image H5, or
    caption strings for caption H5. The script tries the obvious key
    names produced by build_flickr8k_*_db.py.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        # embeddings: ``embeddings`` is the conventional name;
        # accept ``features`` as a fallback.
        emb_key = next((k for k in ("embeddings", "features", "vectors") if k in keys), None)
        if emb_key is None:
            raise KeyError(f"No embeddings dataset in {path} (keys={keys})")
        embeddings = f[emb_key][:]

        # identifiers: ``filenames`` for images, ``captions`` for captions
        id_key = next((k for k in ("captions", "filenames", "ids", "labels") if k in keys), None)
        if id_key is None:
            raise KeyError(f"No identifier dataset in {path} (keys={keys})")
        raw = f[id_key][:]
        identifiers = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]

    return embeddings, identifiers


# ─────────────────────────────────────────────────────────────────────
#  Metric primitives
# ─────────────────────────────────────────────────────────────────────
def rank_of(target_index: int, retrieved_indices: Iterable[int]) -> int | None:
    """Return 1-based rank, or None if not retrieved."""
    for r, idx in enumerate(retrieved_indices, start=1):
        if int(idx) == int(target_index):
            return r
    return None


def recall_at(rank: int | None, k: int) -> int:
    return 1 if (rank is not None and rank <= k) else 0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


# ─────────────────────────────────────────────────────────────────────
#  Core benchmark loop for a single backbone
# ─────────────────────────────────────────────────────────────────────
def benchmark_backbone(
    key: str,
    test_pairs: list[tuple[str, str]],
    k_max: int,
    skip_t2i: bool,
    skip_i2t: bool,
) -> BackboneReport | None:
    meta = BACKBONES[key]
    backbone_id = meta["backbone_id"]
    display = meta["display"]

    img_h5  = H5_DIR / f"embeddings_{backbone_id}.h5"
    capt_h5 = H5_DIR / f"captions_{backbone_id}.h5"

    if not img_h5.exists() or not capt_h5.exists():
        log.warning(
            "Skipping %s — missing H5 (image=%s, caption=%s).",
            key, img_h5.exists(), capt_h5.exists(),
        )
        return None

    log.info("══════════ %s ══════════", display)
    log.info("Image  H5  : %s", img_h5)
    log.info("Caption H5 : %s", capt_h5)

    # ── Load embeddings & build FAISS indices ──
    img_embeds,  img_files    = _load_h5(img_h5)
    capt_embeds, capt_texts   = _load_h5(capt_h5)
    log.info("Loaded %d image embeddings (dim=%d), %d caption embeddings.",
             len(img_files), img_embeds.shape[1], len(capt_texts))

    img_index  = _build_faiss_index(img_embeds.copy())
    capt_index = _build_faiss_index(capt_embeds.copy())

    # Filename → row index lookup (for ground-truth resolution)
    file_to_idx = {fn: i for i, fn in enumerate(img_files)}

    # ── Load the embedder (this is where the heavy weights come in) ──
    from embedders import get_embedder  # local import — avoids loading torch
                                         # if a backbone is missing its H5.

    tracemalloc.start()
    t0 = time.perf_counter()
    embedder = get_embedder(key)
    load_time = time.perf_counter() - t0
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log.info("Model loaded in %.2fs · ~%.1f MB resident delta", load_time, current / 1e6)

    report = BackboneReport(
        key=key,
        backbone_id=backbone_id,
        display_name=display,
        n_queries=len(test_pairs),
        model_load_seconds=round(load_time, 3),
        memory_mb_delta=round(current / 1e6, 1),
        n_images_indexed=len(img_files),
        n_captions_indexed=len(capt_texts),
        embedding_dim=int(img_embeds.shape[1]),
    )

    encode_times: list[float] = []
    faiss_times:  list[float] = []

    # ─────────────────────────────────────────────────────────────
    #  Text → Image
    # ─────────────────────────────────────────────────────────────
    if not skip_t2i:
        t2i_ranks: list[int | None] = []
        for query_text, gt_img in test_pairs:
            if gt_img not in file_to_idx:
                log.warning("  GT image '%s' not in index — skipping query '%s'", gt_img, query_text)
                t2i_ranks.append(None)
                report.rankings.append({
                    "mode": "t2i", "query": query_text, "ground_truth": gt_img,
                    "found": False, "top_k": [], "rank": None,
                })
                continue

            t1 = time.perf_counter()
            qvec = embedder.encode_text(query_text)
            encode_times.append((time.perf_counter() - t1) * 1000)

            t2 = time.perf_counter()
            sims, idxs = _query_index(img_index, np.asarray(qvec), k_max)
            faiss_times.append((time.perf_counter() - t2) * 1000)

            gt_idx = file_to_idx[gt_img]
            rank = rank_of(gt_idx, idxs)
            t2i_ranks.append(rank)

            report.rankings.append({
                "mode": "t2i",
                "query": query_text,
                "ground_truth": gt_img,
                "found": rank is not None,
                "rank": rank,
                "top_k": [
                    {"rank": r + 1,
                     "filename": img_files[int(i)],
                     "similarity": float(s)}
                    for r, (i, s) in enumerate(zip(idxs, sims))
                ],
            })

        report.t2i_recall_at_1  = float(np.mean([recall_at(r, 1)  for r in t2i_ranks]))
        report.t2i_recall_at_5  = float(np.mean([recall_at(r, 5)  for r in t2i_ranks]))
        report.t2i_recall_at_10 = float(np.mean([recall_at(r, 10) for r in t2i_ranks]))
        report.t2i_mrr          = float(np.mean([reciprocal_rank(r) for r in t2i_ranks]))
        log.info("T2I  · R@1=%.3f  R@5=%.3f  R@10=%.3f  MRR=%.3f",
                 report.t2i_recall_at_1, report.t2i_recall_at_5,
                 report.t2i_recall_at_10, report.t2i_mrr)

    # ─────────────────────────────────────────────────────────────
    #  Image → Text  (uses the SAME pairs in reverse: the GT image
    #  is the query, and any caption containing the query text is
    #  considered relevant. Since we don't have caption→image
    #  ground truth here, we treat the original text query as the
    #  target caption and match by exact string.)
    # ─────────────────────────────────────────────────────────────
    if not skip_i2t:
        i2t_ranks: list[int | None] = []
        # Pre-compute caption-text → row index lookup (case-insensitive,
        # whitespace-normalised) so we can resolve the GT caption.
        capt_norm = [c.strip().lower() for c in capt_texts]

        for query_text, gt_img in test_pairs:
            if gt_img not in file_to_idx:
                i2t_ranks.append(None)
                report.rankings.append({
                    "mode": "i2t", "query_image": gt_img, "target_caption": query_text,
                    "found": False, "top_k": [], "rank": None,
                })
                continue

            # Encode the GT image  →  search captions
            from PIL import Image as PILImage
            # Image lives in the standard Flickr8k path. We tolerate
            # both project layouts (Flicker8k_Dataset is the upstream
            # name; some users symlink it).
            img_dir_candidates = [
                Path("Flickr8k_Dataset/Flicker8k_Dataset"),
                Path("Flickr8k_Dataset"),
                Path("Flickr8k_demo"),
            ]
            img_path = next((d / gt_img for d in img_dir_candidates if (d / gt_img).exists()), None)
            if img_path is None:
                log.warning("  Image file '%s' not found on disk — skipping I2T for this pair.", gt_img)
                i2t_ranks.append(None)
                continue

            pil_img = PILImage.open(img_path).convert("RGB")

            t1 = time.perf_counter()
            qvec = embedder.encode_image(pil_img)
            encode_times.append((time.perf_counter() - t1) * 1000)

            t2 = time.perf_counter()
            sims, idxs = _query_index(capt_index, np.asarray(qvec), k_max)
            faiss_times.append((time.perf_counter() - t2) * 1000)

            # GT caption: first caption row whose text matches the query.
            target_norm = query_text.strip().lower()
            try:
                gt_capt_idx = capt_norm.index(target_norm)
            except ValueError:
                # No exact match — use the highest-ranked caption that
                # actually corresponds to gt_img if metadata exists.
                # Fallback: report None, log it for transparency.
                gt_capt_idx = -1

            rank = rank_of(gt_capt_idx, idxs) if gt_capt_idx >= 0 else None
            i2t_ranks.append(rank)

            report.rankings.append({
                "mode": "i2t",
                "query_image": gt_img,
                "target_caption": query_text,
                "target_caption_indexed": gt_capt_idx >= 0,
                "found": rank is not None,
                "rank": rank,
                "top_k": [
                    {"rank": r + 1,
                     "caption": capt_texts[int(i)],
                     "similarity": float(s)}
                    for r, (i, s) in enumerate(zip(idxs, sims))
                ],
            })

        report.i2t_recall_at_1  = float(np.mean([recall_at(r, 1)  for r in i2t_ranks]))
        report.i2t_recall_at_5  = float(np.mean([recall_at(r, 5)  for r in i2t_ranks]))
        report.i2t_recall_at_10 = float(np.mean([recall_at(r, 10) for r in i2t_ranks]))
        report.i2t_mrr          = float(np.mean([reciprocal_rank(r) for r in i2t_ranks]))
        log.info("I2T  · R@1=%.3f  R@5=%.3f  R@10=%.3f  MRR=%.3f",
                 report.i2t_recall_at_1, report.i2t_recall_at_5,
                 report.i2t_recall_at_10, report.i2t_mrr)

    report.encode_ms_mean = round(float(np.mean(encode_times)), 2) if encode_times else 0.0
    report.faiss_ms_mean  = round(float(np.mean(faiss_times)),  3) if faiss_times  else 0.0
    log.info("Latency · encode μ=%.2f ms · FAISS μ=%.3f ms",
             report.encode_ms_mean, report.faiss_ms_mean)

    # Free GPU/CPU memory before loading the next backbone
    del embedder, img_index, capt_index
    gc.collect()

    return report


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "backbone_key", "backbone_id", "display_name", "n_queries",
    "embedding_dim", "n_images_indexed", "n_captions_indexed",
    "t2i_R@1", "t2i_R@5", "t2i_R@10", "t2i_MRR",
    "i2t_R@1", "i2t_R@5", "i2t_R@10", "i2t_MRR",
    "model_load_s", "memory_MB", "encode_ms", "faiss_ms",
]


def write_csv(reports: list[BackboneReport], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in reports:
            w.writerow([
                r.key, r.backbone_id, r.display_name, r.n_queries,
                r.embedding_dim, r.n_images_indexed, r.n_captions_indexed,
                f"{r.t2i_recall_at_1:.4f}",
                f"{r.t2i_recall_at_5:.4f}",
                f"{r.t2i_recall_at_10:.4f}",
                f"{r.t2i_mrr:.4f}",
                f"{r.i2t_recall_at_1:.4f}",
                f"{r.i2t_recall_at_5:.4f}",
                f"{r.i2t_recall_at_10:.4f}",
                f"{r.i2t_mrr:.4f}",
                f"{r.model_load_seconds:.2f}",
                f"{r.memory_mb_delta:.1f}",
                f"{r.encode_ms_mean:.2f}",
                f"{r.faiss_ms_mean:.3f}",
            ])
    log.info("Wrote %s", path)


def write_rankings_json(reports: list[BackboneReport], path: Path) -> None:
    """Per-query rankings for downstream PR-curve plotting."""
    payload = {
        r.key: {
            "backbone_id":  r.backbone_id,
            "display_name": r.display_name,
            "n_queries":    r.n_queries,
            "rankings":     r.rankings,
        }
        for r in reports
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s (%d backbones)", path, len(payload))


def write_failure_modes(reports: list[BackboneReport], path: Path) -> None:
    """Qualitative diff: which queries succeed on backbone A but fail on B."""
    if not reports:
        return

    # Build {(mode, query) -> {backbone_key: rank_or_None}}
    keyed: dict[tuple[str, str], dict[str, int | None]] = {}
    for r in reports:
        for row in r.rankings:
            q = row.get("query") or row.get("target_caption", "")
            mode = row["mode"]
            keyed.setdefault((mode, q), {})[r.key] = row.get("rank")

    lines = [
        "# Failure-Mode Analysis",
        "",
        "_Generated by `compare_backbones.py`. A query is **success@10** if the",
        "ground-truth item appears in the top-10 retrievals, **fail** otherwise._",
        "",
    ]

    for mode in ("t2i", "i2t"):
        sub = {k: v for k, v in keyed.items() if k[0] == mode}
        if not sub:
            continue
        lines.append(f"## {mode.upper()}  ({'Text → Image' if mode == 't2i' else 'Image → Text'})")
        lines.append("")
        header = "| Query | " + " | ".join(r.key for r in reports) + " |"
        sep    = "|---|" + "|".join(["---"] * len(reports)) + "|"
        lines += [header, sep]

        for (_, q), per_backbone in sub.items():
            row = [f"_{q[:60]}{'…' if len(q) > 60 else ''}_"]
            for r in reports:
                rank = per_backbone.get(r.key)
                if rank is None:
                    cell = "✗"
                elif rank <= 10:
                    cell = f"✓ ({rank})"
                else:
                    cell = f"✗ ({rank})"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Pair-wise disagreements summary
        lines.append("### Pair-wise disagreements (top-10 success)")
        lines.append("")
        for i, ra in enumerate(reports):
            for rb in reports[i + 1:]:
                a_only = b_only = both_ok = both_ko = 0
                for (_, q), per in sub.items():
                    a_ok = per.get(ra.key) is not None and per[ra.key] <= 10
                    b_ok = per.get(rb.key) is not None and per[rb.key] <= 10
                    if  a_ok and  b_ok:    both_ok += 1
                    elif a_ok and not b_ok: a_only += 1
                    elif b_ok and not a_ok: b_only += 1
                    else:                   both_ko += 1
                lines.append(
                    f"- **{ra.key}** vs **{rb.key}**: "
                    f"both ✓ = {both_ok} · only {ra.key} ✓ = {a_only} · "
                    f"only {rb.key} ✓ = {b_only} · both ✗ = {both_ko}"
                )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def print_summary_table(reports: list[BackboneReport]) -> None:
    """Pretty stdout summary, easy to paste into a report."""
    print("\n" + "═" * 78)
    print(" Table 5 — Backbone Comparison Summary")
    print("═" * 78)
    header = f"{'Backbone':<32} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'Enc ms':>8}"
    print(header)
    print("─" * 78)
    for r in reports:
        print(f"{r.display_name[:32]:<32} "
              f"{r.t2i_recall_at_1:>6.3f} "
              f"{r.t2i_recall_at_5:>6.3f} "
              f"{r.t2i_recall_at_10:>6.3f} "
              f"{r.t2i_mrr:>6.3f} "
              f"{r.encode_ms_mean:>8.1f}")
    print("═" * 78)
    print(" (T2I metrics shown. Full CSV has I2T columns and engineering cost.)")
    print()


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark all registered VLM backbones for the MIR project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backbones", nargs="+", default=list(BACKBONES.keys()),
        choices=list(BACKBONES.keys()),
        help="Subset of backbones to benchmark.",
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Max K for ranking metrics (R@K reported up to K=10).")
    parser.add_argument("--k-max-rankings", type=int, default=50,
                        help="Number of items kept per ranking dump (for PR curves).")
    parser.add_argument("--skip-t2i", action="store_true",
                        help="Skip Text→Image benchmark.")
    parser.add_argument("--skip-i2t", action="store_true",
                        help="Skip Image→Text benchmark (requires Flickr8k images on disk).")
    parser.add_argument("--out-csv",  default=str(RESULTS_DIR / "table5_backbone_comparison.csv"))
    parser.add_argument("--out-json", default=str(RESULTS_DIR / "backbone_rankings.json"))
    parser.add_argument("--out-md",   default=str(RESULTS_DIR / "failure_modes.md"))
    args = parser.parse_args()

    if args.skip_t2i and args.skip_i2t:
        log.error("--skip-t2i and --skip-i2t together leave nothing to benchmark.")
        return 2

    # Sanity: at least one backbone has its H5 files.
    log.info("Working directory: %s", Path.cwd().resolve())
    log.info("Looking for H5 files under: %s", H5_DIR.resolve())
    if not H5_DIR.exists():
        log.error("Descriptor directory %s does not exist — run build_flickr8k_*_db.py first.",
                  H5_DIR)
        return 1

    test_pairs = load_test_pairs()
    if not test_pairs:
        log.error("No test pairs available.")
        return 1
    log.info("Benchmarking %d backbones over %d query pairs.",
             len(args.backbones), len(test_pairs))

    k_max = max(args.k, args.k_max_rankings)
    reports: list[BackboneReport] = []
    for key in args.backbones:
        try:
            r = benchmark_backbone(key, test_pairs, k_max,
                                   args.skip_t2i, args.skip_i2t)
        except Exception:  # noqa: BLE001 — keep the run going for the others
            log.exception("Backbone '%s' failed — moving on.", key)
            continue
        if r is not None:
            reports.append(r)

    if not reports:
        log.error("No backbone produced a report. Nothing to write.")
        return 1

    write_csv(reports,           Path(args.out_csv))
    write_rankings_json(reports, Path(args.out_json))
    write_failure_modes(reports, Path(args.out_md))
    print_summary_table(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
