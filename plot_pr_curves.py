"""
plot_pr_curves.py
─────────────────────────────────────────────────────────────────────
Group 07 — I-ILIA-014 MIR Project
Phase 2: Visual analytics for retrieval performance.

Produces three publication-ready figures (PNG + PDF) under ``results/``:

    fig_multimodal_recall_at_k.{png,pdf}
        Recall@K curve, one line per VLM backbone, averaged over the
        canonical multimodal query set. The natural metric for caption
        retrieval (Flickr8k has 1 relevant image per text query and
        5 captions per image, so R@K is more informative than PR).

    fig_unimodal_pr_curves.{png,pdf}
        Standard interpolated PR curve, one line per unimodal descriptor.
        Consumes ``results/unimodal_rankings.json`` (see schema below).
        Skipped gracefully if that file is absent.

    fig_latency_quality_tradeoff.{png,pdf}
        Scatter: encode latency (ms) vs Recall@10. Lets the reader see
        the Pareto frontier in one glance — useful for the engineering
        discussion in the report.

Run:
    python plot_pr_curves.py                       # uses defaults
    python plot_pr_curves.py --no-unimodal         # multimodal only
    python plot_pr_curves.py --in-json results/backbone_rankings.json

Expected unimodal JSON schema (write this from your evaluate.py):
    {
      "<descriptor_name>": {
        "queries": [
          {
            "query_id": "0_1_BMW_X3_207.jpg",
            "n_relevant": 14,
            "ranked": [
              {"rank": 1, "filename": "...", "relevant": true},
              {"rank": 2, "filename": "...", "relevant": false},
              ...
            ]
          },
          ...
        ]
      },
      ...
    }

A 12-line snippet at the bottom of evaluate.py is enough to emit this
— see the docstring of ``_load_unimodal_rankings`` below.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Headless plotting — important when this is run on the HF Space or
# inside a Docker container with no display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(
    format="%(asctime)s · %(levelname)-7s · %(message)s",
    datefmt="%H:%M:%S",
    level=os.environ.get("MIR_LOG_LEVEL", "INFO").upper(),
)
log = logging.getLogger("plot_pr_curves")


# ─────────────────────────────────────────────────────────────────────
#  Visual identity — keep the four backbone lines distinguishable in
#  both color and (for B&W printing) line style.
# ─────────────────────────────────────────────────────────────────────
BACKBONE_STYLE: dict[str, dict] = {
    "clip":         {"color": "#5BC0EB", "linestyle": "-",  "marker": "o"},
    "openclip_b32": {"color": "#FDE74C", "linestyle": "--", "marker": "s"},
    "openclip_l14": {"color": "#9BC53D", "linestyle": "-.", "marker": "^"},
    "blip":         {"color": "#E55934", "linestyle": ":",  "marker": "D"},
}

DESCRIPTOR_PALETTE = [
    "#5BC0EB", "#FDE74C", "#9BC53D", "#E55934", "#FA7921",
    "#8338EC", "#3A86FF", "#FF006E", "#06D6A0",
]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "axes.titlecolor":   "black",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#CCCCCC",
    "legend.labelcolor": "black",
    "grid.color":        "#DDDDDD",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────────────
#  Multimodal: Recall@K curves
# ─────────────────────────────────────────────────────────────────────
def _ranks_per_query(rankings: list[dict], mode: str) -> list[int | None]:
    """Pull the 1-based rank of the relevant item from each query, for one mode."""
    out: list[int | None] = []
    for row in rankings:
        if row.get("mode") != mode:
            continue
        out.append(row.get("rank"))
    return out


def _recall_curve(ranks: list[int | None], k_max: int) -> np.ndarray:
    """Recall@K for K = 1..k_max, averaged across queries."""
    if not ranks:
        return np.zeros(k_max, dtype=float)
    arr = np.zeros((len(ranks), k_max), dtype=float)
    for i, r in enumerate(ranks):
        if r is not None and r <= k_max:
            arr[i, r - 1:] = 1.0  # step up at the rank where GT appears
    return arr.mean(axis=0)


def plot_multimodal_recall_at_k(
    backbones_data: dict, out_path_png: Path, out_path_pdf: Path, k_max: int = 50,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    for ax, mode, title in [
        (axes[0], "t2i", "Text → Image"),
        (axes[1], "i2t", "Image → Text"),
    ]:
        any_data = False
        for key, data in backbones_data.items():
            ranks = _ranks_per_query(data["rankings"], mode)
            if not ranks:
                continue
            curve = _recall_curve(ranks, k_max)
            style = BACKBONE_STYLE.get(key, {"color": "#888", "linestyle": "-", "marker": "x"})
            ax.plot(range(1, k_max + 1), curve,
                    label=data.get("display_name", key),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker=style["marker"],
                    markersize=4,
                    markevery=max(1, k_max // 10))
            any_data = True

        ax.set_title(title)
        ax.set_xlabel("K (top-K retrieved)")
        ax.set_ylabel("Recall@K")
        ax.set_xlim(1, k_max)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.6)
        if any_data:
            ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#9AA3B2")

    fig.suptitle("Multimodal Retrieval — Recall@K across VLM Backbones",
                 fontsize=14, color="#FFFFFF", weight="bold")

    for ext, path in [("png", out_path_png), ("pdf", out_path_pdf)]:
        fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Unimodal: interpolated PR curves
# ─────────────────────────────────────────────────────────────────────
def _interpolated_pr(ranked_relevance: list[bool], n_relevant: int,
                     recall_levels: np.ndarray) -> np.ndarray:
    """
    Compute interpolated precision at the given recall levels for a
    single query, following the TREC-style 11-point definition
    generalised to arbitrary recall grids.
    """
    if n_relevant == 0:
        return np.zeros_like(recall_levels)

    precisions = []
    recalls    = []
    tp = 0
    for rank, is_rel in enumerate(ranked_relevance, start=1):
        if is_rel:
            tp += 1
        precisions.append(tp / rank)
        recalls.append(tp / n_relevant)

    precisions = np.asarray(precisions)
    recalls    = np.asarray(recalls)

    interp = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        higher = precisions[recalls >= r]
        interp[i] = higher.max() if higher.size else 0.0
    return interp


def plot_unimodal_pr(
    unimodal_data: dict, out_path_png: Path, out_path_pdf: Path,
) -> None:
    if not unimodal_data:
        log.info("No unimodal data — skipping unimodal PR plot.")
        return

    recall_levels = np.linspace(0.0, 1.0, 11)  # classic 11-point grid
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for i, (desc_name, payload) in enumerate(unimodal_data.items()):
        queries = payload.get("queries", [])
        if not queries:
            continue
        per_query = []
        for q in queries:
            rel_flags = [bool(item.get("relevant")) for item in q.get("ranked", [])]
            n_rel = int(q.get("n_relevant", sum(rel_flags)))
            per_query.append(_interpolated_pr(rel_flags, n_rel, recall_levels))
        avg = np.mean(per_query, axis=0)
        color = DESCRIPTOR_PALETTE[i % len(DESCRIPTOR_PALETTE)]
        ax.plot(recall_levels, avg,
                label=desc_name, color=color, linewidth=2.0, marker="o", markersize=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision (interpolated)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Unimodal Cars Retrieval — Interpolated PR Curves")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)

    for ext, path in [("png", out_path_png), ("pdf", out_path_pdf)]:
        fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Latency-vs-quality scatter
# ─────────────────────────────────────────────────────────────────────
def plot_latency_vs_quality(
    backbones_data: dict, csv_path: Path, out_path_png: Path, out_path_pdf: Path,
) -> None:
    """Reads Table 5 to grab encode latency, joins with R@10 from rankings."""
    import csv
    if not csv_path.exists():
        log.warning("Table 5 CSV not found at %s — skipping latency plot.", csv_path)
        return

    rows = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["backbone_key"]] = row

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for key, data in backbones_data.items():
        if key not in rows:
            continue
        row = rows[key]
        try:
            latency = float(row["encode_ms"])
            r10     = float(row["t2i_R@10"])
        except (KeyError, ValueError):
            continue
        style = BACKBONE_STYLE.get(key, {"color": "#888", "marker": "o"})
        ax.scatter(latency, r10,
                   s=220, color=style["color"],
                   edgecolors="#FFFFFF", linewidths=1.5,
                   marker=style["marker"], zorder=3,
                   label=data.get("display_name", key))
        ax.annotate(key, (latency, r10),
                    xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color="#DDE2EA")

    ax.set_xlabel("Per-query encode latency (ms, CPU)")
    ax.set_ylabel("T2I Recall@10")
    ax.set_title("Quality–Latency Trade-off (lower-left worse · upper-left ideal)")
    ax.set_ylim(0.0, 1.02)
    # Pad the x-axis so right-side labels never clip.
    xlim_lo, xlim_hi = ax.get_xlim()
    ax.set_xlim(xlim_lo, xlim_hi + (xlim_hi - xlim_lo) * 0.12)
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    for ext, path in [("png", out_path_png), ("pdf", out_path_pdf)]:
        fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  I/O
# ─────────────────────────────────────────────────────────────────────
def _load_backbone_rankings(path: Path) -> dict:
    if not path.exists():
        log.error("Multimodal rankings JSON not found: %s — run compare_backbones.py first.", path)
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_unimodal_rankings(path: Path) -> dict:
    """
    Optional. To emit this from evaluate.py, append the equivalent of:

        import json
        dump = {}
        for desc_name, queries in per_descriptor_rankings.items():
            dump[desc_name] = {"queries": [
                {"query_id": q.id,
                 "n_relevant": q.n_relevant,
                 "ranked": [{"rank": r + 1,
                             "filename": fn,
                             "relevant": bool(rel)}
                            for r, (fn, rel) in enumerate(q.ranked)]}
                for q in queries
            ]}
        Path("results/unimodal_rankings.json").write_text(json.dumps(dump))
    """
    if not path.exists():
        log.warning("Unimodal rankings JSON not found: %s — unimodal PR plot will be skipped.", path)
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot retrieval PR / R@K curves for the MIR project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-json",        default="results/backbone_rankings.json",
                        help="Multimodal rankings (output of compare_backbones.py).")
    parser.add_argument("--in-unimodal",    default="results/unimodal_rankings.json",
                        help="Optional unimodal rankings (see docstring for schema).")
    parser.add_argument("--in-table5-csv",  default="results/table5_backbone_comparison.csv",
                        help="Table 5 CSV (for latency scatter).")
    parser.add_argument("--out-dir",        default="results",
                        help="Where to drop the PNG / PDF figures.")
    parser.add_argument("--k-max",          type=int, default=50,
                        help="Maximum K for the multimodal R@K curve.")
    parser.add_argument("--no-multimodal",  action="store_true")
    parser.add_argument("--no-unimodal",    action="store_true")
    parser.add_argument("--no-latency",     action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbones_data = {} if args.no_multimodal else _load_backbone_rankings(Path(args.in_json))
    unimodal_data  = {} if args.no_unimodal   else _load_unimodal_rankings(Path(args.in_unimodal))

    if not backbones_data and not unimodal_data:
        log.error("No data to plot. Run compare_backbones.py and/or emit unimodal_rankings.json.")
        return 1

    if backbones_data:
        plot_multimodal_recall_at_k(
            backbones_data,
            out_dir / "fig_multimodal_recall_at_k.png",
            out_dir / "fig_multimodal_recall_at_k.pdf",
            k_max=args.k_max,
        )

    if unimodal_data:
        plot_unimodal_pr(
            unimodal_data,
            out_dir / "fig_unimodal_pr_curves.png",
            out_dir / "fig_unimodal_pr_curves.pdf",
        )

    if backbones_data and not args.no_latency:
        plot_latency_vs_quality(
            backbones_data,
            Path(args.in_table5_csv),
            out_dir / "fig_latency_quality_tradeoff.png",
            out_dir / "fig_latency_quality_tradeoff.pdf",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
