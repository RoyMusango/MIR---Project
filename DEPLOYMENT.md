# Group 07 — Deployment & Packaging Guide
*I-ILIA-014 · MIR Project · 2025-2026*

---

## 1 — Docker Quick-Start

### Build the image

```bash
# From the project root (where Dockerfile lives)
docker build -t mir-group07 .
```

> First build takes ~8-12 min (downloads PyTorch wheels + caches CLIP weights).
> Subsequent builds are fast — Docker caches the pip layer.

### Run locally

```bash
docker run -p 5000:5000 \
  -v "$(pwd)/descriptors":/app/descriptors:ro \
  -v "$(pwd)/dataset_voitures":/app/dataset_voitures:ro \
  -v "$(pwd)/Flickr8k_Dataset":/app/Flickr8k_Dataset:ro \
  mir-group07
```

Open **http://localhost:5000** in your browser.

> The three `-v` mounts inject the heavy data at runtime.
> The image itself stays lean (~2.5 GB with CPU-PyTorch).

---

## 2 — Free Cloud Deployment (Render)

Render gives a free 512 MB RAM instance — enough for the CLIP + FAISS demo if
HOG and ResNet50 descriptors are excluded from the web demo (they are excluded
as documented in the project state).

### Steps

1. Push the repo to GitHub (without the heavy `descriptors/` and dataset folders —
   add them to `.gitignore`).
2. Go to https://render.com → **New → Web Service** → connect your repo.
3. Set:
   - **Runtime**: Docker
   - **Start command**: *(leave blank — uses CMD in Dockerfile)*
4. Upload the HDF5 files and descriptor `.txt` files as a **Render Disk** (persistent
   storage, $0.25/GB/month) mounted at `/app/descriptors`.
5. Add the environment variable `PORT=5000`.
6. Deploy — Render auto-builds and exposes a public HTTPS URL.

### Alternative: Hugging Face Spaces (ZeroGPU)

1. Create a Space with **SDK: Docker**.
2. Push the same repo.
3. Upload the data files via the Space's *Files* tab or Git LFS.
4. HF Spaces automatically exposes port 7860 — change `EXPOSE` and the gunicorn
   bind in the Dockerfile to `0.0.0.0:7860`.

---


---

## 3 — `.dockerignore` (add to project root)

```
.git
.venv
__pycache__
*.pyc
*.pyo
dataset_voitures/
Flickr8k_Dataset/
descriptors/
results/
*.ipynb
*.egg-info
```

This prevents Docker from copying several GB of data into the build context.

---

## 4 — Submission Archive `Groupe_07.zip`

### Generate the archive

```bash
# From the project root
zip -r Groupe_07.zip . \
  --exclude "*.git*" \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude "dataset_voitures/*" \
  --exclude "Flickr8k_Dataset/*" \
  --exclude "descriptors/*" \
  --exclude "results/*"
```

### Required archive contents checklist

```
Groupe_07.zip
├── README.md
├── Dockerfile
├── requirements.txt
├── functions.py
├── clip_model.py                     ← 3-line shim
├── backbone_manager.py               ← NEW
├── evaluate.py
├── compare_backbones.py              ← NEW
├── plot_pr_curves.py                 ← NEW
├── build_flickr8k_clip_db.py
├── build_flickr8k_captions_db.py
├── descriptors_making.ipynb
├── embedders/                        ← NEW
│   ├── __init__.py
│   ├── base.py
│   ├── clip_hf.py
│   ├── open_clip.py
│   └── blip.py
├── Interface/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── login.html                ← NEW
│   └── static/
│       ├── css/style.css
│       └── img/
│           ├── logo.png              ← NEW (custom logo)
│           └── logo.svg              ← NEW (vector logo)
└── results/
    ├── table3_indexing_performance.csv
    ├── table4_query_metrics.csv
    ├── table5_backbone_comparison.csv  ← NEW (after running compare_backbones.py)
    └── *.png / *.pdf                   ← NEW (after running plot_pr_curves.py)
```

> **Do NOT include** the dataset folders or descriptor `.txt`/`.h5` files in the
> archive — the report PDF already contains all metrics.  
> Provide a `descriptors_download_link.txt` with your NextCloud / GDrive link
> if the grader needs to verify descriptor files.

---

## 5 — Memory budget on cloud

| Component | RAM | Include in web demo? |
|---|---|:---:|
| CLIP ViT-B/32 weights | ~600 MB | ✅ |
| OpenCLIP ViT-B/32 weights | ~600 MB | ✅ |
| OpenCLIP ViT-L/14 weights | ~1.7 GB | ✅ |
| BLIP ITM weights | ~900 MB | ✅ |
| ResNet50 descriptors | 253.9 MB | ⚠️ optional |
| HOG descriptors | 754.3 MB | ❌ too heavy |
| Flickr8k HDF5 image embeddings (×4 backbones) | ~200 MB | ✅ |
| Flickr8k HDF5 caption embeddings (×4 backbones) | ~800 MB | ✅ |
| Flask + FAISS + OS overhead | ~400 MB | — |
| **Worst case (all backbones loaded)** | **~5.4 GB** | |
| **Minimum (CLIP only)** | **~1.3 GB** | |

→ **HF Spaces CPU-Basic (16 GB RAM, ~$0.05/hr) is the target host.**  
→ Pause the Space when not in use — 25h of active runtime costs ~$1.25.  
→ Start the Space the evening before the exam (June 14th) to verify cold start.  
→ Worst-case multi-backbone load (~5.4 GB) is well within the 16 GB ceiling.  
→ Implement LRU eviction (`MIR_MAX_BACKBONES` env var, default 1) for free-tier fallback.