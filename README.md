# 🔍 Machine & Deep Learning for Multimedia Retrieval

> A hybrid multimedia retrieval engine combining classical feature descriptors, deep learning (CNN + ViT), and CLIP-powered multimodal search, served through a modern Flask web interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/Meta-FAISS-0064D2?logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Descriptors & Distances](#descriptors--distances)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

This project implements a **Multimedia Image Retrieval (MIR)** system built for the course *I-ILIA-014 — Machine and Deep Learning for Multimedia Retrieval* at UMONS (2025–2026).

It supports two search modes:

| Mode | Input | Method | Dataset |
|------|-------|--------|---------|
| **Unimodal** | Query image | Feature descriptors + distance metrics | Cars dataset (~5,000 images, even-numbered group) |
| **Multimodal** | Text query or image | CLIP embeddings + FAISS index | Flickr8k (8,091 images / 40,455 captions) |

The unimodal engine uses classical computer vision descriptors alongside deep learning descriptors (ResNet50 CNN + CLIP ViT). The multimodal engine leverages [OpenAI CLIP](https://github.com/openai/CLIP) with a FAISS index to enable fast cross-modal search in both directions: **text → image** and **image → text**.

---

## Features

- 🖼️ **Image-to-Image Search** — Upload a car image or browse the indexed Cars dataset, then retrieve the most similar ones using up to 9 descriptors
- 📝 **Text-to-Image Search** — Type a natural language query and find matching Flickr8k images via CLIP + FAISS
- 🔄 **Image-to-Text Search** — Submit an image (upload or browse Flickr8k dataset) and retrieve the most semantically relevant captions
- 📂 **Interactive Dataset Browser** — Browse and select images directly from the Cars dataset or Flickr8k collection without uploading
- 🧠 **Deep Learning Descriptors** — ResNet50 (CNN) and CLIP ViT-B/32 alongside classical descriptors
- 📊 **Full Evaluation Metrics** — Precision, Recall, AP, mAP, R-Precision at Top-50 and Top-100
- ⚡ **Multi-Descriptor Fusion** — Combine multiple descriptors with normalized score averaging
- 🎨 **Modern Dark UI** — Responsive web interface with smooth animations and modal image preview

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Flask Web App                             │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │     Unimodal Mode       │  │      Multimodal Mode         │   │
│  │                         │  │                              │   │
│  │  Upload image           │  │  Text query / Image upload   │   │
│  │       ↓                 │  │          ↓                   │   │
│  │  Extract descriptor     │  │  CLIP encoder (text/image)   │   │
│  │  (classical / CNN / ViT)│  │          ↓                   │   │
│  │       ↓                 │  │  FAISS k-NN search           │   │
│  │  Compare with           │  │  (image index or text index) │   │
│  │  precomputed DB         │  │          ↓                   │   │
│  │       ↓                 │  │  Top-K images or captions    │   │
│  │  Top-K results          │  │                              │   │
│  │  + P/R/AP/mAP metrics   │  │  + P/R/mAP evaluation        │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── Interface/
│   ├── app.py                        # Flask app (routes, search logic, metrics)
│   ├── templates/
│   │   └── index.html                # Main UI (navbar, sidebars, results grid)
│   └── static/
│       └── css/
│           └── style.css             # Dark-themed responsive styles
│
├── descriptors/                      # Pre-computed descriptor files (git-ignored)
│   ├── Hist_Col/                     # Color histograms (.txt per image)
│   ├── HSV/                          # HSV histograms
│   ├── GLCM/                         # Gray-Level Co-occurrence Matrix
│   ├── LBP/                          # Local Binary Patterns
│   ├── HOG/                          # Histogram of Oriented Gradients
│   ├── SIFT/                         # Scale-Invariant Feature Transform
│   ├── ORB/                          # Oriented FAST and Rotated BRIEF
│   ├── ResNet50/                     # ResNet50 CNN features (2048-d)
│   ├── CLIP_Cars/                    # CLIP ViT-B/32 features for Cars dataset
│   └── CLIP_Flickr8k/                # CLIP embeddings for Flickr8k (HDF5)
│       ├── clip_flickr8k.h5          # Image embeddings (8091 × 512)
│       └── clip_flickr8k_captions.h5 # Caption embeddings (40455 × 512)
│
├── dataset_voitures/                 # Car dataset — git-ignored (download separately)
├── Flickr8k_Dataset/                 # Flickr8k images — git-ignored (download separately)
├── Flickr8k_text/                    # Flickr8k captions & splits
│
├── functions.py                      # All descriptor extraction & distance functions
├── clip_model.py                     # CLIP wrapper (encode_image, encode_text, batch)
├── build_clip_database.py            # Pre-compute CLIP embeddings for Cars dataset
├── build_flickr8k_clip_db.py         # Pre-compute CLIP image embeddings for Flickr8k
├── build_flickr8k_captions_db.py     # Pre-compute CLIP caption embeddings for Flickr8k
├── evaluate.py                       # Batch evaluation script → Tables 3 & 4 (CSV output)
├── descriptors_making.ipynb          # Notebook to compute traditional descriptors
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- *(Optional but recommended)* CUDA-capable GPU for faster CLIP encoding

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/RoyMusango/MIR---Project.git
cd MIR---Project
```

**2. Create a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary>Or install manually</summary>

```bash
pip install flask numpy opencv-python scikit-image scipy \
            torch torchvision transformers \
            h5py faiss-cpu tqdm Pillow
```

> Use `faiss-gpu` instead of `faiss-cpu` if you have a CUDA GPU.

</details>

**4. Download the datasets**

| Dataset | Destination folder |
|---------|-------------------|
| [Cars dataset](https://nextcloud.ig.umons.ac.be/s/BooirG6KkHJ58XB) | `dataset_voitures/dataset/` |
| [Flickr8k images](https://www.kaggle.com/datasets/adityajn105/flickr8k) | `Flickr8k_Dataset/Flicker8k_Dataset/` |
| Flickr8k captions (`Flickr8k.token.txt`) | `Flickr8k_text/` |

**5. Compute descriptors** *(skip if pre-computed files are provided)*

```bash
# Traditional descriptors (Color Hist, HSV, GLCM, LBP, HOG, SIFT, ORB)
# Run all cells in:
jupyter notebook descriptors_making.ipynb

# ResNet50 CNN descriptor
python -c "import functions as f; f.make_descriptor_as_files('dataset_voitures/dataset', f.generateResNet50, 'ResNet50')"

# CLIP ViT descriptor for Cars dataset
python build_clip_database.py

# CLIP embeddings for Flickr8k images (~6 min on CPU)
python build_flickr8k_clip_db.py

# CLIP embeddings for Flickr8k captions (~8 min on CPU)
python build_flickr8k_captions_db.py
```

---

## Usage

### Start the server

```bash
python Interface/app.py
```

Open your browser at **http://127.0.0.1:5000**

### Unimodal Image Search

1. Click the **"Unimodal Image Search"** tab
2. Either **upload** a query car image or click **"📁 Browse Dataset"** to select from the indexed Cars collection
3. Select one or more descriptors
4. Choose a distance metric
5. Adjust the number of results (K)
6. Click **Search** — Precision, Recall, AP, mAP and R-Precision are displayed automatically

### Multimodal Text → Image Search

1. Click the **"Multimodal Search"** tab
2. Type a text description (e.g. *"a dog running on the beach"*)
3. Click **Search by Text**

### Multimodal Image → Text Search

1. Click the **"Multimodal Search"** tab
2. Switch to **Image → Text** mode
3. Upload an image from Flickr8k (or any image)
4. Click **Search** — the top-K matching captions are returned with similarity scores

### Batch Evaluation (Tables 3 & 4)

```bash
python evaluate.py
```

Outputs two CSV files:
- `results/table3_indexing_performance.csv` — indexing time, descriptor size, avg search time
- `results/table4_query_metrics.csv` — R, P, AP, mAP, R-Precision for all 15 queries × top-3 descriptors

---

## Descriptors & Distances

### Image Descriptors

| Descriptor | Type | Dim. | Description |
|-----------|------|------|-------------|
| Color Histogram | Classical | 768 | RGB color distribution (256 bins/channel) |
| HSV Histogram | Classical | 768 | Hue-Saturation-Value distribution |
| GLCM | Classical | 4 | Gray-Level Co-occurrence Matrix (texture) |
| LBP | Classical | 256 | Local Binary Patterns (texture) |
| HOG | Classical | variable | Histogram of Oriented Gradients (shape) |
| SIFT | Keypoint | variable | Scale-Invariant Feature Transform |
| ORB | Keypoint | variable | Oriented FAST and Rotated BRIEF |
| **ResNet50** | **CNN (DL)** | **2048** | **Penultimate layer features, L2-normalized** |
| **ViT (Deep Learning)** | **ViT (DL)** | **768** | **Google ViT-Base-16 image encoder, L2-normalized** |
| **CLIP ViT-B/32** | **ViT (DL)** | **512** | **OpenAI CLIP image encoder, L2-normalized** |

### Distance Metrics

| Metric | Compatible with | Higher is better? |
|--------|----------------|:-----------------:|
| Euclidienne | Vector descriptors | ❌ |
| Chi carré | Vector descriptors | ❌ |
| Correlation | Vector descriptors | ✅ |
| Intersection | Vector descriptors | ✅ |
| Bhattacharyya | Vector descriptors | ❌ |
| Brute Force | Keypoint descriptors | ✅ |
| FLANN | Keypoint descriptors | ✅ |
| Cosine (FAISS) | DL embeddings | ✅ |

---

## Datasets

### Cars Dataset

10,000 images of cars across 10 classes (brands). Even-numbered groups process images with an even first digit (~5,000 images). Classes used for evaluation (even-numbered groups):

| Class | Brand | Queries |
|-------|-------|---------|
| 1 | Kia | R1, R2, R3 |
| 3 | Renault | R4, R5, R6 |
| 5 | Mercedes | R7, R8, R9 |
| 7 | Peugeot | R10, R11, R12 |
| 9 | Audi | R13, R14, R15 |

### Flickr8k

8,091 images from Flickr with 5 human-written captions per image (40,455 captions total). Used for multimodal cross-modal retrieval evaluation.

---

## Evaluation

### Unimodal (Part I)

Metrics computed at Top-50 and Top-100 for each of the 15 mandatory queries:

- **Precision (P)** — fraction of retrieved images that are relevant
- **Recall (R)** — fraction of relevant images that were retrieved
- **Average Precision (AP)** — area under the precision-recall curve for a single query
- **Mean Average Precision (mAP)** — mean of AP across all queries
- **R-Precision** — precision at rank R (where R = number of relevant images in the DB)

### Multimodal (Part II)

Metrics computed on a test corpus of 3 images and 3 text queries:

- **Recall@1 / @5 / @10** — standard cross-modal retrieval metrics
- **Precision@k** — precision at rank k
- **mAP** — mean average precision across test queries

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Project developed as part of I-ILIA-014 — UMONS, 2025–2026.*
