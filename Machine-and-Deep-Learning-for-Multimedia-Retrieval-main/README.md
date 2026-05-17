# 🔍 Machine & Deep Learning for Multimedia Image Retrieval

> A hybrid image retrieval engine combining traditional feature descriptors with CLIP-powered text-to-image search, served through a modern Flask web interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?logo=openai&logoColor=white)
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
- [License](#license)

---

## Overview

This project implements a **Multimedia Image Retrieval (MIR)** system with two search modes:

| Mode | Input | Method | Dataset |
|------|-------|--------|---------|
| **Unimodal** | Query image | Feature descriptors + distance metrics | Car dataset (~5,000 images) |
| **Multimodal** | Text query | CLIP embeddings + cosine similarity | Flickr8k (~8,091 images) |

The unimodal engine uses classical computer vision descriptors (histograms, textures, keypoints) to find visually similar images. The multimodal engine leverages [OpenAI CLIP](https://github.com/openai/CLIP) to match free-form text descriptions to images from the Flickr8k dataset.

---

## Features

- 🖼️ **Image-to-Image Search** — Upload a car image and retrieve the most similar ones using 7 different descriptors
- 🔗 **Text-to-Image Search** — Type a natural language query and retrieve matching Flickr8k images via CLIP
- 📊 **Evaluation Metrics** — Precision and Recall at top-50 and top-100 for unimodal search
- ⚡ **Multi-Descriptor Fusion** — Combine multiple descriptors with normalized score averaging
- 🎨 **Modern Dark UI** — Responsive web interface with smooth animations and modal image preview

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flask Web App                     │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │  Unimodal Mode   │  │   Multimodal Mode        │ │
│  │                  │  │                          │ │
│  │  Upload image    │  │  Enter text query        │ │
│  │       ↓          │  │       ↓                  │ │
│  │  Extract desc.   │  │  CLIP text encoder       │ │
│  │       ↓          │  │       ↓                  │ │
│  │  Compare with    │  │  Cosine similarity vs    │ │
│  │  precomputed DB  │  │  precomputed image emb.  │ │
│  │       ↓          │  │       ↓                  │ │
│  │  Top-K results   │  │  Top-K results           │ │
│  └──────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── Interface/
│   ├── app.py                    # Flask application (routes, search logic)
│   ├── templates/
│   │   └── index.html            # Main UI template (navbar, sidebars, results grid)
│   └── static/
│       └── css/
│           └── style.css         # Dark-themed responsive styles
│
├── descriptors/                  # Pre-computed descriptor files
│   ├── Hist_Col/                 # Color histograms
│   ├── HSV/                      # HSV histograms
│   ├── GLCM/                     # Gray-Level Co-occurrence Matrix
│   ├── LBP/                      # Local Binary Patterns
│   ├── HOG/                      # Histogram of Oriented Gradients
│   ├── SIFT/                     # Scale-Invariant Feature Transform
│   ├── ORB/                      # Oriented FAST and Rotated BRIEF
│   └── CLIP_Flickr8k/            # CLIP image embeddings for Flickr8k (HDF5)
│
├── dataset_voitures/             # Car image dataset (~5,000 images)
├── Flickr8k_Dataset/             # Flickr8k images (~8,091 images)
├── Flickr8k_text/                # Flickr8k captions & splits
│
├── functions.py                  # Descriptor extraction & distance functions
├── clip_model.py                 # CLIP model wrapper (image + text encoding)
├── build_clip_database.py        # Pre-compute CLIP embeddings for car dataset
├── build_flickr8k_clip_db.py     # Pre-compute CLIP embeddings for Flickr8k
├── descriptors_making.ipynb      # Notebook for computing traditional descriptors
├── requirements.py               # Python dependencies (pip freeze)
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/sachadem/Machine-and-Deep-Learning-for-Multimedia-Retrieval.git
cd Machine-and-Deep-Learning-for-Multimedia-Retrieval
```

2. **Create a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install flask numpy opencv-python scikit-image scipy torch torchvision transformers h5py tqdm Pillow
```

4. **Download the datasets**

   - Place the **car dataset** in `dataset_voitures/dataset/`
   - Place the **Flickr8k images** in `Flickr8k_Dataset/Flicker8k_Dataset/`
   - Place the **Flickr8k text files** in `Flickr8k_text/`

5. **Compute traditional descriptors** (if not already present)

   Run the cells in `descriptors_making.ipynb` to generate descriptor files in `descriptors/`.

6. **Compute CLIP embeddings for Flickr8k**

```bash
python build_flickr8k_clip_db.py
```

> This encodes all 8,091 Flickr8k images with CLIP and saves them to `descriptors/CLIP_Flickr8k/clip_flickr8k.h5`. Takes ~6 minutes on CPU.

---

## Usage

### Start the server

```bash
python Interface/app.py
```

Open your browser at **http://127.0.0.1:5000**

### Unimodal Image Search

1. Click the **"Unimodal Image Search"** tab
2. Upload a query car image
3. Select one or more descriptors (e.g. Color Histogram, HOG)
4. Choose a distance metric
5. Adjust the number of results (K)
6. Click **Search**

### Multimodal Text Search

1. Click the **"Multimodal Search"** tab
2. Type a text description (e.g. *"a dog running on the beach"*)
3. Adjust the number of results (K)
4. Click **Search by Text**

---

## Descriptors & Distances

### Image Descriptors

| Descriptor | Type | Description |
|-----------|------|-------------|
| Color Histogram | Vector | RGB color distribution |
| HSV Histogram | Vector | Hue-Saturation-Value color distribution |
| GLCM | Vector | Gray-Level Co-occurrence Matrix (texture) |
| LBP | Vector | Local Binary Patterns (texture) |
| HOG | Vector | Histogram of Oriented Gradients (shape) |
| SIFT | Keypoint | Scale-Invariant Feature Transform |
| ORB | Keypoint | Oriented FAST and Rotated BRIEF |

### Distance Metrics

| Metric | Compatible with | Higher is better? |
|--------|----------------|-------------------|
| Euclidienne | Vector descriptors | No |
| Chi carré | Vector descriptors | No |
| Correlation | Vector descriptors | Yes |
| Intersection | Vector descriptors | Yes |
| Bhattacharyya | Vector descriptors | No |
| Brute Force | Keypoint descriptors | Yes |
| FLANN | Keypoint descriptors | Yes |

### CLIP (Multimodal)

Uses `openai/clip-vit-base-patch32` to encode both text and images into a shared 512-dimensional embedding space. Similarity is computed via dot product (cosine similarity on L2-normalized vectors).

---

## Datasets

### Car Dataset

~5,000 images of various car models organized by brand and model. Used for unimodal image-to-image retrieval.

### Flickr8k

8,091 images from the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) with 5 captions per image. Used for multimodal text-to-image retrieval. The captions file (`Flickr8k.token.txt`) is available for evaluation purposes but is not directly used during search — CLIP handles text-image matching natively.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
