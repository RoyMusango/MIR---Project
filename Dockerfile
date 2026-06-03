# ─────────────────────────────────────────────────────────────────
#  Group 07 — I-ILIA-014 MIR Project  •  Dockerfile
#  Base: python:3.11-slim  (Debian Bookworm, no GUI stack)
#
#  Build:  docker build -t mir-group07 .
#  Run:    docker run -p 5000:5000 \
#            -v /absolute/path/descriptors:/app/descriptors:ro \
#            -v /absolute/path/dataset_voitures:/app/dataset_voitures:ro \
#            -v /absolute/path/Flickr8k_Dataset:/app/Flickr8k_Dataset:ro \
#            mir-group07
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────
# libgl1 / libglib2.0-0: needed by opencv-python-headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Application code ─────────────────────────────────────────────
# Copy only source files (heavy data folders are mounted at runtime)
COPY functions.py        .
COPY clip_model.py       .
COPY backbone_manager.py .
COPY embedders/          embedders/
COPY evaluate.py         .
COPY Interface/app.py    Interface/app.py
COPY Interface/templates Interface/templates
COPY Interface/static    Interface/static

# ── Non-root user (security best-practice) ───────────────────────
RUN useradd -m miruser && chown -R miruser /app
USER miruser

# ── Runtime upload folder ────────────────────────────────────────
RUN mkdir -p /app/Interface/uploads

# ── Pre-cache the CLIP model weights at build time ───────────────
# Avoids a slow cold-start on first request.
# Comment this out if the registry bandwidth / build time is a concern.
RUN python - <<'EOF'
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP weights cached.")
EOF

# ── Environment ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    # Tell HuggingFace to use the image-local cache, not ~/.cache
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# ── Expose & entrypoint ──────────────────────────────────────────
EXPOSE 5000

# gunicorn: 2 workers, 120 s timeout (CLIP encoding can be slow on CPU)
CMD ["gunicorn", \
     "--workers", "2", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "Interface.app:app"]
