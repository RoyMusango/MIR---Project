# HF Spaces Deployment — Corrected Step-by-Step

## Before you start

You need a Hugging Face account (free, no credit card) and `git-lfs` installed locally:

```bash
# macOS
brew install git-lfs
# Ubuntu / Debian
sudo apt-get install git-lfs
# Windows
# Download from https://git-lfs.github.com
git lfs install
```

## Step 1 — Create the Space

1. Go to https://huggingface.co/new-space
2. Owner: your username · Name: `mir-group07` (or whatever)
3. License: MIT
4. **SDK: Docker** → **Template: Blank**
5. Visibility: Public
6. Click **Create Space**

HF gives you a git URL like `https://huggingface.co/spaces/your-username/mir-group07`

## Step 2 — Clone, populate, push

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/your-username/mir-group07
cd mir-group07

# Track the heavy files with Git LFS BEFORE adding them
git lfs track "*.h5"
git lfs track "descriptors/**/*.txt"
git lfs track "*.jpg"
git add .gitattributes

# Copy your project files in
cp ../MIR---Project/Dockerfile.hfspaces ./Dockerfile
cp ../MIR---Project/README_hfspace.md   ./README.md   # ← critical: HF reads this
cp ../MIR---Project/requirements.txt    .
cp ../MIR---Project/.dockerignore       .
cp ../MIR---Project/functions.py        .
cp ../MIR---Project/clip_model.py       .
cp ../MIR---Project/backbone_manager.py .
cp -r ../MIR---Project/embedders        .
cp ../MIR---Project/evaluate.py         .
cp -r ../MIR---Project/Interface        .

# Full datasets — HF Spaces CPU-Basic (16 GB RAM) handles everything
cp -r ../MIR---Project/dataset_voitures .
cp -r ../MIR---Project/Flickr8k_Dataset .
mkdir -p descriptors/CLIP_Flickr8k
cp ../MIR---Project/descriptors/CLIP_Flickr8k/*.h5 descriptors/CLIP_Flickr8k/
mkdir -p descriptors/CLIP descriptors/ResNet50
cp -r ../MIR---Project/descriptors/CLIP/. descriptors/CLIP/
cp -r ../MIR---Project/descriptors/ResNet50/. descriptors/ResNet50/

# Commit and push
git add .
git commit -m "Initial deploy: Group 07 MIR engine"
git push
```

HF builds the container in 5-10 min. Watch the build logs in the Space UI.
When status flips to **Running**, your public URL is `https://your-username-mir-group07.hf.space/`.

## Step 3 — Common build failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Build succeeds, page is blank | Port mismatch | Confirm `app_port: 7860` in `README.md` frontmatter AND Dockerfile EXPOSEs 7860 |
| `OSError: libGL.so.1` | Using `opencv-python` instead of headless | Already fixed in our `requirements.txt` |
| Out of memory during build | CLIP weight pre-cache too heavy | Remove the `RUN python -c "CLIPModel..."` block — accept the cold-start cost |
| `huggingface_hub` import error | Version pin too aggressive | Loosen the pin: `huggingface_hub>=0.24` |
| Files >10 MB rejected on push | Not tracked by LFS | `git lfs migrate import --include="*.h5,*.txt"` then re-push |
| Space evicted at runtime | 50 GB ephemeral disk exceeded | Trim descriptor files; HDF5 is fine |
| `ModuleNotFoundError: embedders` | `embedders/` dir not copied to Space repo | `cp -r embedders .` then re-push |

## Step 4 — One line to add in your IEEE report

> *The retrieval engine was deployed to Hugging Face Spaces as a Docker container
> at `https://your-username-mir-group07.hf.space/`. Three contrastive VLM backbones
> (CLIP ViT-B/32, OpenCLIP ViT-L/14 LAION-2B, BLIP ITM) are available via a live
> selector; models are lazy-loaded on first selection and cached for the session.
> The Linux GUI dependency chain was avoided using `opencv-python-headless`;
> the Flask server is fronted by gunicorn with a 120s worker timeout to absorb
> CPU inference latency. The HF Spaces CPU-Basic tier (16 GB RAM) comfortably
> hosts all FAISS indices and model weights simultaneously.*
