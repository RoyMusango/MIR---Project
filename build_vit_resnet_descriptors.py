# Construction des descripteurs ViT et ResNet50.
# Extrait les features pour les images (1er chiffre pair) -> .txt
# Usage: python build_vit_resnet_descriptors.py

import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# chemins
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
DESCRIPTORS_PATH = os.path.join(PROJECT_ROOT, "descriptors")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

sys.path.insert(0, PROJECT_ROOT)


def load_vit_model():
    # Charge ViT avec poids fine-tunés si dispo.
    import torch
    from transformers import ViTModel, ViTImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)

    weights_path = os.path.join(WEIGHTS_DIR, "vit_finetuned.pth")
    if os.path.exists(weights_path):
        print(f"[ViT] Chargement poids fine-tunés: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"[ViT] {len(result.missing_keys)} clé(s) manquante(s)")
        print(f"[ViT] Modèle chargé sur {device}.")
    else:
        print(f"[WARN] Poids non trouvés: {weights_path}")
        print(f"[ViT] Pretrained par défaut sur {device}.")

    model.eval()
    return model, processor, device


def load_resnet_model():
    # Charge ResNet50 avec poids fine-tunés si dispo.
    import torch
    import torch.nn as nn
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    weights_path = os.path.join(WEIGHTS_DIR, "resnet_finetuned.pth")
    if os.path.exists(weights_path):
        print(f"[ResNet50] Chargement poids fine-tunés: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        resnet.load_state_dict(state_dict, strict=False)
        print(f"[ResNet50] Modèle chargé sur {device}.")
    else:
        print(f"[WARN] Poids non trouvés: {weights_path}")
        print(f"[ResNet50] Pretrained par défaut sur {device}.")

    # suppression FC -> 2048-dim
    model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    model.eval()
    return model, device


def extract_vit_feature(image, model, processor, device):
    # Extrait vecteur ViT 768-dim normalisé L2.
    import torch
    from PIL import Image as PILImage
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        # mean pooling patchs (sans CLS)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        mean_embedding = patch_embeddings.mean(dim=1)

    mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
    return mean_embedding.cpu().numpy().flatten().astype(np.float32)


def extract_resnet_feature(image, model, device):
    # Extrait vecteur ResNet50 2048-dim normalisé L2.
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)

    features = features.squeeze()
    features = features / features.norm()
    return features.cpu().numpy().astype(np.float32)


def main():
    from tqdm import tqdm
    import cv2

    print("=" * 60)
    print("  Construction descripteurs ViT & ResNet50")
    print("=" * 60)

    if not os.path.isdir(DATASET_PATH):
        print(f"[ERREUR] Dataset introuvable: {DATASET_PATH}")
        sys.exit(1)

    # récup images (1er chiffre pair)
    all_files = sorted(os.listdir(DATASET_PATH))
    image_files = [
        fn for fn in all_files
        if fn.endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and fn[0].isdigit() and int(fn[0]) % 2 == 0
    ]

    print(f"[INFO] {len(image_files)} images (1er chiffre pair)")

    if len(image_files) == 0:
        print("[ERREUR] Aucune image.")
        sys.exit(1)

    vit_dir = os.path.join(DESCRIPTORS_PATH, "ViT")
    resnet_dir = os.path.join(DESCRIPTORS_PATH, "ResNet")
    os.makedirs(vit_dir, exist_ok=True)
    os.makedirs(resnet_dir, exist_ok=True)

    # chargement modèles
    print(f"\n[INFO] Chargement modèles...")
    vit_model, vit_processor, vit_device = load_vit_model()
    resnet_model, resnet_device = load_resnet_model()

    # --- ViT ---
    print(f"\n[INFO] Extraction ViT...")
    start_time = time.time()
    vit_count = 0

    for filename in tqdm(image_files, desc="ViT"):
        img_path = os.path.join(DATASET_PATH, filename)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(vit_dir, f"{base_name}_ViT.txt")

        if os.path.exists(out_path):
            vit_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Lecture impossible: {filename}")
            continue

        try:
            feature = extract_vit_feature(img, vit_model, vit_processor, vit_device)
            np.savetxt(out_path, feature)
            vit_count += 1
        except Exception as e:
            print(f"[ERREUR] ViT {filename}: {e}")

    vit_elapsed = time.time() - start_time
    print(f"[INFO] ViT: {vit_count}/{len(image_files)} en {vit_elapsed:.1f}s")

    # --- ResNet50 ---
    print(f"\n[INFO] Extraction ResNet50...")
    start_time = time.time()
    resnet_count = 0

    for filename in tqdm(image_files, desc="ResNet50"):
        img_path = os.path.join(DATASET_PATH, filename)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(resnet_dir, f"{base_name}_ResNet.txt")

        if os.path.exists(out_path):
            resnet_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Lecture impossible: {filename}")
            continue

        try:
            feature = extract_resnet_feature(img, resnet_model, resnet_device)
            np.savetxt(out_path, feature)
            resnet_count += 1
        except Exception as e:
            print(f"[ERREUR] ResNet {filename}: {e}")

    resnet_elapsed = time.time() - start_time
    print(f"[INFO] ResNet50: {resnet_count}/{len(image_files)} en {resnet_elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"  Terminé!")
    print(f"  ViT:      {vit_count} descripteurs dans {vit_dir}")
    print(f"  ResNet50: {resnet_count} descripteurs dans {resnet_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
