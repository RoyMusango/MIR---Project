# Fine-tuning ViT & ResNet50 sur le dataset voitures.
# Entraîne les modèles sur les classes paires puis sauvegarde les poids backbone.
# Usage:
#python finetune_models.py                  # les deux modèles
#python finetune_models.py --model vit      # ViT seulement
#python finetune_models.py --model resnet   # ResNet seulement
#python finetune_models.py --epochs 15      # nb epochs custom

import os
import sys
import argparse
import time
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# chemins
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset_voitures", "dataset")
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")


class CarDataset(Dataset):
    # Dataset voitures. Filtre: 1er chiffre pair. Label = 'chiffre1_chiffre2'.

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        # récup images 1er chiffre pair
        all_files = sorted(os.listdir(dataset_path))
        self.image_files = [
            f for f in all_files
            if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and f[0].isdigit() and int(f[0]) % 2 == 0
        ]

        # mapping classes depuis noms de fichiers
        class_names = sorted(set(
            "_".join(f.split("_")[:2]) for f in self.image_files
        ))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(class_names)

        print(f"[Dataset] {len(self.image_files)} images, {self.num_classes} classes")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.dataset_path, filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # extraction label
        class_key = "_".join(filename.split("_")[:2])
        label = self.class_to_idx[class_key]

        return image, label


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # Entraîne 1 epoch. Retourne (loss_moy, accuracy).
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # gestion format sortie ViT (transformers)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    # Évaluation sur le set de validation. Retourne (loss_moy, accuracy).
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                model_name="model"):
    # Boucle d'entraînement complète avec early stopping.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    best_state = None
    patience = 5
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Entraînement {model_name}")
    print(f"  Epochs: {num_epochs} | LR: {lr} | Device: {device}")
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:>3}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    ★ Meilleure val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping epoch {epoch} (pas d'amélioration depuis {patience} epochs)")
                break

    print(f"\n  Meilleure val accuracy: {best_val_acc:.4f}")
    return best_state, best_val_acc


def finetune_vit(dataset_path, num_epochs=20, lr=2e-5, batch_size=16):
    # Fine-tune ViT sur dataset voitures. Sauvegarde poids backbone.
    from transformers import ViTForImageClassification, ViTImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ViT] Chargement processeur et modèle...")

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # transforms entraînement (data augmentation)
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        ),
    ])

    # transforms validation (pas d'augmentation)
    vit_transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        ),
    ])

    # dataset + split train/val
    full_dataset = CarDataset(dataset_path, transform=vit_transform)
    num_classes = full_dataset.num_classes

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset = CarDataset(dataset_path, transform=vit_transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # chargement ViT avec tête de classification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    best_state, best_acc = train_model(
        model, train_loader, val_loader, num_epochs, lr, device,
        model_name="ViT (google/vit-base-patch16-224)"
    )

    # sauvegarde backbone seul (sans classifieur)
    model.load_state_dict(best_state)
    backbone_state = model.vit.state_dict()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, "vit_finetuned.pth")
    torch.save(backbone_state, save_path)
    print(f"[ViT] Poids backbone sauvés: {save_path}")

    # sauvegarde mapping classes
    mapping_path = os.path.join(WEIGHTS_DIR, "vit_classes.json")
    with open(mapping_path, "w") as fp:
        json.dump(full_dataset.class_to_idx, fp, indent=2)
    print(f"[ViT] Mapping classes sauvé: {mapping_path}")

    return best_acc


def finetune_resnet(dataset_path, num_epochs=20, lr=1e-4, batch_size=32):
    # Fine-tune ResNet50 sur dataset voitures. Sauvegarde poids backbone.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ResNet50] Chargement modèle pretrained...")

    # transforms entraînement avec augmentation
    resnet_transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # transforms validation
    resnet_transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # dataset + split
    full_dataset = CarDataset(dataset_path, transform=resnet_transform_train)
    num_classes = full_dataset.num_classes

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset = CarDataset(dataset_path, transform=resnet_transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ResNet50 + remplacement FC pour nb classes
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    model = resnet.to(device)

    best_state, best_acc = train_model(
        model, train_loader, val_loader, num_epochs, lr, device,
        model_name="ResNet50"
    )

    # sauvegarde backbone seul (sans FC)
    model.load_state_dict(best_state)
    backbone_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("fc.")
    }

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, "resnet_finetuned.pth")
    torch.save(backbone_state, save_path)
    print(f"[ResNet50] Poids backbone sauvés: {save_path}")

    # sauvegarde mapping classes
    mapping_path = os.path.join(WEIGHTS_DIR, "resnet_classes.json")
    with open(mapping_path, "w") as fp:
        json.dump(full_dataset.class_to_idx, fp, indent=2)
    print(f"[ResNet50] Mapping classes sauvé: {mapping_path}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT et/ou ResNet50 sur le dataset voitures"
    )
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["vit", "resnet", "both"],
        help="Modèle à fine-tuner (défaut: both)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Nombre d'epochs (défaut: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Taille de batch (défaut: 16)"
    )
    parser.add_argument(
        "--lr-vit", type=float, default=2e-5,
        help="Learning rate ViT (défaut: 2e-5)"
    )
    parser.add_argument(
        "--lr-resnet", type=float, default=1e-4,
        help="Learning rate ResNet (défaut: 1e-4)"
    )
    args = parser.parse_args()

    # vérif dataset
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERREUR] Dataset introuvable: {DATASET_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("  Fine-tuning ViT & ResNet50 - Dataset Voitures")
    print("  (classes 1er chiffre pair)")
    print("=" * 60)

    results = {}

    if args.model in ("vit", "both"):
        acc = finetune_vit(
            DATASET_PATH,
            num_epochs=args.epochs,
            lr=args.lr_vit,
            batch_size=args.batch_size
        )
        results["vit"] = acc

    if args.model in ("resnet", "both"):
        acc = finetune_resnet(
            DATASET_PATH,
            num_epochs=args.epochs,
            lr=args.lr_resnet,
            batch_size=args.batch_size
        )
        results["resnet"] = acc

    # résumé
    print(f"\n{'=' * 60}")
    print("  Fine-tuning terminé!")
    for name, acc in results.items():
        print(f"  {name.upper():>8}: meilleure val acc = {acc:.4f}")
    print(f"  Poids sauvés dans: {WEIGHTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
