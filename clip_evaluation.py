# Module CLIP evaluation - métriques Precision, Recall, mAP pour le moteur CLIP Flickr8k.

import numpy as np


def precision_at_k(retrieved_image_names, relevant_image_name, k):
    # Proportion de résultats pertinents parmi les k premiers.
    top_k = retrieved_image_names[:k]
    hits = sum(1 for name in top_k if name == relevant_image_name)
    return hits / k


def recall_at_k(retrieved_image_names, relevant_image_name, total_relevant, k):
    # Fraction des pertinents retrouvés parmi les k premiers.
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_image_names[:k]
    hits = sum(1 for name in top_k if name == relevant_image_name)
    return hits / total_relevant


def average_precision(retrieved_image_names, relevant_image_name, total_relevant):
    # Average Precision pour une requête texte->image (relevant = même image).
    if total_relevant == 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, name in enumerate(retrieved_image_names, start=1):
        if name == relevant_image_name:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / total_relevant


def compute_text_to_image_metrics(query_text, target_image, embedder, filenames, img_embeddings, k_values=(1, 5, 10)):
    # Encode le texte, cherche dans les embeddings images, calcule métriques.
    text_emb = embedder.encode_text(query_text)
    similarities = img_embeddings @ text_emb
    ranked_indices = np.argsort(similarities)[::-1]
    retrieved = [filenames[i] for i in ranked_indices]

    # total_relevant = nb d'images avec ce nom exact dans la base
    total_relevant = sum(1 for f in filenames if f == target_image)

    metrics = {}
    for k in k_values:
        metrics[f"P@{k}"] = round(precision_at_k(retrieved, target_image, k), 4)
        metrics[f"R@{k}"] = round(recall_at_k(retrieved, target_image, total_relevant, k), 4)
    metrics["AP"] = round(average_precision(retrieved, target_image, total_relevant), 4)
    return metrics


def compute_image_to_text_metrics(query_image, embedder, cap_image_names, cap_texts, txt_embeddings, k_values=(1, 5, 10)):
    # Encode l'image, cherche dans les embeddings textes, calcule métriques.
    from PIL import Image
    img = Image.open(query_image).convert("RGB")
    img_emb = embedder.encode_image(img)
    similarities = txt_embeddings @ img_emb
    ranked_indices = np.argsort(similarities)[::-1]

    query_basename = os.path.basename(query_image)
    retrieved_image_names = [cap_image_names[i] for i in ranked_indices]

    total_relevant = sum(1 for n in cap_image_names if n == query_basename)

    metrics = {}
    for k in k_values:
        metrics[f"P@{k}"] = round(precision_at_k(retrieved_image_names, query_basename, k), 4)
        metrics[f"R@{k}"] = round(recall_at_k(retrieved_image_names, query_basename, total_relevant, k), 4)
    metrics["AP"] = round(average_precision(retrieved_image_names, query_basename, total_relevant), 4)
    return metrics


import os
