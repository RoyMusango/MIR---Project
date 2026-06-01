"""
Script simple pour générer un document d'évaluation.

Exemples:
    python run_evaluation.py "Color Histogram" "Euclidienne" eval_color.xls
    python run_evaluation.py "HOG" "Euclidienne" eval_hog.xls
    python run_evaluation.py "SIFT" "Brute force" eval_sift.xls
    python run_evaluation.py "ViT" "Cosine Similarity" eval_vit.xls
    python run_evaluation.py "ResNet" "Cosine Similarity" eval_resnet.xls
"""

import sys
import os
from fill_evaluation_document import fill_evaluation_document

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nUtilisation: python run_evaluation.py <descriptor> <comparator> <output_file>")
        print("\nDescripteurs disponibles:")
        print("  - Color Histogram, HSV Histogram (couleur)")
        print("  - GLCM, LBP (texture)")
        print("  - HOG (forme)")
        print("  - SIFT, ORB (points-clés)")
        print("  - ViT, ResNet (deep learning)")
        print("\nComparateurs:")
        print("  - Euclidienne, Chi carre, Correlation, Intersection, Bhattacharyya (vecteurs)")
        print("  - Brute force, Flann (keypoints)")
        print("  - Cosine Similarity (deep learning)")
        return

    descriptor = sys.argv[1]
    comparator = sys.argv[2]
    output_file = sys.argv[3]
    db_path = sys.argv[4] if len(sys.argv) > 4 else "dataset_voitures/dataset"

    print(f"\nGénération d'un document d'évaluation...")
    print(f"  Descripteur: {descriptor}")
    print(f"  Comparateur: {comparator}")
    print(f"  Fichier sortie: {output_file}")
    print(f"  Dataset: {db_path}\n")

    result = fill_evaluation_document(
        dataset_test_path="dataset_test",
        descriptor_names=descriptor,
        comparator=comparator,
        output_filename=output_file,
        db_dataset_path=db_path
    )

    print("\n" + "="*60)
    print("RÉSUMÉ:")
    print("="*60)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
