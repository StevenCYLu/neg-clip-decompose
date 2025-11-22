"""
Image-text retrieval evaluation with negation decomposition.

This script evaluates CLIP models on retrieval tasks from negbench,
with optional negation decomposition for text encoding.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils import (
    load_clip_model,
    RetrievalDataset,
    create_dataloader,
    collate_retrieval,
    save_results,
    print_results,
)
from negation_encoder import NegationDecompositionEncoder


def evaluate_retrieval(
    model,
    tokenizer,
    dataloader,
    device,
    alpha=0.5,
    use_decomposition=True,
    recall_k_list=[1, 5, 10],
):
    """
    Evaluate model on image-text retrieval task.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        dataloader: Retrieval dataloader
        device: Device to use
        alpha: Alpha parameter for negation decomposition
        use_decomposition: If True, use negation decomposition
        recall_k_list: List of K values for Recall@K metric

    Returns:
        Dictionary with evaluation metrics
    """
    # Create negation encoder
    encoder = NegationDecompositionEncoder(model, tokenizer, device, alpha)

    # Collect all embeddings
    all_image_features = []
    all_text_features = []
    num_captions_per_image = []

    print(f"\nEvaluating Retrieval (decomposition={use_decomposition}, alpha={alpha})...")
    print("Encoding images and texts...")

    with torch.no_grad():
        for images, captions_batch, _ in tqdm(dataloader):
            batch_size = images.size(0)

            # Move images to device
            images = images.to(device)

            # Encode images
            image_features = model.encode_image(images, normalize=False)
            image_features = F.normalize(image_features, dim=-1)
            all_image_features.append(image_features.cpu())

            # Encode texts for each image
            for i in range(batch_size):
                captions = captions_batch[i]
                num_captions_per_image.append(len(captions))

                # Encode text with or without decomposition
                text_features = encoder.encode_text(
                    captions,
                    normalize=True,
                    use_decomposition=use_decomposition,
                )
                all_text_features.append(text_features.cpu())

    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)  # (num_images, dim)

    # Handle text features (each image may have multiple captions)
    print("Computing retrieval metrics...")

    # Create mapping from text index to image index
    text_to_image = []
    for img_idx, num_caps in enumerate(num_captions_per_image):
        text_to_image.extend([img_idx] * num_caps)

    text_to_image = np.array(text_to_image)

    # Concatenate all text features
    all_text_features = torch.cat(all_text_features, dim=0)  # (num_texts, dim)

    # Compute similarity matrix: (num_texts, num_images)
    similarity_matrix = all_text_features @ all_image_features.t()

    # Compute text-to-image retrieval metrics
    num_texts = similarity_matrix.size(0)
    num_images = all_image_features.size(0)

    recall_at_k = {}

    for k in recall_k_list:
        correct = 0

        for text_idx in range(num_texts):
            # Get similarities for this text
            sims = similarity_matrix[text_idx]

            # Get top-k image indices
            top_k_indices = torch.topk(sims, k=min(k, num_images)).indices.numpy()

            # Check if correct image is in top-k
            correct_image_idx = text_to_image[text_idx]
            if correct_image_idx in top_k_indices:
                correct += 1

        recall_at_k[f"recall@{k}"] = correct / num_texts

    results = {
        **recall_at_k,
        "num_images": num_images,
        "num_texts": num_texts,
        "alpha": alpha,
        "use_decomposition": use_decomposition,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP on retrieval with negation decomposition"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights",
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to retrieval CSV file (relative to negbench or absolute)",
    )
    parser.add_argument(
        "--negbench-root",
        type=str,
        default="../negbench",
        help="Root directory of negbench",
    )

    # Decomposition arguments
    parser.add_argument(
        "--use-decomposition",
        action="store_true",
        help="Use negation decomposition",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for negative embedding weight",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for Recall@K metric",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (JSON)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Retrieval Evaluation with Negation Decomposition")
    print("=" * 60)
    print(f"Model: {args.model} ({args.pretrained})")
    print(f"Dataset: {args.dataset}")
    print(f"Decomposition: {args.use_decomposition}")
    print(f"Alpha: {args.alpha}")
    print(f"Recall@K: {args.recall_k}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    model, _, preprocess_val, tokenizer = load_clip_model(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    # Load dataset
    dataset = RetrievalDataset(
        csv_path=args.dataset,
        transform=preprocess_val,
        negbench_root=args.negbench_root,
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_retrieval,
    )

    # Evaluate
    results = evaluate_retrieval(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=args.device,
        alpha=args.alpha,
        use_decomposition=args.use_decomposition,
        recall_k_list=args.recall_k,
    )

    # Print results
    print_results(results, title="Retrieval Evaluation Results")

    # Save results
    if args.output:
        save_results(results, args.output)

    return results


if __name__ == "__main__":
    main()
