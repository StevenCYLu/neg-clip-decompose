"""
Multiple Choice Question (MCQ) evaluation with negation decomposition.

This script evaluates CLIP models on MCQ tasks from negbench,
with optional negation decomposition for text encoding.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_clip_model,
    MCQDataset,
    create_dataloader,
    collate_mcq,
    save_results,
    print_results,
)
from negation_encoder import NegationDecompositionEncoder


def evaluate_mcq(
    model,
    tokenizer,
    dataloader,
    device,
    alpha=0.5,
    use_decomposition=True,
):
    """
    Evaluate model on MCQ task.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        dataloader: MCQ dataloader
        device: Device to use
        alpha: Alpha parameter for negation decomposition
        use_decomposition: If True, use negation decomposition

    Returns:
        Dictionary with evaluation metrics
    """
    # Create negation encoder
    encoder = NegationDecompositionEncoder(model, tokenizer, device, alpha)

    total_correct = 0
    total_samples = 0

    # Track accuracy by answer template type
    template_correct = {"positive": 0, "negative": 0, "hybrid": 0}
    template_total = {"positive": 0, "negative": 0, "hybrid": 0}

    print(f"\nEvaluating MCQ (decomposition={use_decomposition}, alpha={alpha})...")

    with torch.no_grad():
        for images, captions_batch, correct_answers, answer_templates in tqdm(dataloader):
            batch_size = images.size(0)

            # Move images to device
            images = images.to(device)

            # Encode images
            image_features = model.encode_image(images, normalize=False)
            image_features = F.normalize(image_features, dim=-1)

            # Process each sample in the batch
            for i in range(batch_size):
                # Get 4 captions for this sample
                captions = captions_batch[i]

                # Encode text with or without decomposition
                text_features = encoder.encode_text(
                    captions,
                    normalize=True,
                    use_decomposition=use_decomposition,
                )

                # Compute similarity scores
                # image_features[i] is (embedding_dim,)
                # text_features is (4, embedding_dim)
                scores = text_features @ image_features[i]

                # Get prediction
                predicted = torch.argmax(scores).item()
                correct = correct_answers[i].item()

                # Update metrics
                is_correct = predicted == correct
                total_correct += is_correct
                total_samples += 1

                # Update template-specific metrics
                template = answer_templates[i]
                if template in template_correct:
                    template_correct[template] += is_correct
                    template_total[template] += 1

    # Compute overall accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Compute template-specific accuracies
    template_accuracies = {}
    for template in ["positive", "negative", "hybrid"]:
        if template_total[template] > 0:
            template_accuracies[f"accuracy_{template}"] = (
                template_correct[template] / template_total[template]
            )
        else:
            template_accuracies[f"accuracy_{template}"] = 0.0

    results = {
        "overall_accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": total_correct,
        **template_accuracies,
        "alpha": alpha,
        "use_decomposition": use_decomposition,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP on MCQ with negation decomposition"
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
        help="Path to MCQ CSV file (relative to negbench or absolute)",
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
    print("MCQ Evaluation with Negation Decomposition")
    print("=" * 60)
    print(f"Model: {args.model} ({args.pretrained})")
    print(f"Dataset: {args.dataset}")
    print(f"Decomposition: {args.use_decomposition}")
    print(f"Alpha: {args.alpha}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    model, _, preprocess_val, tokenizer = load_clip_model(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    # Load dataset
    dataset = MCQDataset(
        csv_path=args.dataset,
        transform=preprocess_val,
        negbench_root=args.negbench_root,
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_mcq,
    )

    # Evaluate
    results = evaluate_mcq(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=args.device,
        alpha=args.alpha,
        use_decomposition=args.use_decomposition,
    )

    # Print results
    print_results(results, title="MCQ Evaluation Results")

    # Save results
    if args.output:
        save_results(results, args.output)

    return results


if __name__ == "__main__":
    main()
