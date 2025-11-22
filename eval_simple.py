"""
Simple MCQ evaluation script with negation decomposition.

This script evaluates CLIP on a fixed set of 30 MCQ samples (10 negative, 10 positive, 10 hybrid)
with configurable alpha parameter for negation decomposition.

All parameters are fixed except for alpha:
- Model: ViT-B-32 (OpenAI)
- Dataset: eval_simple_data/mcq_simple.csv (30 samples)
- Decomposition: Always enabled
- Batch size: 8
- Device: Auto-detect (CUDA if available)

Usage:
    python eval_simple.py --alpha 0.5
    python eval_simple.py --alpha 0.0  # No negation decomposition (baseline)
    python eval_simple.py --alpha 1.0  # Full negation subtraction
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
    print_results,
)
from negation_encoder import NegationDecompositionEncoder


# Fixed configuration
FIXED_CONFIG = {
    "model": "ViT-B-32",
    "pretrained": "openai",
    "dataset": "eval_simple_data/mcq_simple.csv",
    "negbench_root": "../negbench",
    "batch_size": 8,
    "num_workers": 0,
    "use_decomposition": True,
}


def evaluate_mcq_simple(
    model,
    tokenizer,
    dataloader,
    device,
    alpha=0.5,
):
    """
    Evaluate model on simplified MCQ task.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        dataloader: MCQ dataloader
        device: Device to use
        alpha: Alpha parameter for negation decomposition

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

    print(f"\nEvaluating MCQ (alpha={alpha})...")
    print("=" * 60)

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

                # Encode text with decomposition
                text_features = encoder.encode_text(
                    captions,
                    normalize=True,
                    use_decomposition=FIXED_CONFIG["use_decomposition"],
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
        "use_decomposition": FIXED_CONFIG["use_decomposition"],
        "model": FIXED_CONFIG["model"],
        "pretrained": FIXED_CONFIG["pretrained"],
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple MCQ evaluation with configurable alpha parameter"
    )

    # Only alpha is configurable
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for negative embedding weight (0.0 = baseline, 1.0 = full subtraction)",
    )

    args = parser.parse_args()

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Simple MCQ Evaluation with Negation Decomposition")
    print("=" * 60)
    print(f"Model: {FIXED_CONFIG['model']} ({FIXED_CONFIG['pretrained']})")
    print(f"Dataset: {FIXED_CONFIG['dataset']}")
    print(f"Samples: 30 (10 negative, 10 positive, 10 hybrid)")
    print(f"Alpha: {args.alpha}")
    print(f"Decomposition: {FIXED_CONFIG['use_decomposition']}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    print("\nLoading CLIP model...")
    model, _, preprocess_val, tokenizer = load_clip_model(
        model_name=FIXED_CONFIG["model"],
        pretrained=FIXED_CONFIG["pretrained"],
        device=device,
    )

    # Get the script directory to resolve dataset path
    script_dir = Path(__file__).parent
    dataset_path = script_dir / FIXED_CONFIG["dataset"]

    # Use eval_simple_data as the root for image path resolution
    # (images are stored locally in eval_simple_data/images/)
    data_root = script_dir / "eval_simple_data"

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = MCQDataset(
        csv_path=str(dataset_path),
        transform=preprocess_val,
        negbench_root=str(data_root),
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=FIXED_CONFIG["batch_size"],
        num_workers=FIXED_CONFIG["num_workers"],
        collate_fn=collate_mcq,
    )

    # Evaluate
    results = evaluate_mcq_simple(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        alpha=args.alpha,
    )

    # Print results
    print_results(results, title="Simple MCQ Evaluation Results")

    return results


if __name__ == "__main__":
    main()
