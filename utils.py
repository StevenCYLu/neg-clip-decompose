"""
Utility functions for negation decomposition evaluation.

This module provides helpers for:
- Loading CLIP models
- Loading datasets from negbench
- Data preprocessing
- Results logging
"""

import sys
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# Add negbench to path
NEGBENCH_PATH = Path(__file__).parent.parent / "negbench" / "benchmarks" / "src"
sys.path.insert(0, str(NEGBENCH_PATH))

from open_clip import create_model_and_transforms, get_tokenizer
import open_clip

from path_config import PathMapper


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
    precision: str = "fp32",
):
    """
    Load a CLIP model using negbench's open_clip.

    Args:
        model_name: Model architecture (e.g., "ViT-B-32", "ViT-L-14")
        pretrained: Pretrained weights ("openai", "laion400m_e32", etc.)
        device: Device to load model on
        precision: Precision ("fp32", "fp16", "bf16")

    Returns:
        (model, preprocess_train, preprocess_val, tokenizer) tuple
    """
    print(f"Loading CLIP model: {model_name} with {pretrained} weights")

    # Create model and transforms
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device=device,
        output_dict=True,
    )

    # Get tokenizer
    tokenizer = get_tokenizer(model_name)

    # Set to eval mode
    model.eval()

    print(f"Model loaded successfully on {device}")

    return model, preprocess_train, preprocess_val, tokenizer


class MCQDataset(Dataset):
    """Dataset for Multiple Choice Question (MCQ) evaluation."""

    def __init__(self, csv_path: str, transform=None, negbench_root: str = "../negbench"):
        """
        Initialize MCQ dataset.

        Args:
            csv_path: Path to CSV file (relative to negbench or absolute)
            transform: Image preprocessing transform
            negbench_root: Root directory of negbench
        """
        self.negbench_root = Path(negbench_root)
        self.csv_path = self._resolve_path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.path_mapper = PathMapper(negbench_root=negbench_root)

        print(f"Loaded MCQ dataset: {self.csv_path}")
        print(f"  Total samples: {len(self.df)}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to negbench if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.negbench_root / path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get image path from CSV and map to actual location
        image_path = row["image_path"]
        full_image_path = self.path_mapper.map_path(image_path)

        # Load image
        image = Image.open(full_image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get captions (4 choices for MCQ)
        captions = [row[f"caption_{i}"] for i in range(4)]

        # Get correct answer
        correct_answer = int(row["correct_answer"])

        # Get answer template type if available
        answer_template = row.get("correct_answer_template", "unknown")

        return image, captions, correct_answer, answer_template


class RetrievalDataset(Dataset):
    """Dataset for image-text retrieval evaluation."""

    def __init__(self, csv_path: str, transform=None, negbench_root: str = "../negbench"):
        """
        Initialize retrieval dataset.

        Args:
            csv_path: Path to CSV file
            transform: Image preprocessing transform
            negbench_root: Root directory of negbench
        """
        self.negbench_root = Path(negbench_root)
        self.csv_path = self._resolve_path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.path_mapper = PathMapper(negbench_root=negbench_root)

        print(f"Loaded retrieval dataset: {self.csv_path}")
        print(f"  Total samples: {len(self.df)}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to negbench if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.negbench_root / path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get image path from CSV and map to actual location
        image_path = row["filepath"]
        full_image_path = self.path_mapper.map_path(image_path)

        # Load image
        image = Image.open(full_image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get captions (can be multiple captions per image)
        captions = eval(row["captions"]) if isinstance(row["captions"], str) else row["captions"]
        if not isinstance(captions, list):
            captions = [captions]

        return image, captions, idx


def collate_mcq(batch):
    """Custom collate function for MCQ dataloader."""
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]  # List of caption lists
    correct_answers = torch.tensor([item[2] for item in batch])
    answer_templates = [item[3] for item in batch]

    return images, captions, correct_answers, answer_templates


def collate_retrieval(batch):
    """Custom collate function for retrieval dataloader."""
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]  # List of caption lists
    indices = [item[2] for item in batch]

    return images, captions, indices


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    collate_fn=None,
):
    """
    Create a DataLoader.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of worker processes
        collate_fn: Custom collate function

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def save_results(results: dict, output_path: str):
    """
    Save evaluation results to file.

    Args:
        results: Dictionary of results
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def print_results(results: dict, title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.

    Args:
        results: Dictionary of results
        title: Title for the results
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Utility functions for negation decomposition evaluation")
    print("\nAvailable functions:")
    print("  - load_clip_model()")
    print("  - MCQDataset")
    print("  - RetrievalDataset")
    print("  - create_dataloader()")
    print("  - save_results()")
    print("  - print_results()")
