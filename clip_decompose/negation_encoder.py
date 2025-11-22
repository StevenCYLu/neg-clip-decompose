"""
Negation decomposition encoder for CLIP.

This module implements the core negation decomposition logic:
1. Parse prompts into positive and negative concepts
2. Encode each separately with CLIP
3. Combine: final_embedding = positive_embedding - alpha * negative_embedding
4. Re-normalize the result
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from typing import List, Union, Tuple

# Add negbench to path to use their CLIP implementation
NEGBENCH_PATH = Path(__file__).parent.parent / "negbench" / "benchmarks" / "src"
sys.path.insert(0, str(NEGBENCH_PATH))

from negation_parser import NegationParser


class NegationDecompositionEncoder:
    """Encode text with negation decomposition."""

    def __init__(self, model, tokenizer, device: str = "cuda", alpha: float = 0.5):
        """
        Initialize the encoder.

        Args:
            model: CLIP model with encode_text() method
            tokenizer: Text tokenizer
            device: Device to use ("cuda" or "cpu")
            alpha: Weight for negative embedding subtraction (default: 0.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.alpha = alpha
        self.parser = NegationParser()

    def encode_text(
        self,
        captions: Union[str, List[str]],
        normalize: bool = True,
        use_decomposition: bool = True,
    ) -> torch.Tensor:
        """
        Encode text with optional negation decomposition.

        Args:
            captions: Single caption or list of captions
            normalize: If True, L2 normalize the output embeddings
            use_decomposition: If True, use negation decomposition;
                             if False, use standard CLIP encoding

        Returns:
            Text embeddings of shape (batch_size, embedding_dim)
        """
        # Handle single caption
        if isinstance(captions, str):
            captions = [captions]

        if not use_decomposition:
            # Standard CLIP encoding
            return self._encode_standard(captions, normalize)

        # Negation decomposition encoding
        return self._encode_with_decomposition(captions, normalize)

    def _encode_standard(self, captions: List[str], normalize: bool) -> torch.Tensor:
        """
        Standard CLIP text encoding (baseline).

        Args:
            captions: List of text prompts
            normalize: If True, L2 normalize the embeddings

        Returns:
            Text embeddings
        """
        tokens = self.tokenizer(captions).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_text(tokens, normalize=False)

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def _encode_with_decomposition(
        self, captions: List[str], normalize: bool
    ) -> torch.Tensor:
        """
        Encode text with negation decomposition.

        For each caption:
        1. Parse into positive and negative concepts
        2. Encode separately
        3. Combine: final = positive - alpha * negative
        4. Re-normalize

        Args:
            captions: List of text prompts
            normalize: If True, L2 normalize the final embeddings

        Returns:
            Composed text embeddings
        """
        # Parse all captions
        positive_texts, negative_texts = self.parser.parse_batch(captions)

        # Handle cases where there's no positive or negative text
        # For empty positive: use empty string (will get close to zero embedding)
        # For empty negative: no subtraction needed
        positive_texts = [p if p else "" for p in positive_texts]
        negative_texts = [n if n else "" for n in negative_texts]

        # Tokenize
        positive_tokens = self.tokenizer(positive_texts).to(self.device)
        negative_tokens = self.tokenizer(negative_texts).to(self.device)

        # Encode with CLIP
        with torch.no_grad():
            positive_emb = self.model.encode_text(positive_tokens, normalize=False)
            negative_emb = self.model.encode_text(negative_tokens, normalize=False)

        # Normalize before combination (important for stable subtraction)
        positive_emb = F.normalize(positive_emb, dim=-1)
        negative_emb = F.normalize(negative_emb, dim=-1)

        # Combine: final = positive - alpha * negative
        final_emb = positive_emb - self.alpha * negative_emb

        # Re-normalize the result
        if normalize:
            final_emb = F.normalize(final_emb, dim=-1)

        return final_emb

    def set_alpha(self, alpha: float):
        """
        Update the alpha parameter.

        Args:
            alpha: New alpha value for negative weight
        """
        self.alpha = alpha

    def encode_text_batch(
        self,
        captions: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        use_decomposition: bool = True,
    ) -> torch.Tensor:
        """
        Encode text in batches (useful for large datasets).

        Args:
            captions: List of text prompts
            batch_size: Number of captions to process at once
            normalize: If True, L2 normalize the embeddings
            use_decomposition: If True, use negation decomposition

        Returns:
            Text embeddings for all captions
        """
        all_embeddings = []

        for i in range(0, len(captions), batch_size):
            batch = captions[i : i + batch_size]
            embeddings = self.encode_text(batch, normalize, use_decomposition)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


def create_negation_encoder(
    model, tokenizer, device: str = "cuda", alpha: float = 0.5
) -> NegationDecompositionEncoder:
    """
    Convenience function to create a negation decomposition encoder.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        device: Device to use
        alpha: Weight for negative embedding

    Returns:
        NegationDecompositionEncoder instance
    """
    return NegationDecompositionEncoder(model, tokenizer, device, alpha)


if __name__ == "__main__":
    print("Negation Decomposition Encoder")
    print("=" * 60)
    print("\nThis module requires a CLIP model to run tests.")
    print("Use eval_mcq.py or eval_retrieval.py to test the full pipeline.")
    print("\nExample usage:")
    print("  encoder = NegationDecompositionEncoder(model, tokenizer, alpha=0.5)")
    print("  embeddings = encoder.encode_text(['a dog and no grass'])")
