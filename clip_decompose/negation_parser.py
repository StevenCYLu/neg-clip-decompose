"""
Negation parser for extracting positive and negative concepts from text prompts.

This module handles various negation patterns like:
- "a dog and no grass"
- "a cat without a dog"
- "not a bird"
- "no people in the scene"
"""

import re
from typing import Tuple, List


class NegationParser:
    """Parse text prompts to extract positive and negative concepts."""

    # Negation patterns ordered by specificity (most specific first)
    NEGATION_PATTERNS = [
        # "X and no Y" or "X but no Y"
        (r'^(.*?)\s+(?:and|but)\s+no\s+(.+)$', 1, 2),

        # "X without Y"
        (r'^(.*?)\s+without\s+(.+)$', 1, 2),

        # "X and not Y" or "X but not Y"
        (r'^(.*?)\s+(?:and|but)\s+not\s+(.+)$', 1, 2),

        # "no Y in X" or "no Y on X"
        (r'^no\s+(.+?)\s+(?:in|on|near|with)\s+(.+)$', 2, 1),

        # "X with no Y"
        (r'^(.*?)\s+with\s+no\s+(.+)$', 1, 2),

        # Starting with "no X"
        (r'^no\s+(.+)$', None, 1),

        # Starting with "not X"
        (r'^not\s+(.+)$', None, 1),

        # "without X" at the beginning
        (r'^without\s+(.+)$', None, 1),
    ]

    def __init__(self, preserve_articles: bool = True):
        """
        Initialize the parser.

        Args:
            preserve_articles: If True, keep articles (a, an, the) in extracted concepts
        """
        self.preserve_articles = preserve_articles

    def parse(self, caption: str) -> Tuple[str, str]:
        """
        Parse a caption into positive and negative components.

        Args:
            caption: Input text prompt (e.g., "a dog and no grass")

        Returns:
            (positive_text, negative_text) tuple
            - If no negation found: (caption, "")
            - If only negation: ("", negative_concept)
            - If both: (positive_concept, negative_concept)

        Examples:
            >>> parser = NegationParser()
            >>> parser.parse("a dog and no grass")
            ("a dog", "grass")
            >>> parser.parse("a cat without a dog")
            ("a cat", "a dog")
            >>> parser.parse("no birds")
            ("", "birds")
        """
        caption = caption.strip()
        caption_lower = caption.lower()

        # Try each pattern in order
        for pattern, pos_group, neg_group in self.NEGATION_PATTERNS:
            match = re.match(pattern, caption_lower, re.IGNORECASE)
            if match:
                # Extract groups
                positive = match.group(pos_group).strip() if pos_group else ""
                negative = match.group(neg_group).strip() if neg_group else ""

                # Clean up the extracted text
                positive = self._clean_text(positive)
                negative = self._clean_text(negative)

                return positive, negative

        # No negation found - return original caption as positive
        return caption, ""

    def parse_batch(self, captions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Parse a batch of captions.

        Args:
            captions: List of text prompts

        Returns:
            (positive_texts, negative_texts) tuple of lists
        """
        positives = []
        negatives = []

        for caption in captions:
            pos, neg = self.parse(caption)
            positives.append(pos)
            negatives.append(neg)

        return positives, negatives

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        text = text.strip()

        # Remove trailing punctuation
        text = re.sub(r'[,;.!?]+$', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def has_negation(self, caption: str) -> bool:
        """
        Check if a caption contains negation.

        Args:
            caption: Input text prompt

        Returns:
            True if negation is detected, False otherwise
        """
        _, negative = self.parse(caption)
        return len(negative) > 0


def parse_negation(caption: str) -> Tuple[str, str]:
    """
    Convenience function to parse a single caption.

    Args:
        caption: Input text prompt

    Returns:
        (positive_text, negative_text) tuple
    """
    parser = NegationParser()
    return parser.parse(caption)


def parse_negation_batch(captions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Convenience function to parse a batch of captions.

    Args:
        captions: List of text prompts

    Returns:
        (positive_texts, negative_texts) tuple of lists
    """
    parser = NegationParser()
    return parser.parse_batch(captions)


if __name__ == "__main__":
    # Test the parser
    test_cases = [
        "a dog and no grass",
        "a cat without a dog",
        "not a bird",
        "no people in the scene",
        "a beach with no umbrellas",
        "a room but not a kitchen",
        "without any cars",
        "a simple image",  # No negation
    ]

    parser = NegationParser()
    print("Testing Negation Parser:")
    print("-" * 60)
    for caption in test_cases:
        pos, neg = parser.parse(caption)
        print(f"Input:    '{caption}'")
        print(f"Positive: '{pos}'")
        print(f"Negative: '{neg}'")
        print("-" * 60)
