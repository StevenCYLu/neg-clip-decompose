"""
Path configuration and mapping utilities.

This module handles mapping CSV image paths to actual image locations on disk.
"""

import os
from pathlib import Path
import re


class PathMapper:
    """Maps CSV image paths to actual filesystem paths."""

    def __init__(self, negbench_root=None):
        """
        Initialize path mapper.

        Args:
            negbench_root: Root directory of negbench (default: ../negbench)
        """
        self.negbench_root = Path(negbench_root or "../negbench")

    def map_path(self, csv_path: str) -> str:
        """
        Map a CSV image path to the actual filesystem path.

        Args:
            csv_path: Path from CSV file (e.g., "data/coco/images/val2017/000000397133.jpg")

        Returns:
            Actual filesystem path

        Examples:
            >>> mapper = PathMapper()
            >>> mapper.map_path("data/coco/images/val2017/000000397133.jpg")
            "../negbench/data/coco/images/val2017/000000397133.jpg"
        """
        csv_path = str(csv_path)

        # If it's already an absolute path, return it
        if Path(csv_path).is_absolute():
            return csv_path

        # Map relative paths to negbench data directory
        # CSV paths are like: data/coco/images/val2017/000000397133.jpg
        actual_path = self.negbench_root / csv_path

        return str(actual_path)


# Default global mapper
_default_mapper = None


def get_default_mapper() -> PathMapper:
    """Get the default global path mapper."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = PathMapper()
    return _default_mapper


def set_default_mapper(mapper: PathMapper):
    """Set the default global path mapper."""
    global _default_mapper
    _default_mapper = mapper


def map_image_path(csv_path: str) -> str:
    """
    Convenience function to map a CSV path using the default mapper.

    Args:
        csv_path: Path from CSV file

    Returns:
        Actual filesystem path
    """
    return get_default_mapper().map_path(csv_path)


if __name__ == "__main__":
    # Test the path mapper
    mapper = PathMapper()

    test_paths = [
        "data/coco/images/val2017/000000397133.jpg",
        "data/coco/images/val2017/000000037777.jpg",
    ]

    print("Testing Path Mapper:")
    print("=" * 60)

    for csv_path in test_paths:
        try:
            actual_path = mapper.map_path(csv_path)
            exists = Path(actual_path).exists()
            status = "✓ EXISTS" if exists else "✗ NOT FOUND"
            print(f"CSV:    {csv_path}")
            print(f"Actual: {actual_path}")
            print(f"Status: {status}")
            print("-" * 60)
        except Exception as e:
            print(f"CSV:    {csv_path}")
            print(f"Error:  {e}")
            print("-" * 60)
