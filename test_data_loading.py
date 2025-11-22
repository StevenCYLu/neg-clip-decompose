"""
Test script to verify data loading works correctly.
"""

from utils import MCQDataset, RetrievalDataset
from pathlib import Path

def test_mcq_loading():
    """Test MCQ dataset loading."""
    print("=" * 60)
    print("Testing MCQ Dataset Loading")
    print("=" * 60)

    try:
        dataset = MCQDataset(
            csv_path="data/images/COCO_val_mcq_llama3.1_rephrased.csv",
            transform=None,  # No transform for quick test
            negbench_root="../negbench",
        )

        # Try loading first sample
        print(f"\nAttempting to load first sample...")
        image, captions, correct_answer, answer_template = dataset[0]

        print(f"✓ Successfully loaded image: {image.size}")
        print(f"✓ Number of captions: {len(captions)}")
        print(f"✓ Correct answer index: {correct_answer}")
        print(f"✓ Answer template: {answer_template}")
        print(f"\nSample captions:")
        for i, caption in enumerate(captions):
            print(f"  {i}: {caption[:80]}...")

        print("\n✓ MCQ dataset loading works!")
        return True

    except Exception as e:
        print(f"\n✗ Error loading MCQ dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_loading():
    """Test retrieval dataset loading."""
    print("\n" + "=" * 60)
    print("Testing Retrieval Dataset Loading")
    print("=" * 60)

    try:
        dataset = RetrievalDataset(
            csv_path="data/images/COCO_val_retrieval.csv",
            transform=None,  # No transform for quick test
            negbench_root="../negbench",
        )

        # Try loading first sample
        print(f"\nAttempting to load first sample...")
        image, captions, idx = dataset[0]

        print(f"✓ Successfully loaded image: {image.size}")
        print(f"✓ Number of captions: {len(captions)}")
        print(f"✓ Image index: {idx}")
        print(f"\nSample captions:")
        for i, caption in enumerate(captions[:3]):  # Show first 3
            print(f"  {i}: {caption[:80]}...")

        print("\n✓ Retrieval dataset loading works!")
        return True

    except Exception as e:
        print(f"\n✗ Error loading retrieval dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nTesting Data Loading for Negation Decomposition")
    print("=" * 60)

    mcq_ok = test_mcq_loading()
    retrieval_ok = test_retrieval_loading()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"MCQ Dataset:       {'✓ PASS' if mcq_ok else '✗ FAIL'}")
    print(f"Retrieval Dataset: {'✓ PASS' if retrieval_ok else '✗ FAIL'}")

    if mcq_ok and retrieval_ok:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run MCQ evaluation: python eval_mcq.py --dataset 'data/images/COCO_val_mcq_llama3.1_rephrased.csv' --use-decomposition --alpha 0.5")
        print("  2. Run retrieval evaluation: python eval_retrieval.py --dataset 'data/images/COCO_val_retrieval.csv' --use-decomposition --alpha 0.5")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

    print("=" * 60)
