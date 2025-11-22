# CLIP Negation Decomposition

Evaluation script for testing CLIP's handling of negated prompts using **negation decomposition**.

## What is `eval_simple.py`?

`eval_simple.py` evaluates CLIP on 30 Multiple Choice Questions (10 negative, 10 positive, 10 hybrid) with configurable negation strength.

**How it works:**
1. Parses prompts into positive and negative concepts (e.g., "a dog and no grass" → positive: "dog", negative: "grass")
2. Encodes them separately with CLIP
3. Combines via subtraction: `final_embedding = positive_embedding - α × negative_embedding`
4. Re-normalizes for cosine similarity

The `alpha` parameter controls negation strength (0.0 = baseline/no negation, 1.0 = full subtraction).

## Installation

Install required dependencies:
```bash
pip install torch torchvision pillow pandas tqdm open_clip_torch
```

## How to Run

Run the evaluation with different alpha values:

```bash
# Baseline (no negation decomposition)
python eval_simple.py --alpha 0.0

# Weak negation
python eval_simple.py --alpha 0.3

# Medium negation (default)
python eval_simple.py --alpha 0.5

# Strong negation
python eval_simple.py --alpha 0.7

# Full negation
python eval_simple.py --alpha 1.0
```

The script will output accuracy metrics broken down by question type (positive, negative, hybrid).

## Dataset

The evaluation uses 30 samples from `eval_simple_data/mcq_simple.csv`:
- 10 negative templates (e.g., "a dog and no grass")
- 10 positive templates (e.g., "a dog and grass")
- 10 hybrid templates (mix of positive and negative)

Sample images are included in `eval_simple_data/images/`.

## Output

The script displays:
- **Overall accuracy**: Accuracy across all 30 samples
- **Accuracy by template type**: Breakdown for positive, negative, and hybrid questions
- **Model configuration**: CLIP model (ViT-B-32), alpha value, device used
