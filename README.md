# CLIP Negation Decomposition

Evaluation script for testing CLIP's handling of negated prompts using **negation decomposition**.

`eval_simple.py` evaluates CLIP on 30 Multiple Choice Questions (10 negative, 10 positive, 10 hybrid) with configurable negation strength.

**How it works:**

1.  Parses prompts into positive and negative concepts (e.g., "a dog and no grass" → positive: "dog", negative: "grass")
2.  Encodes them separately with CLIP
3.  Combines via subtraction: `final_embedding = positive_embedding - α × negative_embedding`
4.  Re-normalizes for cosine similarity

The `alpha` parameter controls negation strength (0.0 = baseline/no negation, 1.0 = full subtraction).

## Why it Works: The Energy-Based Perspective

Combining embeddings via subtraction is not just a heuristic; it creates a composite energy function that penalizes unwanted features.

In Energy-Based Models (EBMs), the compatibility between an image $I$ and a text description $T$ is often defined by an energy function $E(I, T)$. In CLIP, the dot product of the normalized embeddings serves as a negative energy (or a similarity score):

$$S(I, T) = -E(I, T) = \vec{v}_I \cdot \vec{v}_T$$

Where a higher score indicates higher compatibility.

When we decompose a prompt like "A dog and no grass" into a positive component ($\vec{v}\_{pos}$) and a negative component ($\vec{v}\_{neg}$), we aim to find an image that has **high compatibility with the positive** and **low compatibility with the negative**.

Mathematically, we construct a composite score:

$$S_{composite}(I) = S(I, \text{Pos}) - \alpha \cdot S(I, \text{Neg})$$

Due to the **linearity of the dot product**, this operation in score space is equivalent to vector subtraction in the embedding space:

$$\vec{v}_I \cdot \vec{v}_{pos} - \alpha (\vec{v}_I \cdot \vec{v}_{neg}) = \vec{v}_I \cdot (\vec{v}_{pos} - \alpha \vec{v}_{neg})$$

**Key Takeaway:**
By subtracting the negative text embedding from the positive one, we effectively construct a new energy landscape where images containing the negative concept result in a lower similarity score (higher energy), pushing the model to prefer images that match the positive description while strictly avoiding the negative one.

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

# With negation
python eval_simple.py --alpha 1.5
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
