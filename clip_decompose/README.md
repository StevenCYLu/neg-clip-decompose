# CLIP Negation Decomposition

This directory contains an implementation of **negation decomposition** for CLIP-based image-text retrieval and understanding.

## What is Negation Decomposition?

Negation decomposition is a technique to improve CLIP's handling of negated prompts (e.g., "a dog and no grass", "a cat without a dog").

### The Problem

CLIP's text encoder treats text prompts as holistic embeddings and doesn't understand logical negation. When given "a dog and no grass", CLIP creates a single embedding that mixes both concepts, often failing to properly avoid images with grass.

### The Solution

Instead of relying on CLIP's standard encoding, we:

1. **Parse** the prompt into positive and negative concepts
   - Positive: "a dog"
   - Negative: "grass"

2. **Encode separately** with CLIP
   - `positive_embedding = encode("a dog")`
   - `negative_embedding = encode("grass")`

3. **Combine** using subtraction
   - `final_embedding = positive_embedding - α × negative_embedding`
   - The parameter `α` controls the strength of negation

4. **Re-normalize** for cosine similarity
   - `final_embedding = normalize(final_embedding)`

This creates an embedding that is attracted to images with dogs and repelled from images with grass.

## Directory Structure

```
clip_decompose/
├── README.md                   # This file
├── negation_parser.py          # Parse negations from text
├── negation_encoder.py         # Encode with decomposition
├── utils.py                    # Data loading utilities
├── eval_mcq.py                 # MCQ evaluation script
├── eval_retrieval.py           # Retrieval evaluation script
├── run_evaluation.sh           # Convenience script
└── results/                    # Results directory (created at runtime)
```

## Installation

No additional installation required! This implementation uses the CLIP model and utilities from the `../negbench` directory.

### Requirements

- PyTorch
- PIL (Pillow)
- pandas
- tqdm
- numpy

These should already be installed if you can run negbench.

## Usage

### Quick Start

Run all evaluations with the convenience script:

```bash
cd clip_decompose
./run_evaluation.sh
```

### Individual Evaluations

#### 1. MCQ Evaluation

Evaluate on Multiple Choice Questions:

```bash
# Baseline (no decomposition)
python eval_mcq.py \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset "COCO_val_mcq_llama3.1_rephrased.csv" \
    --batch-size 64

# With negation decomposition
python eval_mcq.py \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset "COCO_val_mcq_llama3.1_rephrased.csv" \
    --use-decomposition \
    --alpha 0.5 \
    --batch-size 64
```

#### 2. Retrieval Evaluation

Evaluate on image-text retrieval:

```bash
# Baseline
python eval_retrieval.py \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv" \
    --recall-k 1 5 10 \
    --batch-size 64

# With negation decomposition
python eval_retrieval.py \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv" \
    --use-decomposition \
    --alpha 0.5 \
    --recall-k 1 5 10 \
    --batch-size 64
```

### Command-Line Arguments

#### Common Arguments

- `--model`: CLIP model architecture (default: `ViT-B-32`)
- `--pretrained`: Pretrained weights (default: `openai`)
- `--dataset`: Path to CSV file (relative to negbench or absolute)
- `--negbench-root`: Root directory of negbench (default: `../negbench`)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--device`: Device to use (`cuda` or `cpu`)

#### Decomposition Arguments

- `--use-decomposition`: Enable negation decomposition (flag)
- `--alpha`: Weight for negative embedding (default: 0.5)

#### MCQ-Specific Arguments

- `--output`: Path to save results JSON

#### Retrieval-Specific Arguments

- `--recall-k`: K values for Recall@K metric (default: [1, 5, 10])
- `--output`: Path to save results JSON

### Experimenting with Alpha

The `alpha` parameter controls how strongly negative concepts are avoided. Try different values:

```bash
# Weak negation
python eval_mcq.py --dataset "..." --use-decomposition --alpha 0.3

# Medium negation (default)
python eval_mcq.py --dataset "..." --use-decomposition --alpha 0.5

# Strong negation
python eval_mcq.py --dataset "..." --use-decomposition --alpha 0.7

# Very strong negation
python eval_mcq.py --dataset "..." --use-decomposition --alpha 1.0
```

## Negation Patterns Supported

The parser handles various negation patterns:

- `"X and no Y"` → positive: X, negative: Y
- `"X without Y"` → positive: X, negative: Y
- `"X but not Y"` → positive: X, negative: Y
- `"no Y in X"` → positive: X, negative: Y
- `"X with no Y"` → positive: X, negative: Y
- `"no X"` → positive: "", negative: X
- `"not X"` → positive: "", negative: X

Examples:
- "a dog and no grass" → ("a dog", "grass")
- "a cat without a dog" → ("a cat", "a dog")
- "no birds in the scene" → ("the scene", "birds")
- "not a car" → ("", "a car")

## Testing the Parser

Test the negation parser:

```bash
python negation_parser.py
```

This will run test cases and show how different prompts are parsed.

## Available Datasets

The following negbench datasets can be used:

### MCQ Datasets
- `COCO_val_mcq_llama3.1_rephrased.csv`
- `VOC2007_mcq_llama3.1_rephrased.csv`
- `synthetic_mcq_llama3.1_rephrased.csv`

### Retrieval Datasets
- `COCO_val_retrieval.csv`
- `COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv`

All paths are relative to the negbench root directory.

## Results

Results are saved in JSON format and include:

### MCQ Results
- `overall_accuracy`: Overall accuracy across all samples
- `accuracy_positive`: Accuracy on positive templates
- `accuracy_negative`: Accuracy on negative templates
- `accuracy_hybrid`: Accuracy on hybrid templates
- `total_samples`: Total number of samples
- `correct_predictions`: Number of correct predictions

### Retrieval Results
- `recall@1`: Recall at 1
- `recall@5`: Recall at 5
- `recall@10`: Recall at 10
- `num_images`: Total number of images
- `num_texts`: Total number of text queries

## Expected Improvements

Negation decomposition typically shows improvements on:

1. **Negative templates**: Questions where the correct answer contains negation
2. **Negated retrieval**: Retrieving images based on negated queries
3. **Compositional understanding**: Better handling of complex prompts

The improvement is most noticeable when:
- Prompts contain clear negation patterns
- The task requires avoiding specific concepts
- Standard CLIP struggles with compositional reasoning

## Implementation Details

### Re-normalization

After computing `positive_emb - α × negative_emb`, we re-normalize the result. This ensures:
- Unit-length vectors for fair cosine similarity comparison
- Stable similarity scores across different alpha values

### Batch Processing

The implementation supports efficient batch processing:
- Images are batched normally
- Text encoding handles variable numbers of captions per image
- GPU memory is managed efficiently

### Compatibility

This implementation:
- Uses negbench's CLIP model utilities
- Does not modify the negbench directory
- Loads data from negbench datasets
- Works with any CLIP model supported by negbench

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. The negbench directory is at `../negbench` relative to clip_decompose
2. You're running scripts from the clip_decompose directory
3. The negbench environment is activated

### Path Errors

If datasets are not found:
1. Check the `--negbench-root` argument
2. Verify dataset paths in the negbench directory
3. Use absolute paths if needed

### CUDA Out of Memory

If you run out of GPU memory:
1. Reduce `--batch-size`
2. Reduce `--num-workers`
3. Use a smaller model (e.g., ViT-B-32 instead of ViT-L-14)

## Citation

If you use this implementation, please cite the relevant papers on:
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
- Negation in vision-language models
- The negbench benchmark

## License

This implementation follows the same license as the negbench project.
