# Simple MCQ Evaluation Dataset

This directory contains a fixed dataset for quick evaluation of CLIP with negation decomposition.

## Contents

- `mcq_simple.csv`: 30 MCQ samples (10 negative, 10 positive, 10 hybrid)
- `images/`: 17 COCO validation images used by the MCQ samples

## Dataset Statistics

- **Total samples**: 30
- **Unique images**: 17
- **Template distribution**:
  - Negative: 10 samples
  - Positive: 10 samples
  - Hybrid: 10 samples

## Image Format

All images are COCO val2017 images in JPG format, copied locally for self-contained evaluation.

## CSV Structure

The CSV file has the following columns:
- `image_path`: Relative path to image (e.g., "images/000000397133.jpg")
- `correct_answer`: Index of correct caption (0-3)
- `caption_0` to `caption_3`: Four multiple choice captions
- `correct_answer_template`: Template type ("negative", "positive", or "hybrid")

## Usage

This dataset is used by `eval_simple.py` for quick evaluation:

```bash
cd /scratch/gautschi/lu842/composition_clip/clip_decompose
python eval_simple.py --alpha 0.5
```

## Source

Data extracted from: `negbench/data/images/COCO_val_mcq_llama3.1_rephrased.csv`
