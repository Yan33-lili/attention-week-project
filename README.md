# Attention Week Project

## Project Overview

This repository collects the Week 3 Attention Week demos for:

- HuggingFace tokenizer basics
- Transformer tensor-shape flow
- CLIP-VLM image-text similarity

The repository is being prepared as a research-style project. This stage preserves only the demos and outputs that were actually completed.

## Project Structure

```text
attention-week-project/
├── README.md
├── REPORT.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
├── experiments/
│   └── week3/
├── tests/
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── metrics/
└── docs/
    └── debug_notes.md
```

## Environment

The original project was developed in a Windows + conda workflow. The current code uses `pathlib` and repository-relative paths so the same commands can run on Windows, Linux, or macOS.

- `torch`
- `transformers`
- `pillow`
- `pyyaml`

## Installation

```bash
pip install -r requirements.txt
```

## Available Demos

- `experiments/week3/tokenizer_demo.py`
- `experiments/week3/transformer_shape_demo.py`
- `experiments/week3/clip_vlm_demo.py`

## Basic Usage

Run the demos from the repository root:

```bash
python -m experiments.week3.tokenizer_demo --config configs/tokenizer.yaml
python -m experiments.week3.transformer_shape_demo --config configs/transformer.yaml
python -m experiments.week3.clip_vlm_demo --config configs/clip.yaml
```

The CLIP demo falls back to synthetic local images when `configs/clip.yaml` does not provide `image_inputs`.

## Configs

- `configs/tokenizer.yaml`: tokenizer model name and example texts
- `configs/transformer.yaml`: seed and Transformer shape parameters
- `configs/clip.yaml`: CLIP model name, texts, optional image inputs, and output path

## Outputs

Formal outputs are stored under `outputs/`:

- `outputs/figures/`
- `outputs/logs/`
- `outputs/metrics/`

Current preserved outputs:

- tokenizer output screenshot
- Transformer shape screenshot
- CLIP-VLM success screenshot
- tokenizer log text
- CLIP similarity matrix summary
- attention shape diagram

## Reproducibility Notes

- The completed scope is limited to tokenizer, Transformer shape, CLIP-VLM, and the attention-shape diagram.
- No new experimental results are introduced in this repository stage.
- Some future Attention Week experiments are intentionally not included yet.
- Local validation in the `vla` conda environment succeeded with the three commands above.

## Known Limitations

- Not all planned Attention Week experiments have been implemented.
- The CLIP demo uses a synthetic local-image fallback when image paths are not supplied.
- Full paper-style analysis and broader ablation work are not yet present.
