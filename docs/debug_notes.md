# Debug Notes

## Environment

- Original workspace: Windows project directory for the Week 3 attention demos
- Original execution context: Windows + VSCode
- Original runtime target: Windows conda environment used during the course work

## Observations

- Tokenizer demo worked with a HuggingFace tokenizer and saved output text plus a screenshot.
- Transformer demo worked and captured the shape flow through attention and feed-forward layers.
- CLIP-VLM demo initially failed on a remote image URL, then was adjusted to use local synthetic images.

## Preserved Debug Artifacts

- `outputs/figures/attention_shape_diagram.md`
- `outputs/logs/tokenizer_output.txt`
- `outputs/metrics/clip_similarity_matrix.md`

## Notes for Next Phase

- Replace hardcoded paths with `pathlib` and repository-relative paths.
- Introduce config loading for demo parameters.
- Add a real smoke-test pipeline once the code is refactored into reusable modules.
