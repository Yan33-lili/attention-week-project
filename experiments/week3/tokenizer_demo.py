"""Tokenizer demo for HuggingFace tokenization basics.

This script shows:
- how a tokenizer splits text into tokens,
- how token ids are produced,
- what attention_mask means,
- how special tokens are added.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from transformers import AutoTokenizer

from src.utils.config import load_yaml_config
from src.utils.paths import resolve_repo_path


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run the HuggingFace tokenizer demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=resolve_repo_path("configs", "tokenizer.yaml"),
        help="Path to tokenizer config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    # Tokenizer is the text front-end of a HuggingFace model.
    # It turns raw text into model-ready numbers.
    model_name = config["model_name"]
    texts = config["texts"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # padding/truncation make a batch align to the same sequence length.
    # return_tensors="pt" converts lists into PyTorch tensors.
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    output_cfg = config.get("output", {})
    log_path_value = output_cfg.get("log_path", "outputs/logs/tokenizer_output.txt")
    log_path = resolve_repo_path(log_path_value) if not Path(log_path_value).is_absolute() else Path(log_path_value)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"config_path: {args.config}",
        f"model_name: {model_name}",
        "input_ids:",
        str(encoded["input_ids"]),
        "attention_mask:",
        str(encoded["attention_mask"]),
    ]

    # Convert ids back to human-readable tokens for inspection.
    for idx, text in enumerate(texts):
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][idx])
        lines.extend(
            [
                "",
                f"text[{idx}]: {text}",
                f"tokens[{idx}]: {tokens}",
                f"decoded[{idx}]: {tokenizer.decode(encoded['input_ids'][idx])}",
            ]
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- token id: the integer index used by the model vocabulary",
            "- attention_mask: 1 means real token, 0 means padding token",
            "- special tokens: tokenizer may add [CLS], [SEP], <s>, </s>, etc.",
        ]
    )

    print("\n".join(lines))
    log_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
