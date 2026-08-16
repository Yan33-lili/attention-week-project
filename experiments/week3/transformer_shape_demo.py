"""Transformer shape demo.

This script demonstrates the core tensor shape flow:
[batch, tokens, dim] -> attention -> feed-forward -> [batch, tokens, dim]
"""

from __future__ import annotations

import torch
from torch import nn

from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.paths import resolve_repo_path


def describe_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run the Transformer shape demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=resolve_repo_path("configs", "transformer.yaml"),
        help="Path to transformer config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    torch.manual_seed(config["seed"])

    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    dim = config["dim"]
    num_heads = config["num_heads"]
    ff_dim = config["ff_dim"]

    # In a real Transformer, text tokens or image patches are first mapped
    # into vectors with the same hidden dimension.
    x = torch.randn(batch_size, seq_len, dim)
    output_cfg = config.get("output", {})
    log_path_value = output_cfg.get("log_path", "outputs/logs/transformer_shape_output.txt")
    log_path = resolve_repo_path(log_path_value) if not Path(log_path_value).is_absolute() else Path(log_path_value)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"config_path: {args.config}"]
    lines.append(f"input: shape={tuple(x.shape)}")

    # MultiheadAttention expects [batch, tokens, dim] when batch_first=True.
    attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    attn_out, attn_weights = attn(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    lines.append(f"attention_output: shape={tuple(attn_out.shape)}")
    lines.append(f"attention_weights: shape={tuple(attn_weights.shape)}")

    # A small feed-forward block keeps the same token count but transforms features.
    ff = nn.Sequential(
        nn.Linear(dim, ff_dim),
        nn.ReLU(),
        nn.Linear(ff_dim, dim),
    )
    ff_out = ff(attn_out)
    lines.append(f"feed_forward_output: shape={tuple(ff_out.shape)}")

    # A Transformer encoder layer ties attention, residual connection,
    # layer norm, and feed-forward network together.
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    encoder_out = encoder(x)
    lines.append(f"encoder_output: shape={tuple(encoder_out.shape)}")
    lines.extend(
        [
            "",
            "Shape summary:",
            "[batch, tokens, dim] -> self-attention -> [batch, tokens, dim]",
            "attention_weights shape = [batch, heads, tokens, tokens]",
            "positional encoding is usually added before attention",
        ]
    )

    print("\n".join(lines))
    log_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
