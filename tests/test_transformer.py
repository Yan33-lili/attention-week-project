from src.utils.config import load_yaml_config
from src.utils.paths import repo_root


def test_transformer_config_matches_expected_scope() -> None:
    config = load_yaml_config(repo_root() / "configs" / "transformer.yaml")
    assert config["seed"] == 42
    assert config["batch_size"] == 2
    assert config["seq_len"] == 5
    assert config["dim"] == 32
    assert config["num_heads"] == 4
    assert config["ff_dim"] == 64


def test_transformer_shape_contract() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from torch import nn

    config = load_yaml_config(repo_root() / "configs" / "transformer.yaml")
    torch.manual_seed(config["seed"])
    x = torch.randn(config["batch_size"], config["seq_len"], config["dim"])
    attn = nn.MultiheadAttention(embed_dim=config["dim"], num_heads=config["num_heads"], batch_first=True)
    attn_out, attn_weights = attn(x, x, x, need_weights=True, average_attn_weights=False)
    ff = nn.Sequential(
        nn.Linear(config["dim"], config["ff_dim"]),
        nn.ReLU(),
        nn.Linear(config["ff_dim"], config["dim"]),
    )
    ff_out = ff(attn_out)
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=config["dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["ff_dim"],
            batch_first=True,
        ),
        num_layers=1,
    )
    encoder_out = encoder(x)

    assert attn_out.shape == (config["batch_size"], config["seq_len"], config["dim"])
    assert attn_weights.shape == (config["batch_size"], config["num_heads"], config["seq_len"], config["seq_len"])
    assert ff_out.shape == (config["batch_size"], config["seq_len"], config["dim"])
    assert encoder_out.shape == (config["batch_size"], config["seq_len"], config["dim"])
