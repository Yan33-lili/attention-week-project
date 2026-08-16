from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.paths import repo_root


def test_tokenizer_config_matches_expected_scope() -> None:
    config = load_yaml_config(repo_root() / "configs" / "tokenizer.yaml")
    assert config["model_name"] == "distilbert-base-uncased"
    assert config["texts"] == [
        "I love myself",
        "Transformers use attention to connect tokens",
    ]


def test_tokenizer_output_artifact_contains_expected_fields() -> None:
    output = (repo_root() / "outputs" / "logs" / "tokenizer_output.txt").read_text(encoding="utf-8")
    assert "model_name: distilbert-base-uncased" in output
    assert "attention_mask:" in output
    assert "text[0]: I love myself" in output
    assert "text[1]: Transformers use attention to connect tokens" in output
