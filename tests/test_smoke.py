from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.paths import repo_root


def test_repository_layout_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "tokenizer.yaml").exists()
    assert (root / "configs" / "transformer.yaml").exists()
    assert (root / "configs" / "clip.yaml").exists()
    assert (root / "experiments" / "week3").exists()
    assert (root / "outputs" / "figures").exists()
    assert (root / "outputs" / "logs").exists()
    assert (root / "outputs" / "metrics").exists()


def test_config_loader_reads_yaml() -> None:
    config = load_yaml_config(repo_root() / "configs" / "tokenizer.yaml")
    assert config["model_name"] == "distilbert-base-uncased"
    assert len(config["texts"]) == 2
