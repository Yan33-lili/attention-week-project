import re

from src.utils.config import load_yaml_config
from src.utils.paths import repo_root


def test_clip_config_matches_expected_scope() -> None:
    config = load_yaml_config(repo_root() / "configs" / "clip.yaml")
    assert config["model_name"] == "openai/clip-vit-base-patch32"
    assert config["texts"] == ["a red square", "a blue square"]


def test_clip_similarity_artifact_has_expected_shapes() -> None:
    artifact = (repo_root() / "outputs" / "metrics" / "clip_similarity_matrix.md").read_text(encoding="utf-8")
    assert "image_embeds shape: (2, 512)" in artifact
    assert "text_embeds shape: (2, 512)" in artifact
    assert "logits_per_image:" in artifact
    assert "row-wise probabilities:" in artifact
    assert artifact.count("- a red square") == 1
    assert artifact.count("- a blue square") == 1
    assert re.search(r"logits_per_image:\s*tensor\(\[\[.*\],\s*\[.*\]\]\)", artifact, re.S)
