"""CLIP / VLM demo for image-text similarity.

The demo builds a 2x2 similarity matrix:
2 images x 2 text prompts.

If model or image downloads are slow, use local image files and/or a cached
HuggingFace environment in Windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from PIL import Image, ImageDraw
import torch
from transformers import CLIPModel, CLIPProcessor

from src.utils.config import load_yaml_config
from src.utils.paths import resolve_repo_path


def build_default_images() -> List[Image.Image]:
    # Using local PIL images removes the dependency on remote image URLs.
    # This keeps the demo runnable even when the network is restricted.
    images: List[Image.Image] = []
    specs = [
        ("red square", (220, 50, 50)),
        ("blue square", (50, 90, 220)),
    ]
    for label, color in specs:
        image = Image.new("RGB", (224, 224), color=color)
        draw = ImageDraw.Draw(image)
        draw.text((18, 96), label, fill="white")
        images.append(image)
    return images


def load_image(source: str) -> Image.Image:
    return Image.open(Path(source)).convert("RGB")


def load_images(sources: List[str]) -> List[Image.Image]:
    return [load_image(source) for source in sources]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CLIP image-text matching demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=resolve_repo_path("configs", "clip.yaml"),
        help="Path to CLIP config YAML.",
    )
    parser.add_argument("--model", help="Optional override for the CLIP model name.")
    parser.add_argument("--image", action="append", dest="images", help="Image path. Pass twice.")
    parser.add_argument("--text", action="append", dest="texts", help="Text prompt. Pass twice.")
    return parser.parse_args()


def _resolve_image_sources(config: dict[str, Any], override_images: List[str] | None) -> List[str] | None:
    if override_images is not None:
        return override_images
    sources = config.get("image_inputs")
    if not sources:
        return None
    return [str(source) for source in sources]


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_name = args.model or config["model_name"]
    texts = args.texts or config["texts"]
    image_sources = _resolve_image_sources(config, args.images)

    if image_sources is not None and len(image_sources) != 2:
        raise ValueError("This demo expects exactly 2 images when --image is used.")
    if len(texts) != 2:
        raise ValueError("This demo expects exactly 2 images and 2 texts.")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    images = load_images(image_sources) if image_sources is not None else build_default_images()

    # The processor converts pixels and text into model-ready tensors.
    # - image pixels become pixel_values for the image encoder
    # - tokenized text becomes input_ids and attention_mask for the text encoder
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        # image_embeds and text_embeds are the projected CLIP embeddings.
        # logits_per_image is the similarity matrix computed from them.
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_logits = outputs.logits_per_image
        probs = image_logits.softmax(dim=1)

    print(f"config_path: {args.config}")
    print(f"model_name: {model_name}")
    print(f"image_embeds shape: {tuple(image_embeds.shape)}")
    print(f"text_embeds shape: {tuple(text_embeds.shape)}")
    print("texts:")
    for idx, text in enumerate(texts):
        print(f"  text[{idx}]: {text}")

    print("\nimage_embeds:")
    print(image_embeds)
    print("\ntext_embeds:")
    print(text_embeds)

    print("\nlogits_per_image (2x2 similarity matrix):")
    print(image_logits)
    print("\nrow-wise probabilities:")
    print(probs)

    output_cfg = config.get("output", {})
    report_path_value = output_cfg.get("report_path", "outputs/metrics/clip_similarity_matrix.md")
    report_path = resolve_repo_path(report_path_value) if not Path(report_path_value).is_absolute() else Path(report_path_value)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# CLIP Similarity Matrix",
        "",
        f"model_name: {model_name}",
        f"image_embeds shape: {tuple(image_embeds.shape)}",
        f"text_embeds shape: {tuple(text_embeds.shape)}",
        "",
        "texts:",
    ]
    report_lines.extend([f"- {text}" for text in texts])
    report_lines.extend(
        [
            "",
            "image_embeds:",
            str(image_embeds),
            "",
            "text_embeds:",
            str(text_embeds),
            "",
            "logits_per_image:",
            str(image_logits),
            "",
            "row-wise probabilities:",
            str(probs),
            "",
            "Interpretation:",
            "- larger logits mean stronger image-text match",
            "- the diagonal should usually be higher if the pair is aligned",
            "- CLIP compares image encoder features with text encoder features",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nInterpretation:")
    print("- larger logits mean stronger image-text match")
    print("- the diagonal should usually be higher if the pair is aligned")
    print("- CLIP compares image encoder features with text encoder features")


if __name__ == "__main__":
    main()
