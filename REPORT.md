# Abstract

This project summarizes the completed Week 3 attention-related demos: HuggingFace tokenization, Transformer tensor-shape flow, CLIP-VLM image-text similarity, and the attention-shape diagram.

# Questions

- How are text tokens converted into model inputs?
- How do shapes move through a Transformer block?
- How does CLIP produce an image-text similarity matrix?

# Hypotheses

- Tokenization should expose `input_ids` and `attention_mask` clearly.
- Transformer self-attention should preserve the `[batch, tokens, dim]` shape.
- CLIP should produce higher similarity scores for aligned image-text pairs.

# Method

The project uses three standalone demos and one explanatory diagram:

- HuggingFace tokenizer demo
- Transformer shape demo
- CLIP-VLM demo
- attention shape diagram

# Experiments

- Tokenizer inspection with example sentences
- Transformer tensor-shape inspection with a small synthetic input
- CLIP image-text matching with a 2x2 setup
- attention-shape diagram summary

# Results

Preserved outputs are stored in `outputs/`.

Observed local results in the `vla` conda environment:

- Tokenizer: `distilbert-base-uncased`, valid `input_ids`, valid `attention_mask`, valid tokens and decoded text, with only a HuggingFace Hub unauthenticated warning.
- Transformer: `input (2, 5, 32)`, `attention_output (2, 5, 32)`, `attention_weights (2, 4, 5, 5)`, `feed_forward_output (2, 5, 32)`, `encoder_output (2, 5, 32)`.
- CLIP-VLM: `openai/clip-vit-base-patch32`, `image_embeds (2, 512)`, `text_embeds (2, 512)`, similarity matrix `[[32.8426, 26.5702], [28.8826, 33.9979]]`, row-wise probabilities `[[0.9981, 0.0019], [0.0060, 0.9940]]`.
- The diagonal image-text pairs were the stronger matches.

The preserved artifacts remain:

- tokenizer screenshot
- Transformer shape screenshot
- CLIP-VLM success screenshot
- tokenizer log text
- CLIP similarity matrix summary
- attention shape diagram

# Discussion

The current stage focuses on repository structure, parameterization, and reproducibility metadata. It preserves the original learning artifacts without adding new conclusions or unverified claims beyond the verified local run outputs above.

# Limitations

- The repository covers only the experiments that were actually completed.
- Broader Attention Week items such as ViT, RoPE, Window Attention, symmetry, CNN, and LSTM remain out of scope.
- No dataset-scale evaluation or training history is present.
- The CLIP demo uses a synthetic local-image fallback when image paths are not supplied.

# Conclusion

The repository is now organized as a research-style archive for the completed Week 3 demos and is ready for future expansion.

# References

- HuggingFace Transformers documentation
- PyTorch documentation
- OpenAI CLIP model documentation
- Original course materials and in-repo debug notes

## Implementation Notes

The demos use official HuggingFace and PyTorch APIs. No third-party code was copied into this repository beyond standard package usage.
