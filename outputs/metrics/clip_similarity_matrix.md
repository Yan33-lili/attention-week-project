# CLIP Similarity Matrix

model_name: openai/clip-vit-base-patch32
image_embeds shape: (2, 512)
text_embeds shape: (2, 512)

texts:
- a red square
- a blue square

image_embeds:
tensor([[ 0.0156,  0.0300,  0.0215,  ..., -0.0075, -0.0509,  0.0068],
        [-0.0226,  0.0389,  0.0202,  ..., -0.0235, -0.0559, -0.0017]])

text_embeds:
tensor([[-0.0126,  0.0297, -0.0109,  ..., -0.0121, -0.0453,  0.0392],
        [ 0.0073, -0.0163,  0.0042,  ..., -0.0185, -0.0452,  0.0327]])

logits_per_image:
tensor([[32.8426, 26.5702],
        [28.8826, 33.9979]])

row-wise probabilities:
tensor([[0.9981, 0.0019],
        [0.0060, 0.9940]])

Interpretation:
- larger logits mean stronger image-text match
- the diagonal should usually be higher if the pair is aligned
- CLIP compares image encoder features with text encoder features