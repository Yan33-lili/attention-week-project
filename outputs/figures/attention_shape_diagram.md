# Attention Shape Diagram

```text
input tokens
    |
    v
token embedding
    |
    v
[batch, tokens, dim]
    |
    +--> linear projection to Q/K/V
    |         |
    |         +--> Q shape: [batch, heads, tokens, head_dim]
    |         +--> K shape: [batch, heads, tokens, head_dim]
    |         +--> V shape: [batch, heads, tokens, head_dim]
    |
    +--> attention score = Q @ K^T
    |         |
    |         v
    |   [batch, heads, tokens, tokens]
    |
    +--> weighted sum of V
    |         |
    |         v
    |   attention output shape: [batch, tokens, dim]
    |
    v
feed-forward network
    |
    v
[batch, tokens, dim]
```

Key idea:
- input shape starts at `[batch, tokens]`
- embedding shape becomes `[batch, tokens, dim]`
- Q/K/V are projected per head
- attention score shape is `[batch, heads, tokens, tokens]`
- output shape returns to `[batch, tokens, dim]`
