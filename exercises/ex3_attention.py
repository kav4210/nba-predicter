"""Exercise 3: scaled dot-product attention.

Implement single-head attention in PyTorch:

    attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Q: (batch, n_queries, d_k), K: (batch, n_keys, d_k), V: (batch, n_keys, d_v).
Return (output, weights) with weights of shape (batch, n_queries, n_keys); each row of weights sums to 1.
If `mask` is given (bool, True = allowed, shape broadcastable to weights), disallowed positions
must get zero weight.
Do not call torch.nn.functional.scaled_dot_product_attention or nn.MultiheadAttention.
"""
import torch


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    raise NotImplementedError("implement me")
