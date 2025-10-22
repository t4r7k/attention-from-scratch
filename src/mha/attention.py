
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: tensors shaped [batch_size, heads, seq_len, d_k]
            mask: optional boolean mask [batch_size, 1, 1, seq_len] where True=keep
        Returns:
            output: [batch_size, heads, seq_len, d_k]
            attn:   [batch_size, heads, seq_len, seq_len]
        """

        scores = torch.matmul(Q, K.transpose(-2, -1))

        d_k = Q.size(-1)
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learnable projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention()

    def _split_heads(self, x):
        """[b, L, d_model] -> [b, h, L, d_k]"""
        b, L, d_model = x.shape
        h, d_k = self.num_heads, self.d_k
        x = x.view(b, L, h, d_k).transpose(1, 2).contiguous()
        return x

    def forward(self, x, mask=None):
        """Self-attention over x.
        Args:
            x: [batch_size, seq_len, d_model]
            mask: optional boolean mask [batch_size, 1, 1, seq_len]
        Returns:
            out:  [batch_size, seq_len, d_model]
            attn: [batch_size, heads, seq_len, seq_len]
        """
        b, L, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        out_heads, attn = self.attn(Q, K, V, mask=mask)

        out = out_heads.transpose(1, 2).contiguous().view(b, L, d_model)
        out = self.W_o(out)

        return out, attn


def test_multihead_attention():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    try:
        out, attn = mha(x)
        print("Output shape:", out.shape)
        print("Attention shape:", attn.shape)
    except NotImplementedError as e:
        print("Implement the TODOs to run the test.")


def acceptance_tests():
    """
    Basic acceptance tests the candidate must pass after implementing TODOs.
    """
    torch.manual_seed(0)
    b, L, d_model, H = 3, 6, 64, 8
    x = torch.randn(b, L, d_model)
    mha = MultiHeadAttention(d_model, H)

    out, attn = mha(x)  # self-attention
    assert out.shape == (b, L, d_model), f"out shape was {out.shape}"
    assert attn.shape == (b, H, L, L), f"attn shape was {attn.shape}"
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "rows must sum to 1"

    loss = out.pow(2).mean()
    loss.backward()
    for name, p in mha.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"

    print("All acceptance tests passed.")


if __name__ == "__main__":
    test_multihead_attention()
    acceptance_tests()
