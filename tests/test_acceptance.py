
import torch
from src.mha import MultiHeadAttention

def test_shapes_and_grads():
    torch.manual_seed(0)
    b, L, d_model, H = 3, 6, 64, 8
    x = torch.randn(b, L, d_model)
    mha = MultiHeadAttention(d_model, H)

    out, attn = mha(x)
    assert out.shape == (b, L, d_model)
    assert attn.shape == (b, H, L, L)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(b, H, L), atol=1e-5)

    loss = out.pow(2).mean()
    loss.backward()
    for _, p in mha.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
