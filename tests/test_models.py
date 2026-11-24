import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbapred.models import MLP, SeqNet, TransNet  # noqa: E402

B, K, F_STEP, F_TAB = 16, 10, 7, 30


def batch():
    torch.manual_seed(0)
    return torch.randn(B, K, F_STEP), torch.randn(B, K, F_STEP), torch.randn(B, F_TAB)


def test_all_models_return_one_logit_per_game_and_backprop():
    sh, sa, x = batch()
    for model, args in ((MLP(F_TAB), (x,)), (SeqNet(F_STEP, F_TAB), (sh, sa, x)), (TransNet(F_STEP, F_TAB, k=K), (sh, sa, x))):
        out = model(*args)
        assert out.shape == (B,)
        out.sum().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_transformer_attention_is_a_distribution_over_the_k_games():
    sh, _, _ = batch()
    net = TransNet(F_STEP, F_TAB, k=K)
    att = net.cls_attention(sh)
    assert att.shape == (B, K)
    # the CLS row also attends to itself, so mass over the K games is <= 1
    assert (att.sum(1) <= 1 + 1e-5).all() and (att >= 0).all()


def test_transformer_uses_order_but_eval_is_deterministic():
    sh, sa, x = batch()
    net = TransNet(F_STEP, F_TAB, k=K).eval()
    a, b = net(sh, sa, x), net(sh, sa, x)
    assert torch.allclose(a, b)
    flipped = net(sh.flip(1), sa.flip(1), x)   # positional embeddings: reversing history changes the output
    assert not torch.allclose(a, flipped, atol=1e-6)
