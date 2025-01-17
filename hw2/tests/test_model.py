import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from lm.model import DecoderLM, MultiHeadAttention

from pytest_utils.decorators import max_score

b, s, d, h, hd = 16, 128, 768, 12, 64


@max_score(5)
def test_q_kT_v():
    mha = MultiHeadAttention(d, h, 0.0)
    x = torch.rand(b, s, d)
    q, kT, v = mha.q_kT_v(x)

    assert q.shape == torch.Size([b, h, s, hd])
    assert kT.shape == torch.Size([b, h, hd, s])
    assert v.shape == torch.Size([b, h, s, hd])


@max_score(5)
def test_self_attention():
    mha = MultiHeadAttention(d, h, 0.0)
    x = torch.rand(b, s, d)
    q, kT, v = mha.q_kT_v(x)
    k = rearrange(kT, "b h hd s -> b h s hd")

    attn = mha.self_attention(q, kT, v)
    attn_ref_multihead = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_ref = rearrange(attn_ref_multihead, "b h s hd -> b s (h hd)")

    assert torch.allclose(attn, attn_ref, atol=1e-5, rtol=1e-3)

    x = torch.rand(2, 5, d)
    q, kT, v = mha.q_kT_v(x)
    k = rearrange(kT, "b h hd s -> b h s hd")
    attention_mask = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    attention_mask_with_causal = torch.tensor(
        [
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, True, True, False],
                [False, False, True, True, True],
            ],
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ],
        ]
    )[:, None]
    attn = mha.self_attention(q, kT, v, attention_mask=attention_mask)
    attn_ref_multihead = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask_with_causal
    )
    attn_ref = rearrange(attn_ref_multihead, "b h s hd -> b s (h hd)")

    assert torch.allclose(
        attn[~attn_ref.isnan()], attn_ref[~attn_ref.isnan()], atol=1e-5, rtol=1e-3
    )


@max_score(5)
def test_mha_forward():
    mha = MultiHeadAttention(d, h, 0.0)
    x = torch.rand(b, s, d)
    y = mha(x)
    assert y.shape == torch.Size([b, s, d])


@max_score(5)
def test_lm_forward_on_cpu():
    # does not actually check whether the logits are correct
    # just check if the code runs and the output size is right

    lm = DecoderLM(10, d, h, 128, 4)
    lm.eval()
    iids = torch.LongTensor([[3, 1, 4], [1, 5, 9]])
    logits = lm(iids)
    logits_masked = lm(iids, attention_mask=torch.ones_like(iids, dtype=torch.float))
    assert logits_masked.shape == logits.shape == torch.Size([2, 3, 10])
    assert torch.allclose(logits, logits_masked, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device found")
def test_lm_forward_on_gpu():
    device = "cuda"
    lm = DecoderLM(10, d, h, 128, 4).to(device)
    iids = torch.LongTensor([[3, 1, 4], [1, 5, 9]]).to(device)
    logits = lm(iids)
    assert logits.shape == torch.Size([2, 3, 10])
