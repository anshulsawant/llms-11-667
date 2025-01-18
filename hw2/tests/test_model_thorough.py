import pytest
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from lm.model import DecoderLM, MultiHeadAttention, FeedForward, DecoderBlock
from lm.nano_model import CausalSelfAttention, MLP, Block, GPT, GPTConfig

from pytest_utils.decorators import max_score

b, s, d, h, hd = 2, 3, 8, 2, 4
config = GPTConfig(block_size=s, vocab_size=10, n_layer=2, n_head=h, n_embd=d, dropout=0.0, bias=False)

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def test_mha():
    mha = MultiHeadAttention(d, h, 0.0)
    mha_ref = CausalSelfAttention(config)

    mha.apply(_init_weights)

    mha_ref.q_attn.weight = mha.q_attn.weight
    mha_ref.k_attn.weight = mha.k_attn.weight
    mha_ref.v_attn.weight = mha.v_attn.weight
    mha_ref.c_proj.weight = mha.proj.weight
    
    x = torch.rand(b, s, d)

    y = mha(x)
    y_ref = mha_ref(x)

    assert not torch.allclose(y, torch.tensor(0.0), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-3)


def test_ff():
    ff = FeedForward(config.n_embd, p_dropout=config.dropout)
    ff.apply(_init_weights)
    ff_ref = MLP(config)

    ff_ref.c_fc.weight = ff.linear_in.weight
    ff_ref.c_proj.weight = ff.linear_out.weight

    x = torch.rand(b, s, d)

    y = ff(x)
    y_ref = ff_ref(x)
    
    assert not torch.allclose(y, torch.tensor(0.0), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-3)

    
def test_decoder_block():
    block = DecoderBlock(config.n_embd, config.n_head, p_dropout=config.dropout)
    block.apply(_init_weights)
    block_ref = Block(config)

    block_ref.attn.q_attn.weight = block.mha.q_attn.weight
    block_ref.attn.k_attn.weight = block.mha.k_attn.weight
    block_ref.attn.v_attn.weight = block.mha.v_attn.weight
    block_ref.attn.c_proj.weight = block.mha.proj.weight
    block_ref.mlp.c_fc.weight = block.ff.linear_in.weight
    block_ref.mlp.c_proj.weight = block.ff.linear_out.weight

    x = torch.rand(b, s, d)

    y = block(x, attention_mask=None)
    y_ref = block_ref(x)
    
    assert not torch.allclose(y, torch.tensor(0.0), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-3)

    
def test_decoder():
    decoder = DecoderLM(n_vocab=config.vocab_size,
                        n_embd=config.n_embd,
                        n_head=config.n_head,
                        n_positions=config.block_size,
                        n_layer=config.n_layer,
                        p_dropout=config.dropout)
    decoder_ref = GPT(config)

    decoder.apply(_init_weights)
    
    block = decoder.blocks[0]
    block_ref = decoder_ref.transformer.h[0]
    block_ref.attn.q_attn.weight = block.mha.q_attn.weight
    block_ref.attn.k_attn.weight = block.mha.k_attn.weight
    block_ref.attn.v_attn.weight = block.mha.v_attn.weight
    block_ref.attn.c_proj.weight = block.mha.proj.weight
    block_ref.mlp.c_fc.weight = block.ff.linear_in.weight
    block_ref.mlp.c_proj.weight = block.ff.linear_out.weight

    block = decoder.blocks[1]
    block_ref = decoder_ref.transformer.h[1]
    block_ref.attn.q_attn.weight = block.mha.q_attn.weight
    block_ref.attn.k_attn.weight = block.mha.k_attn.weight
    block_ref.attn.v_attn.weight = block.mha.v_attn.weight
    block_ref.attn.c_proj.weight = block.mha.proj.weight
    block_ref.mlp.c_fc.weight = block.ff.linear_in.weight
    block_ref.mlp.c_proj.weight = block.ff.linear_out.weight

    decoder_ref.lm_head.weight = decoder.linear.weight
    decoder_ref.transformer.wte.weight = decoder.token_embeddings.weight
    decoder_ref.transformer.wpe.weight = decoder.position_embeddings.weight
    
    
    x = torch.randint(high=config.vocab_size, size=(b, s))

    y = decoder(x, attention_mask=None)
    y_ref = decoder_ref(x)

    assert not torch.allclose(y, torch.tensor(0.0), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-3)
