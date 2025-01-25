import math
from contextlib import nullcontext

import torch
import wandb
from pytest_utils.decorators import max_score

from lm.model import DecoderLM
from lm.train import (
    compute_language_modeling_loss,
    cosine_lr_schedule,
    evaluate,
    random_batch_sampler,
    sequential_batch_sampler,
    train,
)


wandb.init(mode="disabled")


@max_score(5)
def test_random_batch_sampler():
    tokens = torch.arange(100000)
    batch_size, seq_len = 32, 256
    sampler = random_batch_sampler(tokens, "cpu", batch_size, seq_len)

    for _ in range(100):
        # check the output size is right
        sample = next(sampler)
        assert sample.shape == torch.Size([batch_size, seq_len])

        # check every sequence looks like [x, x+1, x+2, ...]
        sample_diff = sample - sample[:, [0]]
        assert torch.all(sample_diff == torch.arange(seq_len))

    # take a small number of tokens and verify all of them are sampled
    tokens = torch.arange(7)
    sampler = random_batch_sampler(tokens, "cpu", 64, 4)

    number_set = set()
    for _ in range(100):
        sample = next(sampler).flatten().tolist()
        number_set |= set(sample)

    assert number_set == set(range(7))


@max_score(5)
def test_sequential_batch_sampler():
    num_batches, batch_size, seq_len = 24, 32, 256
    tokens = torch.arange(batch_size * seq_len * num_batches)
    sampler = sequential_batch_sampler(tokens, "cpu", batch_size, seq_len)
    samples = list(sampler)

    assert len(samples) == num_batches
    assert all(sample.shape == torch.Size([batch_size, seq_len]) for sample in samples)
    assert torch.all(torch.cat(samples, 0).flatten() == tokens)

    tokens = torch.arange(batch_size * seq_len * num_batches + 7)

    sampler = sequential_batch_sampler(tokens, "cpu", batch_size, seq_len)
    samples = list(sampler)

    assert len(samples) == num_batches
    assert all(sample.shape == torch.Size([batch_size, seq_len]) for sample in samples)
    assert torch.all(
        torch.cat(samples, 0).flatten() == tokens[: batch_size * seq_len * num_batches]
    )


@max_score(5)
def test_cosine_lr_schedule():
    num_warmup_steps, num_training_steps, min_lr, max_lr = 5, 10, 0.01, 0.1
    schedule = cosine_lr_schedule(num_warmup_steps, num_training_steps, min_lr, max_lr)

    lrs = [schedule(step) for step in range(15)]
    expected_lrs = [
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.091405,
        0.068905,
        0.041094,
        0.018594,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]

    assert all(
        math.isclose(lr, e_lr, rel_tol=1e-3) for lr, e_lr in zip(lrs, expected_lrs)
    )


@max_score(5)
def test_compute_language_modeling_loss():
    input_ids = torch.LongTensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.FloatTensor([[[1.0, -2.0, 3.0] for _ in range(3)] for _ in range(2)])

    assert torch.allclose(
        compute_language_modeling_loss(input_ids, logits),
        torch.FloatTensor([3.1328]),
        rtol=1e-3,
    )


@max_score(5)
def test_train():
    """Fit the model on a trivially learnable dataset - checks drop in loss"""
    device = "cpu"

    # initialize tokenizer and model
    n_vocab = 8
    n_embd = 16
    n_head = 8
    n_positions = 16
    n_layer = 2

    num_warmup_steps = 10
    num_training_steps = 300
    min_lr = 1e-4
    max_lr = 5e-4
    grad_accumulation_steps = 2
    train_batch_size = 4
    seq_len = 4

    model = DecoderLM(n_vocab, n_embd, n_head, n_positions, n_layer).to(device)

    # create a synthetic dataset for testing
    train_tokens = val_tokens = torch.arange(n_vocab)

    train_sampler = random_batch_sampler(
        train_tokens, device, train_batch_size, seq_len
    )
    val_sampler = sequential_batch_sampler(val_tokens, device, 2, seq_len)

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        num_warmup_steps, num_training_steps, min_lr, max_lr
    )
    autocast = nullcontext()
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        num_training_steps,
        grad_accumulation_steps,
    )

    # evaluation
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    assert eval_results["val-loss"] <= 1.3, "failed to fit"


@max_score(0)
def test_train_sanity_check():
    """
    If your language model learns an *unlearnable sequence*, it must be cheating
    by looking at tokens ahead. If you fail this test case, it usually means a
    mistake in manipulating dimensions in the self-attention operation.

    Note: Failing this test case would prevent you from scoring points on
    Question 1.2, 1.3 and 1.4.
    """
    device = "cpu"

    # initialize tokenizer and model
    n_vocab = 8
    n_embd = 16
    n_head = 8
    n_positions = 4
    n_layer = 2

    num_warmup_steps = 10
    num_training_steps = 300
    min_lr = 1e-4
    max_lr = 5e-4
    grad_accumulation_steps = 2
    train_batch_size = 8
    seq_len = 4

    model = DecoderLM(n_vocab, n_embd, n_head, n_positions, n_layer).to(device)

    # generate random bytes
    token_count = (
        train_batch_size * grad_accumulation_steps * seq_len * num_training_steps
    )
    train_tokens = torch.randint(0, n_vocab, (token_count,))
    val_tokens = torch.randint(0, n_vocab, (token_count,))

    train_sampler = random_batch_sampler(
        train_tokens, device, train_batch_size, seq_len
    )
    val_sampler = sequential_batch_sampler(val_tokens, device, 2, seq_len)

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        num_warmup_steps, num_training_steps, min_lr, max_lr
    )
    autocast = nullcontext()
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        num_training_steps,
        grad_accumulation_steps,
    )

    # evaluation
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print(eval_results)
    assert eval_results["val-loss"] >= math.log(n_vocab) * 0.95, "model is cheating"