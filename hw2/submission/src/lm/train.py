import json
import math
import os
import sys
import time
import random
import itertools
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable
from rich import print

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lm.model import DecoderLM
from lm.utils import (
    count_params,
    determine_device,
    enable_tf32,
    estimate_model_disk_size,
)

import datetime


def random_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        An infinite generator that samples batches of sequences from the
        tokens. Each batch has shape (B x S). Every sequence in the batch is
        a contiguous subsequence of x, sampled uniformly at random. The
        output tensor should be on the right device.
    """

    max_idx = len(tokens) - seq_len
    while True:
        start_indices = torch.randint(0, max_idx + 1, (batch_size,))
        batch = torch.stack([tokens[i:i + seq_len] for i in start_indices]).to(device)
        yield batch


def sequential_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        A generator that yields a batch of tokens at a time. Each batch has
        shape (B x S). Every sequence in the batch is a contiguous subsequence
        of x in sequential order. The output tensor should be on the right
        device.

    Note: If the last batch is incomplete, which could happen when the number
        of tokens is not divisible by (batch_size * seq_len), you could drop
        the last batch.
    """

    n_tokens = batch_size * seq_len
    n_batches = len(tokens) // n_tokens
    for batch in range(n_batches):
        yield tokens[batch * n_tokens : (batch + 1) * n_tokens
                     ].reshape((batch_size, seq_len)).to(device=device)


def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t

        Hint: Question 3.2
        """

        assert max_lr >= min_lr >= 0.0
        assert num_training_steps >= num_warmup_steps >= 0

        if t <= num_warmup_steps:
            lr = max_lr * t/num_warmup_steps
        elif t >= num_training_steps:
            lr = min_lr
        else:  # t >= num_training_steps
            theta = math.pi * (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            lr = min_lr + (math.cos(theta) + 1) * (max_lr - min_lr) / 2
        return lr

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
    input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits

    Args:
        input_ids: the input token ids
        logits: the next token logits produced by the language model

    Returns:
        loss: the mean cross entropy loss for next token prediction

    Hint: Think about what are the groundtruth labels for next token prediction.
    """

    return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))


def train(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    optimizer: torch.optim.Optimizer,
    lr_schedule: Callable[[int], float],
    autocast: torch.autocast | nullcontext = nullcontext(),
    num_training_steps: int = 0,
    grad_accumulation_steps: int = 1,
) -> None:
    """A training loop for the language model

    Args:
        model: the decoder LM
        batch_sampler: a generator that produces batches of token ids
        optimizer: an optimizer for gradient update
        lr_schedule: a callable that produces the learning at a step number
        autocast: a context manager that handles tensor casting (you do not need
          to care about this for your implementation)
        num_training_steps: number of steps to train for
        grad_accumulation_steps: number of "micro" training steps before each
          gradient update
    """
    # stores training losses for the 20 latest steps
    losses = deque(maxlen=20 * grad_accumulation_steps)

    for step in (pbar := trange(num_training_steps)):
        t0 = time.time()
        optimizer.zero_grad()
        lr = lr_schedule(step)
        set_lr(optimizer, lr)

        for _ in range(grad_accumulation_steps):
            # TODO: sample a batch, generate logits and compute loss
            input_ids = next(batch_sampler)
            with autocast:
                logits = model(input_ids)
            loss = compute_language_modeling_loss(input_ids, logits)
            (loss / grad_accumulation_steps).backward()
            loss_f = loss.item()
            losses.append(loss_f)

        # TODO: update the model using the accumulated gradients
        optimizer.step()
        loss_mean = np.mean(losses).item()

        FLOPs_per_step = (
            model.flops_per_token
            * input_ids.shape[0]
            * input_ids.shape[1]
            * grad_accumulation_steps
        )
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        pbar.set_postfix(
            {
                "train loss": f"{loss_mean:.2f}",
                "TFLOPS": f"{FLOPs_per_step / dt / 1e12:.1f}",
            }
        )
        wandb.log({"train-loss": loss_mean, "learning-rate": lr}, step=step)


@torch.inference_mode()
def evaluate(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    autocast: torch.autocast | nullcontext = nullcontext(),
) -> dict[str, float]:
    losses = []

    for input_ids in tqdm(batch_sampler, desc="evaluating.."):
        with autocast:
            logits = model(input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


def load_model(config):
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
    return model, model_disk_size_MB 


def load_tokens():
    input_file = 'tokens.npz'
    tokens = np.load(os.path.join("data", input_file))

    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))
    return train_tokens, val_tokens


def estimate_flops(config, model, train_tokens):
    return (
        model.flops_per_token
        * config.num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )



def product_dict(d):
    res = []
    keys = d.keys()
    for instance in itertools.product(*d.values()):
        res.append(dict(zip(keys, instance)))
    return res


def try_random_configs():
    candidates = dict(
        n_embd=[128, 256, 512],
        n_head=[4, 8],
        n_positions=[64, 128, 256, 512],
        n_layer=[4, 6, 8],
        batch_size=[128, 256, 512],
        grad_accumulation_steps=[1],
        min_lr=[1e-3, 1e-4, 1e-5])
    configs = product_dict(candidates)
    random.shuffle(configs)
    tokenizer = tiktoken.get_encoding('gpt2')
    device = determine_device()
    train_tokens, val_tokens = load_tokens()
    i = 0
    for config in configs:
        config = OmegaConf.create(config)
        model = DecoderLM(tokenizer.n_vocab,
                          n_embd=config.n_embd,
                          n_head=config.n_head,
                          n_positions=config.n_positions,
                          n_layer=config.n_layer).to(device)
        model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
        ## We want the model to be as large as possible, but not any larger.
        if model_disk_size_MB > 98 or model_disk_size_MB < 60:
            print(f'Model too big or small, size: {model_disk_size_MB} MB.')
            model = None
            continue
        config.output_dir = f'outputs/random-sweep/{i}'
        os.makedirs(config.output_dir, exist_ok=True)
        run_name = f'{config.output_dir} {datetime.datetime.now()}';
        config.num_training_steps = int(1e15//(model.flops_per_token*config.grad_accumulation_steps*config.batch_size*config.seq_len))
        config.max_lr = config.min_lr * 10
        config.num_warmup_steps = config.num_training_steps//10
        config.seq_len = config.n_positions
        train_sampler = random_batch_sampler(
            train_tokens, device, config.batch_size, config.seq_len
        )
        val_sampler = sequential_batch_sampler(
            val_tokens, device, config.batch_size, config.seq_len
        )

        # prepare optimizer and lr schedule
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,  # will set this dynamically in the training loop
            betas=(0.9, 0.95),
            fused=device == "cuda",
        )
        lr_schedule = cosine_lr_schedule(
            config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
        )
        autocast = (
            torch.autocast(
                device,
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
            )
            if device == "cuda"
            else nullcontext()
        )
        # training
        wandb.init(project="llms-hw2", config=OmegaConf.to_container(config), name=run_name)
        model.train()
        try:
            train(
                model,
                train_sampler,
                optimizer,
                lr_schedule,
                autocast,
                config.num_training_steps,
                config.grad_accumulation_steps,
            )
        except:
            print('Training failed. Most likely OOM.')
            wandb.finish(-1)
            model = None
            continue
        # evaluation
        model.eval()
        eval_results = evaluate(model, val_sampler, autocast)
        print("evaluation results:", json.dumps(eval_results))
        with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        wandb.finish(0)
        print("done!")
        i += 1
        if i >= 50:
            break


def main():
    enable_tf32()

    # create an output directory and dump the configuration file
    assert len(sys.argv) > 1, "provide a configuration file"
    config = OmegaConf.load(sys.argv[1])
    os.makedirs(config.output_dir, exist_ok=True)

    run_name = f'{config.output_dir} {datetime.datetime.now()}';
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="\n")
    wandb.init(project="llms-hw2", config=OmegaConf.to_container(config), name=run_name)

    # initialize tokenizer and model
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
    if model_disk_size_MB > 98:
        print(
            f"[red]WARNING: your model is {model_disk_size_MB:.1f}MB. "
            "The largest model size allowed by GradeScope is 100MB, "
            "and you may have trouble with submitting the assignment. "
            "Please update your config so your model is at most 100 MB.[/red]"
        )
    else:
        print(
            f"Your model is {model_disk_size_MB:.1f}MB. This should be within "
            "the 100MB limit of Gradescope."
        )

    # prepare data and data generator
    assert config.seq_len <= config.model_config.n_positions

    input_file = config.get('input_file', 'tokens.npz')
    tokens = np.load(os.path.join("data", input_file))

    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )
    print(f"train dataset tokens = {len(train_tokens) / 1e6:.0f}M")
    FLOPs = (
        model.flops_per_token
        * config.num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    print(f"train FLOPs = {FLOPs:.2e}")
    if FLOPs > 1e17:
        print(
            f"[red]WARNING: your train FLOPs is {FLOPs:.2e}. "
            "This is more than the max compute that we allow (1e+17). "
            "Please reduce your model size or train steps.[/red]"
        )

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # will set this dynamically in the training loop
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    autocast = (
        torch.autocast(
            device,
            dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        )
        if device == "cuda"
        else nullcontext()
    )
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        config.num_training_steps,
        config.grad_accumulation_steps,
    )

    # save the trained model
    model_path = os.path.join(config.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # evaluation
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print("evaluation results:", json.dumps(eval_results))
    with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print("done!")


if __name__ == "__main__":
    main()
