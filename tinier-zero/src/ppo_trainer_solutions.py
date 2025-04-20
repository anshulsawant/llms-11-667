# src/ppo_trainer_solutions.py
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.optim import AdamW
# Try importing 8-bit AdamW from bitsandbytes
try:
    import bitsandbytes.optim as bnb_optim
    bnb_available = True
except ImportError:
    # print("Warning: bitsandbytes not found. 8-bit Adam optimizer will not be available.")
    bnb_available = False

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from trl.models import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import numpy as np
import random
import re
import math
from tqdm.auto import tqdm
import os
from omegaconf import OmegaConf, DictConfig  # Import OmegaConf
import argparse  # For command-line arguments
import sys  # To modify path for tests if needed


# --- Helper Functions (Masked Ops, Reward) ---
def masked_mean(tensor, mask, dim=None):
    """Calculates mean of tensor elements specified by mask."""
    if mask is None:
        return torch.mean(tensor, dim=dim)
    # Ensure mask is boolean and expanded
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    # Ensure shapes match after expansion
    if tensor.shape != mask.shape:
        mask = mask.expand_as(tensor)

    masked_tensor = torch.where(
        mask, tensor,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    # Add epsilon for stability in case mask sum is zero
    mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8)
    return mean


def masked_whiten(tensor, mask, shift_mean=True):
    """Whitens the tensor values specified by the mask."""
    # Ensure mask is boolean and expanded
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    # Ensure shapes match after expansion
    if tensor.shape != mask.shape:
        mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_variance = torch.where(
        mask, (tensor - mean)**2,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    variance = masked_mean(masked_tensor_variance, mask, dim=None)
    # Add epsilon for numerical stability
    std = torch.sqrt(variance + 1e-8)

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    # Return 0 where mask is False for whitened tensor to avoid propagating non-sensical values
    return torch.where(
        mask, whitened,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))


def extract_gsm8k_solution(solution_str):
    """Extracts the numerical answer from the #### format."""
    # Use strict method first
    solution = re.search(r"####\s*([-+]?\s*[\d\.\,]+)", solution_str)
    if solution is None:
        # Try finding the last number if strict format fails (more flexible)
        answer = re.findall(r"([-+]?\s*[\d\.\,]+)", solution_str)
        if len(answer) > 0:
            # Take the last number found
            final_answer_str = answer[-1].replace(',', '').replace(' ', '')
            try:
                # Validate if it's a number
                float(final_answer_str)
                return final_answer_str
            except ValueError:
                return None  # Last "number" wasn't valid
        else:
            return None  # No number found
    else:
        # Extract from #### format
        final_answer_str = solution.group(1).replace(',', '').replace(' ', '')
        return final_answer_str


def compute_gsm8k_reward(generated_text, ground_truth_str):
    """Computes reward: 1.0 if extracted answer matches ground truth, 0 otherwise."""
    extracted_answer_str = extract_gsm8k_solution(generated_text)
    if extracted_answer_str is None:
        return 0.0
    try:
        # Compare numerically after converting
        extracted_answer = float(extracted_answer_str)
        ground_truth = float(ground_truth_str)
        # Use math.isclose for robust float comparison
        if math.isclose(extracted_answer, ground_truth):
            return 1.0
        else:
            return 0.0
    except ValueError:
        # Handle cases where extraction gives non-numeric results despite regex
        return 0.0


# --- Core PPO Logic (Implementations Included) ---

# Implementation 1: Policy Loss
def compute_policy_loss(log_probs_new, log_probs_old, advantages,
                        response_mask, clip_ratio):
    """
    Computes PPO policy loss (clipped surrogate objective).
    Reference: TinyZero/verl/trainer/ppo/core_algos.py `compute_policy_loss`
    """
    with torch.no_grad(
    ):  # Should not update advantages during policy loss calculation
        # Ensure mask aligns with logprobs length (batch, resp_len)
        mask = response_mask.bool()
        # Align advantages: assumes advantages correspond to states *before* actions
        if advantages.shape != log_probs_old.shape:
            raise ValueError(
                f"Shape mismatch between advantages {advantages.shape} and log_probs {log_probs_old.shape} in policy loss"
            )
        advantages_aligned = advantages

    # Calculate ratio, clamp log_ratio for stability
    log_ratio = (log_probs_new - log_probs_old).clamp(-20, 20)
    ratio = torch.exp(log_ratio)

    # Calculate surrogate objectives
    surr1 = ratio * advantages_aligned
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio,
                        1.0 + clip_ratio) * advantages_aligned

    # Policy loss = - mean(min(surr1, surr2)) over mask
    policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

    # Calculate clip fraction and approx KL for logging
    with torch.no_grad():
        clip_frac = masked_mean(
            torch.gt(torch.abs(ratio - 1.0), clip_ratio).float(), mask)
        # Negative approx KL (used in some implementations like TRL's) = log_ratio
        # approx_kl = masked_mean(-log_ratio, mask) # KL(new || old)
        # Or KL(old || new) as calculated before:
        approx_kl = masked_mean(log_probs_old - log_probs_new, mask)

    return policy_loss, clip_frac, approx_kl


# Implementation 2: Value Loss
def compute_value_loss(values_new, values_old, returns, response_mask,
                       clip_range_value):
    """
    Computes PPO value loss (clipped).
    Reference: TinyZero/verl/trainer/ppo/core_algos.py `compute_value_loss`
    """
    # Ensure mask aligns with values length (batch, resp_len)
    mask = response_mask.bool()

    # Clip predicted values based on old values
    values_pred_clipped = values_old + torch.clamp(
        values_new - values_old, -clip_range_value, clip_range_value)
    # Calculate squared error losses
    vf_loss1 = (values_new - returns)**2
    vf_loss2 = (values_pred_clipped - returns)**2
    # Value loss = 0.5 * mean(max(vf_loss1, vf_loss2)) over mask
    value_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), mask)

    # Calculate clip fraction for logging
    with torch.no_grad():
        vf_clip_frac = masked_mean(torch.gt(vf_loss2, vf_loss1).float(), mask)

    return value_loss, vf_clip_frac


# Implementation 3: Entropy Loss
def compute_entropy_loss(logits_new, response_mask):
    """
    Computes entropy loss to encourage exploration.
    Reference: TinyZero/verl/trainer/ppo/core_algos.py `compute_entropy_loss`
    """
    # Ensure mask aligns with logits length (batch, resp_len)
    mask = response_mask.bool()
    # Ensure logits are float32 for stable distribution calculation
    logits_float = logits_new.float()

    # Calculate entropy from logits
    dist = torch.distributions.Categorical(logits=logits_float)
    entropy = dist.entropy()  # Shape: (batch, resp_len)

    # Entropy loss = - mean(entropy) over mask
    entropy_loss = -masked_mean(entropy,
                                mask)  # Negative because we maximize entropy

    return entropy_loss


# Implementation 4: GAE Advantages
def compute_gae_advantages(final_rewards, kl_penalties, values, response_mask,
                           gamma, lam):
    """
    Computes GAE advantages based on TinyZero/verl/trainer/ppo/core_algos.py.
    KL penalty is incorporated into the rewards here.
    """
    with torch.no_grad():
        response_length = values.shape[1]
        advantages_reversed = []
        last_gae_lam = 0

        # Construct token-level rewards: 0 everywhere except last token (gets final reward), minus KL penalty
        token_level_rewards = torch.zeros_like(values)
        sequence_lengths = response_mask.sum(dim=1)
        last_token_indices = (sequence_lengths - 1).clamp(
            min=0)  # Ensure indices are valid

        valid_indices = sequence_lengths > 0
        if valid_indices.any():
            batch_indices = torch.arange(values.shape[0],
                                         device=values.device)[valid_indices]
            indices_to_update = last_token_indices[valid_indices]
            rewards_to_apply = final_rewards[valid_indices]
            token_level_rewards.scatter_(1, indices_to_update.unsqueeze(1),
                                         rewards_to_apply.unsqueeze(1))

        # Subtract KL penalty at each step
        token_level_rewards = token_level_rewards - kl_penalties  # Assumes kl_penalties = kl_coeff * kl_divergence

        # GAE calculation loop
        for t in reversed(range(response_length)):
            next_values = values[:, t +
                                 1] if t < response_length - 1 else torch.zeros_like(
                                     values[:, 0])
            current_mask = response_mask[:, t].float()
            delta = token_level_rewards[:,
                                        t] + gamma * next_values * current_mask - values[:,
                                                                                         t]
            last_gae_lam = delta + gamma * lam * last_gae_lam * current_mask
            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values  # Return = Advantage + Value

        # Whiten advantages (normalize)
        advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


# --- Rollout Phase ---
def perform_rollouts(actor_model, ref_model, tokenizer, prompt_dataloader,
                     gen_config, device):
    """
    Generates responses and computes necessary data for PPO update.
    (Full implementation included below)
    """
    # (Implementation remains the same as previous response)
    rollout_buffer = {
        "prompt_input_ids": [],
        "prompt_attention_mask": [],
        "response_input_ids": [],
        "response_attention_mask": [],
        "logprobs": [],
        "ref_logprobs": [],
        "values": [],
        "rewards": [],
        "full_texts": [],
        "ground_truth_answers": []
    }
    actor_model.eval()
    ref_model.eval()
    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"]
        with torch.no_grad():
            generated_output = actor_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            response_ids = generated_output.sequences[:, prompt_ids.shape[1]:]
            full_ids = torch.cat((prompt_ids, response_ids), dim=1)
            response_mask = (response_ids
                             != tokenizer.pad_token_id).long().to(device)
            full_mask = torch.cat((prompt_mask, response_mask), dim=1)
            outputs = actor_model(full_ids, attention_mask=full_mask)
            logits = outputs.logits
            values = outputs.value.squeeze(-1)
            ref_outputs = ref_model(full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits
            prompt_len = prompt_ids.shape[1]
            response_len = response_ids.shape[1]
            if response_len > 0:
                logits_for_logprobs = logits[:, prompt_len - 1:prompt_len +
                                             response_len - 1, :]
                ref_logits_for_logprobs = ref_logits[:, prompt_len -
                                                     1:prompt_len +
                                                     response_len - 1, :]
                target_ids = response_ids
                logprobs_all_vocab = F.log_softmax(logits_for_logprobs, dim=-1)
                ref_logprobs_all_vocab = F.log_softmax(ref_logits_for_logprobs,
                                                       dim=-1)
                logprobs = torch.gather(logprobs_all_vocab, 2,
                                        target_ids.unsqueeze(-1)).squeeze(-1)
                ref_logprobs = torch.gather(
                    ref_logprobs_all_vocab, 2,
                    target_ids.unsqueeze(-1)).squeeze(-1)
                values_response = values[:, prompt_len - 1:prompt_len +
                                         response_len - 1]
                logprobs = logprobs * response_mask
                ref_logprobs = ref_logprobs * response_mask
                values_response = values_response * response_mask
            else:
                logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                ref_logprobs = torch.zeros((prompt_ids.shape[0], 0),
                                           device=device)
                values_response = torch.zeros((prompt_ids.shape[0], 0),
                                              device=device)
            full_decoded_texts = tokenizer.batch_decode(
                full_ids, skip_special_tokens=True)
            rewards = torch.tensor([
                compute_gsm8k_reward(txt, gt)
                for txt, gt in zip(full_decoded_texts, ground_truths)
            ],
                                   dtype=torch.float32,
                                   device=device)
            rollout_buffer["prompt_input_ids"].append(prompt_ids.cpu())
            rollout_buffer["prompt_attention_mask"].append(prompt_mask.cpu())
            rollout_buffer["response_input_ids"].append(response_ids.cpu())
            rollout_buffer["response_attention_mask"].append(
                response_mask.cpu())
            rollout_buffer["logprobs"].append(logprobs.cpu())
            rollout_buffer["ref_logprobs"].append(ref_logprobs.cpu())
            rollout_buffer["values"].append(values_response.cpu())
            rollout_buffer["rewards"].append(rewards.cpu())
            rollout_buffer["full_texts"].extend(full_decoded_texts)
            rollout_buffer["ground_truth_answers"].extend(ground_truths)
    collated_buffer = {}
    padding_value_map = {
        "input_ids":
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "attention_mask":
        0,
    }
    for key, data_list in rollout_buffer.items():
        if key in ["full_texts", "ground_truth_answers"]:
            collated_buffer[key] = data_list
        elif key == "rewards":
            if data_list: collated_buffer[key] = torch.cat(data_list, dim=0)
            else: collated_buffer[key] = torch.empty(0)
        else:
            if data_list:
                if data_list[0].dim() > 0:
                    pad_val = padding_value_map.get(key.split("_")[-1], 0.0)
                    if all(isinstance(t, torch.Tensor) for t in data_list):
                        try:
                            collated_buffer[
                                key] = torch.nn.utils.rnn.pad_sequence(
                                    data_list,
                                    batch_first=True,
                                    padding_value=pad_val)
                        except Exception as e_pad:
                            print(
                                f"Error during padding for key '{key}': {e_pad}. Trying simple concat."
                            )
                            try:
                                collated_buffer[key] = torch.cat(data_list,
                                                                 dim=0)
                            except Exception as e_cat:
                                print(
                                    f"Concat failed for key '{key}': {e_cat}")
                                collated_buffer[key] = []
                    else:
                        print(f"Warning: Non-tensor data for key '{key}'.")
                        collated_buffer[key] = []
                else:
                    collated_buffer[key] = torch.cat(data_list, dim=0)
            else:
                collated_buffer[key] = torch.empty(0)
    return collated_buffer


# --- Update Phase (Implementation Filled In) ---
def perform_ppo_update(actor_model, optimizer, rollout_buffer, cfg: DictConfig,
                       device):
    """
    Performs PPO optimization steps on the collected rollout data.
    """
    actor_model.train()  # Set model to training mode
    aggregate_metrics = {}  # To store aggregated metrics over all mini-batches
    ppo_step_count = 0  # Counter for grad accumulation

    # --- 1. Prepare Data ---
    try:
        # Load tensors from buffer to the training device
        prompt_ids = rollout_buffer["prompt_input_ids"].to(device)
        response_ids = rollout_buffer["response_input_ids"].to(device)
        response_mask = rollout_buffer["response_attention_mask"].to(
            device)  # Crucial mask
        logprobs_old = rollout_buffer["logprobs"].to(
            device)  # Shape: (batch, resp_len)
        ref_logprobs = rollout_buffer["ref_logprobs"].to(
            device)  # Shape: (batch, resp_len)
        values_old = rollout_buffer["values"].to(
            device)  # Shape: (batch, resp_len)
        final_rewards = rollout_buffer["rewards"].to(device)  # Shape: (batch,)

        if response_ids.numel() == 0 or response_ids.shape[1] == 0:
            print("Warning: Skipping update. Empty responses in buffer.")
            return {}
        # Check shapes - logprobs, values, advantages, returns, mask should match response length
        if not (logprobs_old.shape == response_ids.shape == ref_logprobs.shape
                == values_old.shape == response_mask.shape):
            print(
                f"Warning: Shape mismatch detected after collation. Logprobs: {logprobs_old.shape}, Values: {values_old.shape}, Response: {response_ids.shape}, Mask: {response_mask.shape}. Skipping update."
            )
            return {}

    except (KeyError, AttributeError, ValueError,
            TypeError) as e:  # Catch potential errors
        print(
            f"Error loading/validating data from rollout buffer: {e}. Skipping update."
        )
        return {}

    # --- 2. Compute Advantages and Returns (Once before PPO epochs) ---
    with torch.no_grad():
        kl_per_token = logprobs_old - ref_logprobs
        kl_penalties = cfg.ppo.kl_coeff * kl_per_token  # Shape: (batch, resp_len)
        advantages, returns = compute_gae_advantages(
            final_rewards, kl_penalties, values_old, response_mask,
            cfg.ppo.gamma, cfg.ppo.lam)  # Shapes: (batch, resp_len)

    # --- 3. PPO Epoch Loop ---
    num_samples = prompt_ids.shape[0]
    indices = np.arange(num_samples)
    prompt_len = prompt_ids.shape[1]  # Max prompt length in batch
    response_len = response_ids.shape[
        1]  # Max response length in batch after padding

    # Combine prompt and response for forward pass during update
    full_input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    prompt_mask_dev = rollout_buffer["prompt_attention_mask"].to(
        device)  # Load prompt mask to device
    full_attention_mask = torch.cat((prompt_mask_dev, response_mask), dim=1)

    optimizer.zero_grad()  # Zero gradients once before starting updates

    for ppo_epoch in range(cfg.ppo.epochs):
        np.random.shuffle(indices)
        # --- 4. Mini-batch Loop ---
        for i in range(0, num_samples, cfg.ppo.mini_batch_size):
            ppo_step_count += 1
            batch_indices = indices[i:i + cfg.ppo.mini_batch_size]

            # Slice mini-batch data
            batch_full_ids = full_input_ids[batch_indices]
            batch_full_mask = full_attention_mask[batch_indices]
            batch_logprobs_old = logprobs_old[
                batch_indices]  # Shape: (mini_batch, resp_len)
            batch_values_old = values_old[
                batch_indices]  # Shape: (mini_batch, resp_len)
            batch_advantages = advantages[
                batch_indices]  # Shape: (mini_batch, resp_len)
            batch_returns = returns[
                batch_indices]  # Shape: (mini_batch, resp_len)
            batch_response_mask = response_mask[
                batch_indices]  # Shape: (mini_batch, resp_len)
            # Need response tokens for calculating new logprobs from logits
            batch_response_tokens = response_ids[
                batch_indices]  # Shape: (mini_batch, resp_len)

            # --- Implementation of PPO Mini-batch Update ---
            # 1. Forward Pass: Get new logits and values from actor_model
            outputs = actor_model(batch_full_ids,
                                  attention_mask=batch_full_mask)
            logits_new = outputs.logits  # Shape: (mini_batch, full_seq_len, vocab_size)
            values_new = outputs.value.squeeze(
                -1)  # Shape: (mini_batch, full_seq_len)

            # 2. Calculate New Logprobs & Extract New Values for response part
            if response_len > 0:
                logits_new_response = logits_new[:, prompt_len - 1:prompt_len +
                                                 response_len - 1, :]
                logprobs_all_vocab_new = F.log_softmax(logits_new_response,
                                                       dim=-1)
                logprobs_new = torch.gather(
                    logprobs_all_vocab_new, 2,
                    batch_response_tokens.unsqueeze(-1)).squeeze(
                        -1)  # Shape: (mini_batch, resp_len)
                values_new_response = values_new[:, prompt_len - 1:prompt_len +
                                                 response_len -
                                                 1]  # Shape: (mini_batch, resp_len)

                # Apply mask to ensure consistency (although loss functions should also use mask)
                logprobs_new = logprobs_new * batch_response_mask
                values_new_response = values_new_response * batch_response_mask
            else:  # Should have been caught earlier, but safety check
                logprobs_new = torch.zeros_like(batch_logprobs_old)
                values_new_response = torch.zeros_like(batch_values_old)

            # 4. Calculate Losses
            policy_loss, p_clip_frac, approx_kl = compute_policy_loss(
                logprobs_new, batch_logprobs_old, batch_advantages,
                batch_response_mask, cfg.ppo.clip_ratio)
            value_loss, v_clip_frac = compute_value_loss(
                values_new_response, batch_values_old, batch_returns,
                batch_response_mask, cfg.ppo.clip_range_value)
            # Ensure logits passed to entropy loss match the dimension of the mask (resp_len)
            entropy_loss = compute_entropy_loss(
                logits_new[:, prompt_len - 1:prompt_len + response_len -
                           1, :],  # Logits for response tokens
                batch_response_mask  # Mask for response tokens
            )

            # 5. Combine Losses
            loss = policy_loss + cfg.ppo.vf_coeff * value_loss + cfg.ppo.entropy_coeff * entropy_loss

            # 6. Backward Pass
            scaled_loss = loss / cfg.ppo.gradient_accumulation_steps
            scaled_loss.backward()

            # 7. Store Metrics
            current_metrics = {
                'loss/policy': policy_loss.item(),
                'loss/value': value_loss.item(),
                'loss/entropy': -entropy_loss.item(),  # Store positive entropy
                'loss/total': loss.item(),
                'params/policy_clip_frac': p_clip_frac.item(),
                'params/value_clip_frac': v_clip_frac.item(),
                'params/approx_kl': approx_kl.item(),
            }
            for key, val in current_metrics.items():
                aggregate_metrics.setdefault(key, []).append(val)

            # 8. Optimizer Step
            if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
                if any(p.grad is not None for p in actor_model.parameters()):
                    # Optional: Gradient Clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor_model.parameters(), max_norm=1.0)
                    aggregate_metrics.setdefault('params/grad_norm',
                                                 []).append(grad_norm.item())

                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    # Aggregate metrics over the PPO epoch
    final_metrics = {
        key: np.mean(val)
        for key, val in aggregate_metrics.items() if val
    }  # Avoid division by zero
    return final_metrics


# --- Main Training Function ---
def train(cfg: DictConfig):
    """Main training loop, now takes OmegaConf DictConfig object"""
    # --- Setup ---
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.training.seed)
    else:
        if cfg.training.device == "cuda":
            print(
                "Warning: CUDA requested but not available, falling back to CPU."
            )
        device = torch.device("cpu")
    print(f"Using device: {device}")

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    try:
        OmegaConf.save(cfg, os.path.join(output_dir, "effective_config.yaml"))
    except Exception as e:
        print(f"Error saving effective config: {e}")

    # --- Load Model & Tokenizer ---
    print(f"Loading tokenizer: {cfg.model.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {cfg.model.name}")
    model_kwargs = {}
    model_dtype_str = cfg.model.get("torch_dtype", "auto")
    model_dtype = getattr(torch, model_dtype_str,
                          "auto") if model_dtype_str != "auto" else "auto"
    if model_dtype != "auto": print(f"Setting model dtype to: {model_dtype}")
    model_kwargs["torch_dtype"] = model_dtype
    if cfg.model.get("trust_remote_code", False):
        model_kwargs["trust_remote_code"] = True

    try:
        actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model.name, **model_kwargs)
        actor_model.to(device)
        if actor_model.config.pad_token_id is None:
            actor_model.config.pad_token_id = tokenizer.pad_token_id

        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model.name, **model_kwargs)
        ref_model.to(device)
        if ref_model.config.pad_token_id is None:
            ref_model.config.pad_token_id = tokenizer.pad_token_id
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        print("Models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- Load Dataset & Preprocess ---
    print(f"Loading dataset: {cfg.dataset.name}")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.split)

    def preprocess_dataset(example):
        example["prompt"] = cfg.dataset.prompt_format.format(
            question=example["question"])
        tokenized_prompt = tokenizer(example["prompt"],
                                     max_length=cfg.dataset.max_prompt_length,
                                     truncation=True,
                                     padding=False,
                                     return_tensors=None)
        example["input_ids"] = tokenized_prompt["input_ids"]
        example["attention_mask"] = tokenized_prompt["attention_mask"]
        example["ground_truth_answer"] = example["answer"].split(
            "####")[-1].strip()
        return example

    tokenized_dataset = dataset.map(preprocess_dataset,
                                    remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type="torch")
    print(f"Dataset preprocessed. Samples: {len(tokenized_dataset)}")

    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        padded_inputs = tokenizer.pad({"input_ids": input_ids},
                                      padding='longest',
                                      return_tensors="pt",
                                      return_attention_mask=True)
        ground_truths = [item['ground_truth_answer'] for item in batch]
        return {
            "prompt_input_ids": padded_inputs["input_ids"],
            "prompt_attention_mask": padded_inputs["attention_mask"],
            "ground_truth_answers": ground_truths
        }

    # --- Optimizer ---
    use_8bit = cfg.ppo.get("use_8bit_adam", False)
    if use_8bit and bnb_available and device.type == "cuda":
        print("Using 8-bit AdamW Optimizer (bitsandbytes)")
        optimizer = bnb_optim.AdamW8bit(actor_model.parameters(),
                                        lr=cfg.ppo.learning_rate)
    else:
        if use_8bit:
            print(
                "Warning: 8-bit Adam not used (bitsandbytes unavailable or not on CUDA). Using standard AdamW."
            )
        else:
            print("Using standard AdamW Optimizer")
        optimizer = AdamW(actor_model.parameters(), lr=cfg.ppo.learning_rate)

    # --- Generation Config ---
    gen_config = GenerationConfig(max_new_tokens=cfg.generation.max_new_tokens,
                                  min_new_tokens=cfg.generation.min_new_tokens,
                                  temperature=cfg.generation.temperature,
                                  top_k=cfg.generation.top_k,
                                  top_p=cfg.generation.top_p,
                                  do_sample=cfg.generation.do_sample)

    # --- Main Training Loop ---
    print("--- Starting PPO Training ---")  # Changed title slightly
    for ppo_step in range(cfg.training.total_ppo_steps):
        print(f"\nPPO Step {ppo_step + 1}/{cfg.training.total_ppo_steps}")
        print("Phase 1: Generating Rollouts...")
        dataloader = torch.utils.data.DataLoader(tokenized_dataset,
                                                 batch_size=cfg.ppo.batch_size,
                                                 shuffle=True,
                                                 collate_fn=collate_fn)
        rollout_buffer = perform_rollouts(actor_model, ref_model, tokenizer,
                                          dataloader, gen_config, device)
        avg_reward = 0.0
        num_rollouts = 0
        if rollout_buffer and "rewards" in rollout_buffer and rollout_buffer[
                "rewards"].numel() > 0:
            num_rollouts = rollout_buffer["rewards"].shape[0]
            avg_reward = rollout_buffer["rewards"].mean().item()
            print(
                f"Rollout complete ({num_rollouts} samples). Average reward: {avg_reward:.4f}"
            )
        else:
            print("Rollout buffer seems empty or invalid after generation.")

        print("Phase 2: Performing PPO Updates...")
        is_buffer_valid = (
            rollout_buffer and num_rollouts > 0
            and all(k in rollout_buffer
                    for k in ["response_input_ids", "logprobs", "values"])
            and rollout_buffer["response_input_ids"].numel() > 0
            and rollout_buffer["response_input_ids"].shape[1] > 0)
        if is_buffer_valid:
            metrics = perform_ppo_update(actor_model, optimizer,
                                         rollout_buffer, cfg, device)
            if metrics and (ppo_step + 1) % cfg.training.log_interval == 0:
                print(f"PPO Step {ppo_step+1} Metrics (Avg over Epoch):")
                log_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(log_str)
                print(f"  Reward (mean from rollout): {avg_reward:.4f}")
            elif not metrics:
                print("Update function returned empty metrics.")
        else:
            print("Skipping update step because rollout buffer is invalid.")

        if (ppo_step + 1) % cfg.training.save_interval == 0:
            step_output_dir = os.path.join(output_dir, f"step_{ppo_step + 1}")
            print(f"Saving model checkpoint to {step_output_dir}...")
            os.makedirs(step_output_dir, exist_ok=True)
            try:
                actor_model.save_pretrained(step_output_dir)
                tokenizer.save_pretrained(step_output_dir)
                print(f"Model saved.")
            except Exception as e:
                print(f"Error saving model: {e}")

    print("--- PPO Training Finished ---")
    final_output_dir = os.path.join(output_dir, "final")
    print(f"Saving final model to {final_output_dir}...")
    os.makedirs(final_output_dir, exist_ok=True)
    try:
        actor_model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Final model saved.")
    except Exception as e:
        print(f"Error saving final model: {e}")


# --- Entry Point ---
def main_cli():
    """Handles command-line argument parsing and initiates training."""
    # (Implementation remains the same as previous response)
    parser = argparse.ArgumentParser(description="PPO RL Tutorial Trainer")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to the config directory relative to project root")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config.yaml",
        help="Name of the config file (e.g., config.yaml, config_debug.yaml)")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values "
        "(e.g., training.device=cuda:0 ppo.learning_rate=1e-5)",
    )
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir_abs = os.path.join(project_root, args.config_path)
    conf_path = os.path.join(config_dir_abs, args.config_name)
    if not os.path.exists(conf_path):
        alt_conf_path = os.path.join(args.config_path, args.config_name)
        conf_path = alt_conf_path if os.path.exists(alt_conf_path) else None
    if conf_path is None or not os.path.exists(conf_path):
        print(f"Error: Config file not found at specified/relative paths.")
        sys.exit(1)
    print(f"Loading config from: {conf_path}")
    cfg = OmegaConf.load(conf_path)
    if 'defaults' in cfg and isinstance(cfg.defaults, list) and cfg.defaults:
        base_conf_name = cfg.defaults[0] + ".yaml"
        base_conf_path_abs = os.path.join(config_dir_abs, base_conf_name)
        base_conf_path_rel = os.path.join(args.config_path, base_conf_name)
        base_path_to_load = base_conf_path_abs if os.path.exists(
            base_conf_path_abs) else base_conf_path_rel if os.path.exists(
                base_conf_path_rel) else None
        if base_path_to_load:
            print(f"Loading base config from: {base_path_to_load}")
            base_cfg = OmegaConf.load(base_path_to_load)
            cfg = OmegaConf.merge(base_cfg, cfg)
        else:
            print(f"Warning: Base config {base_conf_name} not found.")
        if 'defaults' in cfg:
            try:
                OmegaConf.set_struct(cfg, False)
                cfg.pop('defaults')
                OmegaConf.set_struct(cfg, True)
            except Exception as e_pop:
                print(f"Note: Could not pop 'defaults' key: {e_pop}")
    if args.overrides:
        print(f"Applying overrides: {args.overrides}")
        cli_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)
        print("--------- Final Configuration ---------")
        try:
            OmegaConf.resolve(cfg)
            print(OmegaConf.to_yaml(cfg))
        except Exception as e_resolve:
            print(f"Error resolving config: {e_resolve}")
            print(OmegaConf.to_yaml(cfg, resolve=False))
    print("---------------------------------------")
    train(cfg)


if __name__ == "__main__":
    main_cli()
