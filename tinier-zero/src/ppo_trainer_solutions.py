# src/ppo_trainer_solutions.py
# -*- coding: utf-8 -*-
"""
Refactored PPO Trainer script - SOLUTION FILE.
This file contains the complete implementations for the exercises
in ppo_trainer.py. It focuses on modularity and clarity.
"""
import wandb
import accelerate
accelerator = accelerate.Accelerator()
from accelerate import logging
logger = logging.get_logger(__name__, log_level="INFO")
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
# Try importing 8-bit AdamW from bitsandbytes
try:
    import bitsandbytes.optim as bnb_optim
    bnb_available = True
except ImportError:
    bnb_available = False

from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding  # Can be useful if not using custom collate
)
from datasets import load_dataset, Dataset
import numpy as np
import random
import re
import math
from tqdm.auto import tqdm
import os
from omegaconf import OmegaConf, DictConfig
import argparse
import sys
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# == 1. Helper Functions (Masking, Reward, Padding)
# ==============================================================================


def masked_mean(tensor: torch.Tensor,
                mask: Optional[torch.Tensor],
                dim: Optional[int] = None) -> torch.Tensor:
    """Calculates mean of tensor elements specified by mask."""
    if mask is None:
        return torch.mean(tensor, dim=dim)
    mask = mask.bool()
    # Expand mask dimensions if necessary
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)  # Ensure shapes match

    masked_tensor = torch.where(
        mask, tensor,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8
                                         )  # Add epsilon for stability
    return mean


def masked_whiten(tensor: torch.Tensor,
                  mask: Optional[torch.Tensor],
                  shift_mean: bool = True) -> torch.Tensor:
    """Whitens the tensor values specified by the mask."""
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_variance = torch.where(
        mask, (tensor - mean)**2,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    variance = masked_mean(masked_tensor_variance, mask, dim=None)
    std = torch.sqrt(variance + 1e-8)  # Add epsilon for stability

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(
        mask, whitened,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))


def extract_gsm8k_solution(solution_str: str) -> Optional[str]:
    """Extracts the numerical answer from the #### format.

    Does not handle scientific notation.
    """
    # Use strict method first: Search for #### followed by a potential number
    # The regex captures digits, optional sign, dots, and commas.
    solution_match = re.search(r"####\s*([-+]?\s*[\d\.\,]+)(?:\s|$)+", solution_str)
    solution = '' if solution_match is None else solution_match.group(1) 
    try:
        float(solution) 
        return solution
    except ValueError:
        return None # Not a valid number, return None


def compute_gsm8k_reward(generated_text: str, ground_truth_str: str) -> float:
    """Computes reward: 1.0 if extracted answer matches ground truth,
    0.1 if extracted answer is a number (easy reward for getting format right),
    0 otherwise."""
    extracted_answer_str = extract_gsm8k_solution(generated_text)
    if extracted_answer_str is None: return 0.0
    try:
        extracted_answer = float(extracted_answer_str)
        ground_truth = float(ground_truth_str)
        return 1.0 if math.isclose(extracted_answer, ground_truth) else 0.1
    except ValueError:
        return 0.0  # Handle non-numeric cases


def pad_and_collate_tensors(tensor_list: List[torch.Tensor],
                            padding_value: float = 0.0) -> torch.Tensor:
    """
    Pads tensors in a list to the maximum length of the second dimension
    and concatenates them along the first dimension.

    Assumes input tensors are 2D.
    """
    # Find max length in the second dimension (sequence length)
    max_len = max([t.shape[1] for t in tensor_list])
    
    if max_len == 0:  # Handle cases where all sequences have length 0
        total_batch_size = sum(t.shape[0] for t in tensor_list)
        # Return shape (TotalB, 0, ...) matching original dims > 1
        original_shape = tensor_list[0].shape
        return torch.empty((total_batch_size, 0) + original_shape[2:],
                           dtype=tensor_list[0].dtype,
                           device=tensor_list[0].device)

    # Pad each tensor and collect in a new list
    padded_list = []
    for t in tensor_list:
        current_len = t.shape[1]
        padding_needed = max_len - current_len
        if padding_needed > 0:
            pad_tuple = (0, padding_needed, 0, 0)
            t = F.pad(t,
                             tuple(pad_tuple),
                             mode='constant',
                             value=padding_value)
        padded_list.append(t)  # No padding needed or already max length

    # Concatenate the padded tensors along the batch dimension (dim=0)
    return torch.cat(padded_list, dim=0)


# ==============================================================================
# == 2. Core PPO Algorithm Components
# ==============================================================================


def compute_policy_loss(
        log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
        advantages: torch.Tensor, response_mask: torch.Tensor,
        clip_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes PPO policy loss (clipped surrogate objective)."""
    with torch.no_grad():
        mask = response_mask.bool()
        # Advantages should align with states *before* actions (logprobs)
        if advantages.shape != log_probs_old.shape:
            raise ValueError(
                f"Shape mismatch: advantages {advantages.shape}, log_probs {log_probs_old.shape}"
            )
    log_ratio = (log_probs_new - log_probs_old).clamp(
        -20, 20)  # Clamp for stability
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

    with torch.no_grad():
        clip_frac = masked_mean(
            torch.gt(torch.abs(ratio - 1.0), clip_ratio).float(), mask)
        approx_kl = masked_mean(log_probs_old - log_probs_new,
                                mask)  # KL(old || new)

    return policy_loss, clip_frac, approx_kl


def compute_value_loss(
        values_new: torch.Tensor, values_old: torch.Tensor,
        returns: torch.Tensor, response_mask: torch.Tensor,
        clip_range_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes PPO value loss (clipped)."""
    mask = response_mask.bool()
    values_pred_clipped = values_old + torch.clamp(
        values_new - values_old, -clip_range_value, clip_range_value)
    vf_loss1 = (values_new - returns)**2
    vf_loss2 = (values_pred_clipped - returns)**2
    value_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), mask)

    with torch.no_grad():
        vf_clip_frac = masked_mean(torch.gt(vf_loss2, vf_loss1).float(), mask)

    return value_loss, vf_clip_frac


def compute_entropy_loss(logits_new: torch.Tensor,
                         response_mask: torch.Tensor) -> torch.Tensor:
    """Computes entropy loss to encourage exploration."""
    mask = response_mask.bool()
    # Use float32 for stability
    dist = torch.distributions.Categorical(logits=logits_new.float())
    entropy = dist.entropy()
    # Maximize entropy -> minimize negative entropy
    entropy_loss = -masked_mean(entropy, mask)
    return entropy_loss


def compute_gae_advantages(
        final_rewards: torch.Tensor,  # Shape (batch_size,)
        kl_penalties: torch.Tensor,  # Shape (batch_size, resp_len)
        values: torch.Tensor,       # Shape (batch_size, resp_len) - V(s_t)
        response_mask: torch.Tensor,# Shape (batch_size, resp_len)
        gamma: float,
        lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes GAE advantages and returns."""
    with torch.no_grad():
        response_length = values.shape[1]
        advantages_reversed = []
        last_gae_lam = 0 # Stores A_{t+1} during the loop

        # Assign final reward to the last actual token step
        token_level_rewards = torch.zeros_like(values)
        # Ensure response_mask is long for sum()
        sequence_lengths = response_mask.long().sum(dim=1)
        # Ensure indices are long and handle 0-length sequences
        last_token_indices = (sequence_lengths - 1).clamp(min=0)

        valid_indices = sequence_lengths > 0
        if valid_indices.any():
            # Ensure indices used for indexing final_rewards are valid
            rewards_to_apply = final_rewards[valid_indices]
            indices_to_update = last_token_indices[valid_indices]
            rewards_to_apply = rewards_to_apply.to(token_level_rewards.dtype)
            # Scatter reward to the step *before* the terminal state
            token_level_rewards.scatter_(
                1,
                indices_to_update.long().unsqueeze(1), # Ensure index is long
                rewards_to_apply.unsqueeze(1)
            )

        # Incorporate KL penalty at each step
        token_level_rewards = token_level_rewards - kl_penalties

        # GAE loop (Iterate backwards from T-1 down to 0)
        for t in reversed(range(response_length)):
            # Value of next state V(s_{t+1})
            next_values = values[:, t + 1] if t < response_length - 1 else torch.zeros_like(values[:, 0])
            # Mask for *next* step (needed for propagation)
            next_mask = response_mask[:, t + 1].float() if t < response_length - 1 else torch.zeros_like(response_mask[:, 0].float())

            # TD residual: delta_t = r_t + gamma * V(s_{t+1})*mask_{t+1} - V(s_t)
            # Note: We use next_mask for V(s_{t+1}) term. V(s_t) term (values[:, t]) is used regardless of mask_t
            # because delta is non-zero even if s_t is terminal (used for A_t calc).
            delta = token_level_rewards[:, t] + gamma * next_values * next_mask - values[:, t]
            # GAE: A_t = delta_t + gamma * lambda * A_{t+1} * mask_{t+1}
            # last_gae_lam holds A_{t+1} from the previous iteration (or 0 if t+1 was out of bounds/masked).
            # The next_mask ensures that if s_{t+1} is a padded/terminal state, its contribution (A_{t+1}) is zeroed out.
            last_gae_lam = delta + gamma * lam * last_gae_lam * next_mask
            advantages_reversed.append(last_gae_lam)

        

        # Reverse the list and stack into a tensor
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # Returns = Advantages + Values (Return G_t = A_t + V(s_t))
        returns = advantages + values

        # Whiten advantages (normalize) - Use the original response_mask here
        advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


# ==============================================================================
# == 3. Actor Model Definition
# ==============================================================================


class ActorModelWithValueHead(nn.Module):
    """
    Wraps a pre-trained transformer, adding a value head and generation method.
    Computes per-token values.
    """

    def __init__(self, model_name_or_path: str, **kwargs_model_load):
        """Initializes the model, loading the base transformer and value head."""
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **kwargs_model_load)
        self.config = self.base_model.config  # Store config
        # Value head maps hidden states to scalar value
        self.value_head = nn.Linear(self.config.hidden_size, 1)

        # Ensure value_head matches base model dtype ---
        base_model_dtype = next(self.base_model.parameters()).dtype
        self.value_head.to(base_model_dtype)
        
        # Basic initialization for value head
        self.value_head.weight.data.normal_(mean=0.0, std=0.01)
        self.value_head.bias.data.zero_()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: computes logits and per-token values."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Need hidden states for value head
            **kwargs)
        logits = outputs.logits
        # Get last hidden state (batch, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        # Compute value for each token's hidden state
        values = self.value_head(last_hidden_state).squeeze(
            -1)  # Shape: (batch, seq_len)
        return logits, values

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Forwards generate call to the base model."""
        return self.base_model.generate(*args, **kwargs)


# ==============================================================================
# == 4. Rollout Phase Logic
# ==============================================================================


def generate_responses(
        model: ActorModelWithValueHead, tokenizer: PreTrainedTokenizerBase,
        prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
        gen_config: GenerationConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates responses for a batch of prompts."""
    model.eval()  # Ensure model is in eval mode for generation
    with torch.no_grad():
        generated_output = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=gen_config,
            pad_token_id=tokenizer.pad_token_id  # Important for generation
        )
        # Extract only generated tokens (after prompt)
        response_ids = generated_output[:, prompt_ids.shape[1]:]
        # Create response mask (1 for real tokens, 0 for padding)
        response_mask = (response_ids != tokenizer.pad_token_id).long()
    return response_ids, response_mask


def calculate_rollout_stats(
    actor_model: ActorModelWithValueHead,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,  # Shape (batch, prompt_len)
    prompt_mask: torch.Tensor,  # Shape (batch, prompt_len)
    response_ids: torch.Tensor,  # Shape (batch, resp_len)
    response_mask: torch.Tensor  # Shape (batch, resp_len)
) -> Dict[str, torch.Tensor]:
    """Calculates logprobs, ref_logprobs, values, and rewards for a batch."""
    actor_model.eval()
    ref_model.eval()
    with torch.no_grad():
        # Combine prompt and response for forward passes
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        full_mask = torch.cat((prompt_mask, response_mask), dim=1)

        # Get actor logits and values
        actor_logits, actor_values = actor_model(full_ids,
                                                 attention_mask=full_mask)
        # Get reference model logits
        ref_logits = ref_model(full_ids, attention_mask=full_mask).logits

        # --- Calculate Logprobs and Values for the RESPONSE part ---
        prompt_len = prompt_ids.shape[1]
        resp_len = response_ids.shape[1]
        full_len = full_ids.shape[1]

        # Logits/Values indices: We need state BEFORE generating token R_t
        # Corresponds to indices from prompt_len-1 to full_len-2
        start_idx = prompt_len - 1
        end_idx = full_len - 1  # Slice up to (but not including) this index

        if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
            # Handle cases with empty response or invalid indices
            logprobs = torch.empty((prompt_ids.shape[0], 0),
                                   dtype=torch.float,
                                   device=prompt_ids.device)
            ref_logprobs = torch.empty((prompt_ids.shape[0], 0),
                                       dtype=torch.float,
                                       device=prompt_ids.device)
            values = torch.empty((prompt_ids.shape[0], 0),
                                 dtype=torch.float,
                                 device=prompt_ids.device)
        else:
            logits_resp = actor_logits[:, start_idx:end_idx, :]
            ref_logits_resp = ref_logits[:, start_idx:end_idx, :]
            values = actor_values[:, start_idx:
                                  end_idx]  # Values for states BEFORE response tokens

            # Target IDs are the response tokens
            target_ids = response_ids

            # Ensure shapes match before gather (can differ if generation stopped early)
            current_resp_len = logits_resp.shape[1]
            if current_resp_len != target_ids.shape[1]:
                min_len = min(current_resp_len, target_ids.shape[1])
                logits_resp = logits_resp[:, :min_len, :]
                ref_logits_resp = ref_logits_resp[:, :min_len, :]
                target_ids = target_ids[:, :min_len]
                values = values[:, :min_len]
                # Adjust response mask as well if lengths mismatch
                response_mask_adjusted = response_mask[:, :min_len]
            else:
                response_mask_adjusted = response_mask

            # Calculate log probabilities
            logprobs_all = F.log_softmax(logits_resp, dim=-1)
            ref_logprobs_all = F.log_softmax(ref_logits_resp, dim=-1)
            logprobs = torch.gather(logprobs_all, 2,
                                    target_ids.unsqueeze(-1)).squeeze(-1)
            ref_logprobs = torch.gather(ref_logprobs_all, 2,
                                        target_ids.unsqueeze(-1)).squeeze(-1)

            # Apply mask (mask should match the potentially adjusted length)
            logprobs = logprobs * response_mask_adjusted
            ref_logprobs = ref_logprobs * response_mask_adjusted
            values = values * response_mask_adjusted

    return {
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
        "values": values,
    }


def perform_rollouts(actor_model: ActorModelWithValueHead,
                     ref_model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizerBase,
                     prompt_dataloader: DataLoader,
                     gen_config: GenerationConfig,
                     device: torch.device) -> Dict[str, Any]:
    """Generates responses and computes stats for PPO update."""
    # Temporary buffer to store results from each batch before collation

    buffer_lists = {
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

    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        if batch is None:  # Handle potential error from collate_fn
            logger.info("Warning: Skipping None batch from dataloader.")
            continue
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"]  # List of strings

        # 1. Generate responses
        response_ids, response_mask = generate_responses(
            actor_model, tokenizer, prompt_ids, prompt_mask,
            gen_config)  # Shapes: (B, R_i), (B, R_i)

        # 2. Calculate stats (logprobs, values, etc.)
        stats = calculate_rollout_stats(
            actor_model, ref_model, tokenizer, prompt_ids, prompt_mask,
            response_ids, response_mask)  # Dict of tensors (B, R_i)

        # 3. Decode texts and calculate rewards
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        full_decoded_texts = tokenizer.batch_decode(full_ids,
                                                    skip_special_tokens=True)
        rewards = torch.tensor(
            [
                compute_gsm8k_reward(txt, gt)
                for txt, gt in zip(full_decoded_texts, ground_truths)
            ],
            dtype=torch.float32,
            device='cpu'  # Calculate reward on CPU
        )  # Shape: (B,)

        # 4. Append results to buffer lists (moving tensors to CPU)
        buffer_lists["prompt_input_ids"].append(prompt_ids.cpu())
        buffer_lists["prompt_attention_mask"].append(prompt_mask.cpu())
        buffer_lists["response_input_ids"].append(response_ids.cpu())
        buffer_lists["response_attention_mask"].append(response_mask.cpu())
        buffer_lists["logprobs"].append(stats["logprobs"].cpu())
        buffer_lists["ref_logprobs"].append(stats["ref_logprobs"].cpu())
        buffer_lists["values"].append(stats["values"].cpu())
        buffer_lists["rewards"].append(rewards)  # Already on CPU
        buffer_lists["full_texts"].extend(full_decoded_texts)
        buffer_lists["ground_truth_answers"].extend(ground_truths)

    # --- Collate the buffer lists into single tensors ---
    individual_lengths = []
    # Iterate through the list of batch masks
    for mask_batch in buffer_lists["response_attention_mask"]:
        if mask_batch.numel() > 0: # Ensure tensor is not empty
            # Sum along the sequence dimension (dim=1) to get lengths per sequence in the batch
            lengths_in_batch = mask_batch.sum(dim=1)
            # Extend the master list with individual lengths from this batch
            individual_lengths.extend(lengths_in_batch.cpu().numpy()) # Use .numpy() or .tolist()

    avg_resp_len = np.mean(individual_lengths) if individual_lengths else 0.0

    logger.info(f"Average response length for this rollout: {avg_resp_len:.2f}")
    collated_buffer = {}
    padding_value_map = {
        "input_ids":
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "attention_mask":
        0,
        "logprobs":
        0.0,
        "ref_logprobs":
        0.0,
        "values":
        0.0,
    }
    keys_to_pad_and_cat = [
        "prompt_input_ids", "prompt_attention_mask", "response_input_ids",
        "response_attention_mask", "logprobs", "ref_logprobs", "values"
    ]

    for key, data_list in buffer_lists.items():
        if key in ["full_texts", "ground_truth_answers"]:
            collated_buffer[key] = data_list  # Keep as list
        elif not data_list:
            collated_buffer[key] = torch.empty(0)  # Handle empty list
        elif key == "rewards":
            collated_buffer[key] = torch.cat(
                data_list, dim=0)  # Simple concat for 1D rewards
        elif key in keys_to_pad_and_cat:
            # Determine padding value
            pad_val = 0.0
            for suffix, val in padding_value_map.items():
                if key.endswith(suffix):
                    pad_val = val
                    break
            # Pad list elements to max seq len and concatenate
            collated_buffer[key] = pad_and_collate_tensors(
                data_list, padding_value=pad_val)
        else:
            logger.info(f"Warning: Unexpected key '{key}' in buffer collation.")
            collated_buffer[key] = data_list  # Keep as is

    return collated_buffer


# ==============================================================================
# == 5. PPO Update Phase Logic
# ==============================================================================


def run_ppo_update_epoch(
        actor_model: ActorModelWithValueHead,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        collated_buffer: Dict[
            str, torch.Tensor],  # Assumes tensors are on correct device
        cfg: DictConfig,
        device: torch.device) -> Dict[str, float]:
    """Runs one PPO epoch with mini-batch updates."""
    actor_model.train()
    aggregate_metrics = {}
    ppo_step_count = 0  # For gradient accumulation tracking

    # Load data from buffer (already collated)
    prompt_ids = collated_buffer["prompt_input_ids"]
    prompt_mask = collated_buffer["prompt_attention_mask"]
    response_ids = collated_buffer["response_input_ids"]
    response_mask = collated_buffer["response_attention_mask"]
    logprobs_old = collated_buffer["logprobs"]
    ref_logprobs = collated_buffer["ref_logprobs"]
    values_old = collated_buffer["values"]
    final_rewards = collated_buffer["rewards"]

    # Combine inputs for forward pass
    full_input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    full_attention_mask = torch.cat((prompt_mask, response_mask), dim=1)

    # --- Calculate Advantages and Returns (Once per epoch) ---
    with torch.no_grad():
        kl_per_token = logprobs_old - ref_logprobs
        kl_penalties = cfg.ppo.kl_coeff * kl_per_token
        advantages, returns = compute_gae_advantages(final_rewards,
                                                     kl_penalties, values_old,
                                                     response_mask,
                                                     cfg.ppo.gamma,
                                                     cfg.ppo.lam)

    # --- Mini-batch Loop ---
    num_samples = full_input_ids.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]

    for i in range(0, num_samples, cfg.ppo.mini_batch_size):
        ppo_step_count += 1
        batch_indices = indices[i:i + cfg.ppo.mini_batch_size]

        # Slice mini-batch data
        batch_full_ids = full_input_ids[batch_indices]
        batch_full_mask = full_attention_mask[batch_indices]
        batch_logprobs_old = logprobs_old[batch_indices]
        batch_values_old = values_old[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_returns = returns[batch_indices]
        batch_response_mask = response_mask[batch_indices]
        batch_response_tokens = response_ids[batch_indices]

        # --- PPO Mini-batch Update ---
        # 1. Forward Pass
        logits_new, values_new = actor_model(batch_full_ids,
                                             attention_mask=batch_full_mask)

        # 2. Extract response parts and calculate new logprobs
        start_idx = prompt_len - 1
        end_idx = prompt_len + resp_len - 1
        if start_idx < 0 or end_idx <= start_idx or end_idx > logits_new.shape[
                1]:
            print(
                f"Warning: Invalid slice indices in PPO update. Skipping mini-batch {i // cfg.ppo.mini_batch_size}."
            )
            continue  # Skip this mini-batch

        logits_new_resp = logits_new[:, start_idx:end_idx, :]
        values_new_resp = values_new[:, start_idx:end_idx]

        # Check shape consistency before gather
        if logits_new_resp.shape[1] != batch_response_tokens.shape[1]:
            print(
                f"Warning: Mismatch logits/response len in PPO update. Skipping mini-batch {i // cfg.ppo.mini_batch_size}."
            )
            continue

        logprobs_all_new = F.log_softmax(logits_new_resp, dim=-1)
        logprobs_new = torch.gather(
            logprobs_all_new, 2,
            batch_response_tokens.unsqueeze(-1)).squeeze(-1)

        # Apply mask
        logprobs_new = logprobs_new * batch_response_mask
        values_new_resp = values_new_resp * batch_response_mask

        # 3. Calculate Losses
        policy_loss, p_clip_frac, approx_kl = compute_policy_loss(
            logprobs_new, batch_logprobs_old, batch_advantages,
            batch_response_mask, cfg.ppo.clip_ratio)
        value_loss, v_clip_frac = compute_value_loss(values_new_resp,
                                                     batch_values_old,
                                                     batch_returns,
                                                     batch_response_mask,
                                                     cfg.ppo.clip_range_value)
        entropy_loss = compute_entropy_loss(logits_new_resp,
                                            batch_response_mask)

        # 4. Combine Losses
        loss = policy_loss + cfg.ppo.vf_coeff * value_loss + cfg.ppo.entropy_coeff * entropy_loss

        # 5. Backward Pass & Gradient Accumulation
        scaled_loss = loss / cfg.ppo.gradient_accumulation_steps
        scaled_loss.backward()

        found_none_grad = False
        found_zero_grad = False
        found_non_zero_grad = False

        # Iterate through all named parameters in your model
        # Ensure 'model' is your actual model variable
        for name, param in actor_model.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.grad is None:
                logger.warning(f"Gradient is None for trainable parameter: {name}")
                found_none_grad = True
            elif torch.all(param.grad == 0):
                # Check if the entire gradient tensor is zero
                found_zero_grad = True
            else:
                # Gradient exists and is non-zero
                found_non_zero_grad = True
                # --- Summary ---
                if not found_non_zero_grad:
                    logger.critical("No non-zero gradients found for any trainable parameter!")
                elif found_none_grad:
                    logger.warning("Some trainable parameters have None gradients (check graph connection).")
                    # Optional: Log if only zero grads were found (might indicate issues)
                elif found_zero_grad and not found_none_grad:
                    logger.info("All found gradients are zero (check loss function / saturation).")
                else:
                    pass

        # 6. Store Metrics
        current_metrics = {
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/entropy': -entropy_loss.item(),
            'loss/total': loss.item(),  # Store positive entropy
            'params/policy_clip_frac': p_clip_frac.item(),
            'params/value_clip_frac': v_clip_frac.item(),
            'params/approx_kl': approx_kl.item(),
        }
        for key, val in current_metrics.items():
            aggregate_metrics.setdefault(key, []).append(val)

        # 7. Optimizer Step (if accumulation cycle complete)
        if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
            # Check if grads exist before clipping/stepping
            grads_exist = any(p.grad is not None
                              for p in actor_model.parameters()
                              if p.requires_grad)
            if grads_exist:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_model.parameters(), max_norm=cfg.ppo.max_grad_norm)
                aggregate_metrics.setdefault('params/grad_norm',
                                             []).append(grad_norm.item())
                optimizer.step()
                lr_scheduler.step()
            # Zero grad AFTER potential step
            optimizer.zero_grad(set_to_none=True)

    # --- End of Epoch ---
    # Average metrics over the epoch
    final_metrics = {
        key: np.mean(val)
        for key, val in aggregate_metrics.items() if val
    }
    return final_metrics


def perform_ppo_updates(
        actor_model: ActorModelWithValueHead,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        rollout_buffer: Dict[str, Any],  # Can contain lists or tensors
        cfg: DictConfig,
        device: torch.device) -> Dict[str, float]:
    """Performs multiple PPO epochs on the collected rollout data."""
    # Move collated tensors from buffer to the training device
    # This assumes collation happened correctly and produced tensors
    try:
        buffer_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in rollout_buffer.items()
        }
    except AttributeError as e:
        print(
            f"Error moving buffer to device, likely due to non-tensor data: {e}"
        )
        # Print buffer keys and types for debugging
        logger.info("Rollout Buffer Contents (Keys and Types):")
        for k, v in rollout_buffer.items():
            logger.info(f"  {k}: {type(v)}")
        return {}  # Cannot proceed

    # Basic validation after moving to device
    if "response_input_ids" not in buffer_on_device or \
       not isinstance(buffer_on_device["response_input_ids"], torch.Tensor) or \
       buffer_on_device["response_input_ids"].numel() == 0:
        print(
            "Warning: No response tokens found in buffer on device. Skipping PPO update."
        )
        return {}

    all_epoch_metrics = {}
    for ppo_epoch in range(cfg.ppo.epochs):
        epoch_metrics = run_ppo_update_epoch(actor_model, optimizer, lr_scheduler,
                                             buffer_on_device, cfg, device)
        # Aggregate metrics across epochs (e.g., average or store last)
        # Here, we just store the metrics from the last epoch for simplicity
        all_epoch_metrics = epoch_metrics  # Overwrite with last epoch's metrics

    return all_epoch_metrics


# ==============================================================================
# == 6. Training Setup and Orchestration
# ==============================================================================


def setup_training(cfg: DictConfig) -> Tuple[torch.device, str]:
    """Sets random seeds, device, and output directory."""
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.training.seed)
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        if cfg.training.device == "cuda":
            logger.info("Warning: CUDA requested but unavailable, using CPU.")
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    # Save config (optional, can be done in main train loop)
    # try: OmegaConf.save(cfg, os.path.join(output_dir, "effective_config.yaml"))
    # except Exception as e: logger.info(f"Error saving config: {e}")

    return device, output_dir


def load_models_and_tokenizer(
    cfg: DictConfig, device: torch.device
) -> Tuple[ActorModelWithValueHead, PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads tokenizer, actor model (with value head), and reference model."""
    logger.info(f"Loading tokenizer: {cfg.model.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    # --- Set Padding Token ---
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.info("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Explicitly set padding side (optional, default is often right)
    # tokenizer.padding_side = 'left' # Uncomment to use left padding

    logger.info(f"Loading models: {cfg.model.name}")
    # --- Model Kwargs (dtype, quantization, etc.) ---
    model_kwargs = {}
    model_dtype_str = cfg.model.get("torch_dtype", "auto")
    if model_dtype_str != "auto":
        try:
            model_kwargs["torch_dtype"] = getattr(torch, model_dtype_str)
        except AttributeError:
            print(
                f"Warning: Invalid torch_dtype '{model_dtype_str}'. Using auto."
            )
    if cfg.model.get("trust_remote_code", False):
        model_kwargs["trust_remote_code"] = True
    if cfg.model.get("quantization"):
        q_cfg = cfg.model.quantization
        logger.info(f"Applying quantization: {q_cfg}")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=q_cfg.get("load_in_8bit", False),
            load_in_4bit=q_cfg.get("load_in_4bit", False),
            bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, q_cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant",
                                                False),
        )

    # --- Load Actor Model ---
    actor_model = ActorModelWithValueHead(cfg.model.name, **model_kwargs)
    if not cfg.model.get("quantization"):
        actor_model.to(device)  # Move if not quantized
    # Ensure pad token ID is set in model config
    if actor_model.config.pad_token_id is None:
        actor_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Actor model loaded.")
    if cfg.training.get("gradient_checkpointing", False):
        try:
            # Enable on the base model wrapped by ActorModelWithValueHead
            actor_model.base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for actor model.")
        except AttributeError:
             print("Warning: Could not enable gradient checkpointing. Method not found on base_model.")
 
    # --- Load Reference Model ---
    ref_model_kwargs = model_kwargs.copy()
    ref_model_kwargs.pop("quantization_config",
                         None)  # No quantization for ref model
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                     **ref_model_kwargs)
    ref_model.to(device)
    if ref_model.config.pad_token_id is None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    logger.info("Reference model loaded and frozen.")

    return actor_model, ref_model, tokenizer


def load_and_preprocess_dataset(cfg: DictConfig,
                                tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Loads the dataset and preprocesses it."""
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    try:
        dataset = load_dataset(cfg.dataset.name,
                               cfg.dataset.get("config"),
                               split=cfg.dataset.split)
    except Exception as e:
        logger.info(f"Error loading dataset '{cfg.dataset.name}': {e}")
        raise  # Re-raise critical error

    # --- Subsetting ---
    num_samples = cfg.training.get("num_samples")
    if num_samples is not None and num_samples > 0 and num_samples <= len(
            dataset):
        logger.info(f"Subsetting dataset to {num_samples} samples.")
        dataset = dataset.select(range(num_samples))

    # --- Preprocessing Function ---
    def preprocess_function(example):
        try:
            example["prompt"] = cfg.dataset.prompt_format.format(
                question=example["question"])
            example["ground_truth_answer"] = example["answer"].split(
                "####")[-1].strip()
        except KeyError as e:
            print(
                f"Error processing example: Missing key {e}. Skipping prompt/answer."
            )
            example["prompt"] = ""
            example["ground_truth_answer"] = ""
        # Tokenize prompt only (no padding here)
        tokenized_prompt = tokenizer(example["prompt"],
                                     max_length=cfg.dataset.max_prompt_length,
                                     truncation=True,
                                     padding=False)
        example["input_ids"] = tokenized_prompt["input_ids"]
        example["attention_mask"] = tokenized_prompt["attention_mask"]
        return example

    # --- Apply Preprocessing ---
    try:
        processed_dataset = dataset.map(
            preprocess_function,
            remove_columns=dataset.column_names  # Keep only processed columns
        )
        processed_dataset.set_format(type="torch")  # Set format for DataLoader
        logger.info(f"Dataset preprocessed. Samples: {len(processed_dataset)}")
        return processed_dataset
    except Exception as e:
        logger.info(f"Error during dataset mapping: {e}")
        raise  # Re-raise critical error


def setup_optimizer(cfg: DictConfig,
                    model: nn.Module) -> torch.optim.Optimizer:
    """Sets up the optimizer based on configuration."""
    use_8bit = cfg.ppo.get("use_8bit_adam", False)
    lr = cfg.ppo.learning_rate

    if use_8bit and bnb_available and isinstance(
            next(model.parameters()).device, torch.device) and next(
                model.parameters()).device.type == "cuda":
        # Check for quantization conflict (optional, depends on specific use case)
        is_quantized = hasattr(model, 'quantization_config') and \
                       (model.quantization_config.load_in_8bit or model.quantization_config.load_in_4bit)
        if is_quantized:
            print(
                "Warning: Using 8-bit AdamW with a quantized model. Consider standard AdamW."
            )
            optimizer = AdamW(model.parameters(), lr=lr)
        else:
            logger.info("Using 8-bit AdamW Optimizer (bitsandbytes)")
            optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        if use_8bit:
            print(
                "Info: 8-bit Adam not used (requirements not met). Using standard AdamW."
            )
        else:
            logger.info("Using standard AdamW Optimizer")
        optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        cfg.ppo.scheduler, optimizer, cfg.ppo.warmup_steps,
        num_training_steps=(
            cfg.training.total_ppo_steps *
            ((cfg.ppo.rollout_samples // cfg.ppo.mini_batch_size) //
             cfg.ppo.gradient_accumulation_steps) * cfg.ppo.epochs),
        scheduler_specific_kwargs={'min_lr' : cfg.ppo.min_lr})
    return optimizer, lr_scheduler


def create_generation_config(
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase) -> GenerationConfig:
    """Creates the GenerationConfig object."""
    return GenerationConfig(max_new_tokens=cfg.generation.max_new_tokens,
                            min_new_tokens=cfg.generation.min_new_tokens,
                            temperature=cfg.generation.temperature,
                            top_k=cfg.generation.top_k,
                            top_p=cfg.generation.top_p,
                            do_sample=cfg.generation.do_sample,
                            pad_token_id=tokenizer.pad_token_id)


def save_model(model: nn.Module, tokenizer: PreTrainedTokenizerBase,
               save_path: str):
    """Saves the model and tokenizer."""
    if not accelerator.is_main_process:
        return
    logger.info(f"Saving model checkpoint to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    try:
        # Handle potential model wrappers (like ActorModelWithValueHead)
        unwrapped_model = getattr(model, "base_model", model)
        # Handle PEFT model saving if applicable in the future
        # if hasattr(unwrapped_model, 'save_pretrained'):
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved.")
    except Exception as e:
        logger.info(f"Error saving model: {e}")


# ==============================================================================
# == 7. Main Training Orchestration
# ==============================================================================


def train(cfg: DictConfig):
    """Main PPO training loop."""
    # --- 1. Initial Setup ---
    device, output_dir = setup_training(cfg)

    # Save final config after setup
    OmegaConf.save(cfg, os.path.join(output_dir, "effective_config.yaml"))

    if cfg.wandb.report_to_wandb and accelerator.is_main_process:
        # Initialise your wandb run, passing wandb parameters and any config information
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.wandb.get("name", None)}}
        )

    # --- 2. Load Models and Tokenizer ---
    try:
        actor_model, ref_model, tokenizer = load_models_and_tokenizer(
            cfg, device)
    except Exception as e:
        logger.info(f"Failed to load models/tokenizer: {e}")
        return  # Cannot proceed

    # --- 3. Load and Preprocess Dataset ---
    try:
        processed_dataset = load_and_preprocess_dataset(cfg, tokenizer)
    except Exception as e:
        logger.info(f"Failed to load/preprocess dataset: {e}")
        return  # Cannot proceed

    # --- 4. Setup Optimizer ---
    optimizer, lr_scheduler = setup_optimizer(cfg, actor_model)

    # --- 5. Generation Config ---
    gen_config = create_generation_config(cfg, tokenizer)

    # --- 6. Collate Function for DataLoader ---
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        try:
            # Use tokenizer.pad (respects tokenizer.padding_side)
            padded_inputs = tokenizer.pad({"input_ids": input_ids},
                                          padding='longest',
                                          return_tensors="pt",
                                          return_attention_mask=True)
        except Exception as e:
            logger.info(f"Error during tokenizer.pad in collate_fn: {e}")
            return None  # Signal error to dataloader loop
        ground_truths = [item['ground_truth_answer'] for item in batch]
        return {
            "prompt_input_ids": padded_inputs["input_ids"],
            "prompt_attention_mask": padded_inputs["attention_mask"],
            "ground_truth_answers": ground_truths
        }

    # --- 7. Main PPO Loop ---
    logger.info("\n--- Starting PPO Training ---")
    for ppo_step in range(cfg.training.total_ppo_steps):
        print(
            f"\n===== PPO Step {ppo_step + 1}/{cfg.training.total_ppo_steps} ====="
        )

        # --- Phase 1: Rollout ---
        logger.info("Phase 1: Generating Rollouts...")
        prompt_dataloader = DataLoader(processed_dataset.shuffle(
            seed=cfg.training.seed).select(range(cfg.ppo.rollout_samples)),
                                       batch_size=cfg.ppo.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn)
        try:
            rollout_buffer = perform_rollouts(actor_model, ref_model,
                                              tokenizer, prompt_dataloader,
                                              gen_config, device)
        except Exception as e:
            logger.info(f"Error during rollout phase: {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip to next PPO step

        # Validate rollout buffer
        if not rollout_buffer or "rewards" not in rollout_buffer or \
           not isinstance(rollout_buffer["rewards"], torch.Tensor) or \
           rollout_buffer["rewards"].numel() == 0:
            print(
                "Warning: Invalid rollout buffer generated. Skipping update.")
            continue

        avg_reward = rollout_buffer["rewards"].mean().item()
        num_rollouts = rollout_buffer["rewards"].shape[0]
        print(
            f"Rollout complete ({num_rollouts} samples). Average reward: {avg_reward:.4f}"
        )

        # --- Phase 2: Update ---
        logger.info("Phase 2: Performing PPO Updates...")
        metrics = perform_ppo_updates(actor_model, optimizer, lr_scheduler,
                                          rollout_buffer, cfg, device)
        log_data = {}
        log_data.update(metrics)
        # Log metrics
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Update Metrics (Avg over Epoch): {log_str}")
        logger.info(f"  Rollout Reward (for this step): {avg_reward:.4f}")
        log_data["rollout/reward_mean"] = avg_reward # Add rollout reward
        if cfg.wandb.report_to_wandb:
            wandb.log(log_data, step=ppo_step)

        # --- Phase 3: Save Checkpoint ---
        if (ppo_step + 1) % cfg.training.save_interval == 0:
            save_model(actor_model, tokenizer,
                       os.path.join(output_dir, f"step_{ppo_step + 1}"))


    if cfg.wandb.report_to_wandb:
        wandb.finish()
    # --- 8. Final Save ---
    logger.info("\n--- PPO Training Finished ---")
    save_model(actor_model, tokenizer, os.path.join(output_dir, "final"))


# ==============================================================================
# == 8. Command-Line Interface Logic
# ==============================================================================


def load_config_with_cli_overrides() -> DictConfig:
    """Loads OmegaConf config, handling defaults and CLI overrides."""
    parser = argparse.ArgumentParser(description="PPO RL Trainer")
    parser.add_argument("--config", type=str)
    parser.add_argument("overrides",
                        nargs="*",
                        help="Key=value config overrides")
    args = parser.parse_args()

    config_path = args.config 

    if not os.path.exists(config_path):
        print(
            f"Error: Config file '{args.config_name}' not found in '{config_dir_abs}' or '{args.config_path}'."
        )
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Handle 'defaults' for base config merging (simplified)
    if 'defaults' in cfg:
        base_conf_path = cfg.defaults[0]
        logger.info(f"Loading base config from: {base_conf_path}")
        base_cfg = OmegaConf.load(base_conf_path)
        cfg = OmegaConf.merge(base_cfg, cfg)  # Merge base first

    # Apply CLI overrides
    if args.overrides:
        logger.info(f"Applying overrides: {args.overrides}")
        cli_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    # Resolve interpolations
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        logger.warn(f"Warning: Config resolution error: {e}")

    logger.info("--------- Final Configuration ---------")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))  # Print resolved config
    logger.info("---------------------------------------")
    return cfg


# ==============================================================================
# == 9. Entry Point
# ==============================================================================

if __name__ == "__main__":
    config = load_config_with_cli_overrides()
    train(config)
