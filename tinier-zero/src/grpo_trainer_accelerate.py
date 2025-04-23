# -*- coding: utf-8 -*-
"""
GRPO Trainer script adapted for distributed training using Hugging Face Accelerate.
"""
import wandb
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    import bitsandbytes.optim as bnb_optim
    bnb_available = True
except ImportError:
    bnb_available = False

from ppo_trainer_solutions import (
    load_and_preprocess_dataset,
    pad_and_collate_tensors,
)

from grpo_trainer import (
    create_generation_config_grpo
)

from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
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
import time
import pickle

# --- Import Accelerate ---
from accelerate import Accelerator, DistributedDataParallelKwargs
import accelerate

# ==============================================================================
# == 1. Helper Functions (Unchanged from previous GRPO version)
# ==============================================================================

def masked_mean(tensor: torch.Tensor,
                mask: Optional[torch.Tensor],
                dim: Optional[int] = None) -> torch.Tensor:
    """Calculates mean of tensor elements specified by mask."""
    if mask is None:
        return torch.mean(tensor, dim=dim)
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)
    masked_tensor = torch.where(
        mask, tensor,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    # Add epsilon to denominator for stability
    mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8)
    return mean

def masked_whiten(tensor: torch.Tensor,
                  mask: Optional[torch.Tensor],
                  shift_mean: bool = True) -> torch.Tensor:
    """Whitens the tensor values specified by the mask."""
    if mask is None or mask.numel() == 0 or mask.sum() == 0:
        # logger.warning("Mask is empty or all zeros in masked_whiten. Returning original tensor.")
        # Return tensor of zeros matching shape if mask is all zero, else original if mask is None
        if mask is None: return tensor
        else: return torch.zeros_like(tensor)

    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_variance = torch.where(
        mask, (tensor - mean)**2,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    variance = masked_mean(masked_tensor_variance, mask, dim=None)
    std = torch.sqrt(variance + 1e-8)

    if std < 1e-8:
         # logger.warning(f"Standard deviation is near zero ({std.item()}) in masked_whiten. Returning un-whitened tensor.")
         return torch.where(mask, tensor - mean if shift_mean else tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(
        mask, whitened,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))

def extract_gsm8k_solution(solution_str: str) -> Optional[str]:
    """Extracts the numerical answer from the #### format."""
    solution_match = re.search(r"####\s*([-+]?\s*[\d\.\,]+)(?:\s|$)+", solution_str)
    if solution_match:
        potential_answer_str = solution_match.group(1).replace(',', '').replace(' ', '')
        try: float(potential_answer_str); return potential_answer_str
        except ValueError: return None
    else:
        answer_list = re.findall(r"([-+]?\s*[\d\.\,]+)(?:\s|$)+", solution_str)
        if answer_list:
            final_answer_str = answer_list[-1].replace(',', '').replace(' ', '')
            try: float(final_answer_str); return final_answer_str
            except ValueError: return None
        else: return None

def compute_gsm8k_reward(generated_text: str, ground_truth_str: str) -> float:
    """Computes reward: 1.0 if extracted answer matches ground truth, 0 otherwise."""
    extracted_answer_str = extract_gsm8k_solution(generated_text)
    if extracted_answer_str is None: return 0.0
    try:
        extracted_answer = float(extracted_answer_str)
        ground_truth = float(ground_truth_str)
        return 1.0 if math.isclose(extracted_answer, ground_truth, rel_tol=1e-4) else 0.0
    except ValueError: return 0.0

def pad_and_collate_tensors(tensor_list: List[torch.Tensor],
                            padding_value: float = 0.0) -> torch.Tensor:
    """Pads and collates tensors. Handles empty lists."""
    if not tensor_list:
        logger.warning("pad_and_collate_tensors received an empty list.")
        # Need a default tensor type/device if list is empty
        # This part might need adjustment based on expected empty behavior
        return torch.empty(0)

    non_empty_tensors = [t for t in tensor_list if t.numel() > 0 and t.dim() > 1 and t.shape[1] > 0]

    if not non_empty_tensors:
        total_batch_size = sum(t.shape[0] for t in tensor_list)
        if not tensor_list: return torch.empty(0) # Truly empty
        original_shape = tensor_list[0].shape
        logger.warning("All tensors in list were empty or had 0 sequence length.")
        return torch.empty((total_batch_size, 0) + original_shape[2:],
                           dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    max_len = max([t.shape[1] for t in non_empty_tensors])
    padded_list = []
    for t in tensor_list:
        if t.numel() == 0 or t.dim() <= 1: # Handle empty or 1D tensors
             # Pad appropriately based on expected dimensions if needed
             # For simplicity, assuming we pad to match others if it was meant to have seq dim
             padded_t = torch.full((t.shape[0], max_len) + t.shape[2:], padding_value, dtype=t.dtype, device=t.device) if t.dim() > 1 else t
             padded_list.append(padded_t)
             continue

        current_len = t.shape[1]
        padding_needed = max_len - current_len
        if padding_needed > 0:
            pad_dims = []
            for _ in range(t.dim() - 2): pad_dims.extend([0, 0])
            pad_dims.extend([0, padding_needed]) # Pad sequence dim (right)
            pad_dims.extend([0, 0]) # Don't pad batch dim
            pad_tuple = tuple(pad_dims)
            t = F.pad(t, pad_tuple, mode='constant', value=padding_value)
        padded_list.append(t)

    # Ensure all tensors are on the same device before cat
    target_device = padded_list[0].device
    padded_list = [p.to(target_device) for p in padded_list]
    return torch.cat(padded_list, dim=0)

# ==============================================================================
# == 2. Core GRPO Algorithm Components (Unchanged)
# ==============================================================================

def compute_grpo_advantages(
    rewards: torch.Tensor, kl_penalties: torch.Tensor, response_mask: torch.Tensor,
    group_size: int, kl_coeff: float
) -> torch.Tensor:
    """Computes GRPO advantages by normalizing rewards within each group."""
    # (Code identical to previous version)
    with torch.no_grad():
        num_samples = rewards.shape[0]
        if num_samples == 0:
            logger.warning("compute_grpo_advantages received empty rewards tensor.")
            return torch.zeros_like(kl_penalties)

        if num_samples % group_size != 0:
             logger.warning(f"Total samples ({num_samples}) not divisible by group_size ({group_size}).")
             num_prompts = num_samples // group_size # Proceed with integer division
             # Consider trimming last incomplete group if necessary upstream
        else:
             num_prompts = num_samples // group_size

        rewards = rewards.to(kl_penalties.device)

        if rewards.dim() == 1:
            if num_prompts * group_size > num_samples: # Check if truncation happened
                logger.warning(f"Reshaping rewards based on calculated num_prompts ({num_prompts}) due to non-divisible samples.")
                rewards = rewards[:num_prompts * group_size] # Trim rewards
            elif num_prompts * group_size != num_samples:
                 logger.error(f"Cannot reshape rewards ({num_samples},) to ({num_prompts}, {group_size}).")
                 return torch.zeros_like(kl_penalties)
            rewards = rewards.view(num_prompts, group_size)
        elif rewards.shape[0] != num_prompts: # Check if already shaped but wrong size
             logger.error(f"Unexpected rewards shape: {rewards.shape}. Expected ({num_prompts}, {group_size}).")
             return torch.zeros_like(kl_penalties)

        mean_kl_penalty_per_seq = masked_mean(kl_penalties, response_mask, dim=1)

        if num_prompts * group_size > mean_kl_penalty_per_seq.shape[0]:
            logger.warning(f"Reshaping KL penalties based on calculated num_prompts ({num_prompts}) due to non-divisible samples.")
            mean_kl_penalty_per_seq = mean_kl_penalty_per_seq[:num_prompts * group_size]
        elif num_prompts * group_size != mean_kl_penalty_per_seq.shape[0]:
            logger.error(f"Cannot reshape KL penalties ({mean_kl_penalty_per_seq.shape[0]},) to ({num_prompts}, {group_size}).")
            return torch.zeros_like(kl_penalties)

        mean_kl_penalty_per_seq = mean_kl_penalty_per_seq.view(num_prompts, group_size)
        adjusted_rewards = rewards - kl_coeff * mean_kl_penalty_per_seq
        group_mean = adjusted_rewards.mean(dim=1, keepdim=True)
        group_std = adjusted_rewards.std(dim=1, keepdim=True)
        advantages = (adjusted_rewards - group_mean) / (group_std + 1e-8)
        advantages = advantages.view(num_prompts * group_size, 1)
        advantages = advantages.expand(-1, response_mask.shape[1])
        advantages = advantages * response_mask.float()
    return advantages


def compute_grpo_policy_loss(
    log_probs_new: torch.Tensor, log_probs_old: torch.Tensor, advantages: torch.Tensor,
    response_mask: torch.Tensor, clip_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes GRPO policy loss (clipped surrogate objective)."""
    # (Code identical to previous version, including error handling)
    with torch.no_grad():
        mask = response_mask.bool()
        if advantages.shape != log_probs_old.shape:
            logger.error(f"Shape mismatch in compute_grpo_policy_loss: advantages {advantages.shape}, log_probs {log_probs_old.shape}")
            dummy_loss = torch.tensor(0.0, device=log_probs_new.device, requires_grad=True)
            dummy_frac = torch.tensor(0.0, device=log_probs_new.device)
            dummy_kl = torch.tensor(0.0, device=log_probs_new.device)
            return dummy_loss, dummy_frac, dummy_kl
    log_ratio = (log_probs_new - log_probs_old).clamp(-20, 20)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2), mask)
    with torch.no_grad():
        clip_frac = masked_mean(torch.gt(torch.abs(ratio - 1.0), clip_ratio).float(), mask)
        approx_kl = masked_mean(log_probs_old - log_probs_new, mask)
    return policy_loss, clip_frac, approx_kl


def compute_grpo_entropy_loss(logits_new: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Computes entropy loss."""
    # (Code identical to previous version)
    mask = response_mask.bool()
    dist = torch.distributions.Categorical(logits=logits_new.float())
    entropy = dist.entropy()
    entropy_loss = -masked_mean(entropy, mask)
    return entropy_loss

# ==============================================================================
# == 3. Actor Model Definition (No Value Head)
# ==============================================================================
# Using AutoModelForCausalLM directly

# ==============================================================================
# == 4. Rollout Phase Logic (Modified for Group Generation & Accelerate)
# ==============================================================================

def generate_responses_grouped(
    model: PreTrainedModel, # Standard LM model
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_config: GenerationConfig,
    group_size: int,
    accelerator: Accelerator # Added accelerator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates G responses for each prompt in a batch, handles device placement."""
    model.eval()
    batch_size = prompt_ids.shape[0]

    current_gen_config = GenerationConfig(**gen_config.to_dict())
    current_gen_config.do_sample = True

    with torch.no_grad():
        # Input needs repeating for generation: (B, L) -> (B*G, L)
        # Ensure prompt_ids/mask are on the correct device before repeat
        expanded_prompt_ids = prompt_ids.to(accelerator.device).repeat_interleave(group_size, dim=0)
        expanded_prompt_mask = prompt_mask.to(accelerator.device).repeat_interleave(group_size, dim=0)

        # --- Use accelerator.unwrap_model for generation ---
        # This ensures generation works correctly with DDP/FSDP wrappers
        unwrapped_model = accelerator.unwrap_model(model)

        generated_output = unwrapped_model.generate(
            input_ids=expanded_prompt_ids,
            attention_mask=expanded_prompt_mask,
            generation_config=current_gen_config,
            pad_token_id=tokenizer.pad_token_id
        )

        prompt_len = prompt_ids.shape[1]
        if generated_output.shape[1] <= prompt_len:
             logger.warning(f"Generated output length ({generated_output.shape[1]}) not greater than prompt length ({prompt_len}).")
             resp_shape = (expanded_prompt_ids.shape[0], 0)
             return torch.empty(resp_shape, dtype=torch.long, device=accelerator.device), \
                    torch.empty(resp_shape, dtype=torch.long, device=accelerator.device)

        response_ids = generated_output[:, prompt_len:]
        response_mask = (response_ids != tokenizer.pad_token_id).long()

    # Return tensors on the accelerator's device
    return response_ids, response_mask


def calculate_rollout_stats_grpo(
    actor_model: PreTrainedModel,
    ref_model: PreTrainedModel, # Already on accelerator.device
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,      # Shape (batch, prompt_len)
    prompt_mask: torch.Tensor,     # Shape (batch, prompt_len)
    response_ids: torch.Tensor,    # Shape (batch*G, resp_len)
    response_mask: torch.Tensor,   # Shape (batch*G, resp_len)
    group_size: int,
    accelerator: Accelerator # Added accelerator
) -> Dict[str, torch.Tensor]:
    """Calculates logprobs, ref_logprobs for grouped responses on correct device."""
    # Models are already wrapped or on the correct device
    actor_model.eval()
    ref_model.eval() # ref_model is not wrapped by accelerator but moved to device

    batch_size = prompt_ids.shape[0] # B
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]
    expected_batch_group_size = batch_size * group_size # B*G

    if response_ids.shape[0] != expected_batch_group_size:
         logger.error(f"Batch size mismatch in calculate_rollout_stats_grpo: response_ids has {response_ids.shape[0]}, expected {expected_batch_group_size}")
         return {"logprobs": torch.empty(0), "ref_logprobs": torch.empty(0)}

    # Expand prompts on the correct device
    expanded_prompt_ids = prompt_ids.to(accelerator.device).repeat_interleave(group_size, dim=0)
    expanded_prompt_mask = prompt_mask.to(accelerator.device).repeat_interleave(group_size, dim=0)

    # Ensure response tensors are also on the correct device
    response_ids = response_ids.to(accelerator.device)
    response_mask = response_mask.to(accelerator.device)

    full_ids = torch.cat((expanded_prompt_ids, response_ids), dim=1)
    full_mask = torch.cat((expanded_prompt_mask, response_mask), dim=1)
    full_len = full_ids.shape[1]

    with torch.no_grad():
        # Actor forward pass (uses the model prepared by accelerator)
        actor_logits = actor_model(full_ids, attention_mask=full_mask).logits
        # Reference forward pass (uses model manually moved to accelerator.device)
        ref_logits = ref_model(full_ids, attention_mask=full_mask).logits

        start_idx = prompt_len - 1
        end_idx = full_len - 1

        if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
            logger.warning(f"Invalid slice range or empty response in calculate_rollout_stats_grpo.")
            empty_shape = (expected_batch_group_size, 0)
            return {
                "logprobs": torch.empty(empty_shape, dtype=torch.float, device=accelerator.device),
                "ref_logprobs": torch.empty(empty_shape, dtype=torch.float, device=accelerator.device),
            }

        logits_resp = actor_logits[:, start_idx:end_idx, :]
        ref_logits_resp = ref_logits[:, start_idx:end_idx, :]
        target_ids = response_ids

        current_resp_len = logits_resp.shape[1]
        if current_resp_len != target_ids.shape[1]:
             logger.warning(f"Logits length ({current_resp_len}) != Target IDs length ({target_ids.shape[1]}). Truncating.")
             min_len = min(current_resp_len, target_ids.shape[1])
             logits_resp = logits_resp[:, :min_len, :]
             ref_logits_resp = ref_logits_resp[:, :min_len, :]
             target_ids = target_ids[:, :min_len]
             response_mask_adjusted = response_mask[:,:min_len]
        else:
             response_mask_adjusted = response_mask

        logprobs_all = F.log_softmax(logits_resp, dim=-1)
        ref_logprobs_all = F.log_softmax(ref_logits_resp, dim=-1)
        logprobs = torch.gather(logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = torch.gather(ref_logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)

        logprobs = logprobs * response_mask_adjusted
        ref_logprobs = ref_logprobs * response_mask_adjusted

    # Return tensors on the accelerator's device
    return {
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
    }


def perform_rollouts_grpo(
    actor_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_dataloader: DataLoader, # This will be prepared by accelerator
    gen_config: GenerationConfig,
    group_size: int,
    accelerator: Accelerator # Added accelerator
) -> Dict[str, Any]:
    """Generates groups of responses and computes stats for GRPO update using Accelerate."""
    rollout_start_time = time.time()
    buffer_lists = {
        "prompt_input_ids": [], "prompt_attention_mask": [],
        "response_input_ids": [], "response_attention_mask": [],
        "logprobs": [], "ref_logprobs": [],
        "rewards": [], "full_texts": [], "ground_truth_answers": []
    }
    timing_data = {"gen_time": 0.0, "stats_time": 0.0, "cpu_time": 0.0}

    # Only show progress bar on the main process
    iterable_dataloader = prompt_dataloader
    if accelerator.is_main_process:
        iterable_dataloader = tqdm(prompt_dataloader, desc="GRPO Rollout", leave=False)

    for batch_idx, batch in enumerate(iterable_dataloader):
        # Batch data is already on the correct device thanks to accelerator.prepare(dataloader)
        if batch is None:
            logger.warning(f"Skipping None batch from dataloader (Index: {batch_idx}).")
            continue
        cpu_start_time = time.time()
        # No need for .to(device) here
        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        ground_truths = batch["ground_truth_answers"] # Still a list of strings
        cpu_prep_time = time.time() - cpu_start_time

        # 1. Generate G responses per prompt
        gen_start_time = time.time()
        response_ids, response_mask = generate_responses_grouped(
            actor_model, tokenizer, prompt_ids, prompt_mask, gen_config, group_size, accelerator
        )
        timing_data["gen_time"] += (time.time() - gen_start_time)

        if response_ids.numel() == 0:
             logger.warning(f"generate_responses_grouped returned empty response_ids for batch {batch_idx}.")
             continue

        # 2. Calculate stats (logprobs, ref_logprobs)
        stats_start_time = time.time()
        # Pass accelerator to stats function
        stats = calculate_rollout_stats_grpo(
            actor_model, ref_model, tokenizer,
            prompt_ids, prompt_mask, response_ids, response_mask, group_size, accelerator
        )
        timing_data["stats_time"] += (time.time() - stats_start_time)

        if not stats or stats["logprobs"].numel() == 0:
             logger.warning(f"calculate_rollout_stats_grpo returned empty stats for batch {batch_idx}.")
             continue

        # 3. Decode texts and calculate rewards (CPU work)
        cpu_work_start_time = time.time()
        # Gather tensors from all processes for decoding if needed, or decode locally?
        # For simplicity here, assume we decode locally and gather rewards later if needed.
        # Need to move generated IDs to CPU for decoding.
        response_ids_cpu = response_ids.cpu()
        prompt_ids_cpu = prompt_ids.cpu() # Also need prompts on CPU
        
        expanded_prompt_ids_cpu = prompt_ids_cpu.repeat_interleave(group_size, dim=0)
        if expanded_prompt_ids_cpu.shape[0] != response_ids_cpu.shape[0]:
            logger.error(f"Shape mismatch before decoding (CPU): expanded_prompts {expanded_prompt_ids_cpu.shape[0]}, response_ids {response_ids_cpu.shape[0]}.")
            continue
        full_ids_cpu = torch.cat((expanded_prompt_ids_cpu, response_ids_cpu), dim=1)
        full_decoded_texts = tokenizer.batch_decode(full_ids_cpu, skip_special_tokens=True)
        
        expanded_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]
        
        rewards = torch.tensor(
            [compute_gsm8k_reward(txt, gt) for txt, gt in zip(full_decoded_texts, expanded_ground_truths)],
            dtype=torch.float32, device='cpu' # Rewards computed and kept on CPU
        )

        # 4. Append results to buffer lists (keep on CPU for now)
        buffer_lists["prompt_input_ids"].append(prompt_ids.cpu())
        buffer_lists["prompt_attention_mask"].append(prompt_mask.cpu())
        buffer_lists["response_input_ids"].append(response_ids.cpu())
        buffer_lists["response_attention_mask"].append(response_mask.cpu())
        buffer_lists["logprobs"].append(stats["logprobs"].cpu())
        buffer_lists["ref_logprobs"].append(stats["ref_logprobs"].cpu())
        buffer_lists["rewards"].append(rewards)
        buffer_lists["full_texts"].extend(full_decoded_texts)
        buffer_lists["ground_truth_answers"].extend(expanded_ground_truths)
        timing_data["cpu_time"] += (time.time() - cpu_work_start_time) + cpu_prep_time

    # --- Collate buffer lists on CPU ---
    # (Collation code remains largely the same, operating on CPU tensors)
    collation_start_time = time.time()
    collated_buffer = {}
    if not buffer_lists["prompt_input_ids"]:
         logger.warning("Rollout phase produced no valid data. Returning empty buffer.")
         return {}

    num_prompts_processed = sum(p.shape[0] for p in buffer_lists["prompt_input_ids"])
    num_total_samples = sum(r.shape[0] for r in buffer_lists["response_input_ids"])
    if accelerator.is_main_process: # Log only once
        logger.info(f"Collating buffer: {num_prompts_processed} prompts, {num_total_samples} total samples.")

    padding_value_map = {
        "input_ids": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "attention_mask": 0, "logprobs": 0.0, "ref_logprobs": 0.0,
    }
    keys_to_pad_and_cat = [
        "prompt_input_ids", "prompt_attention_mask",
        "response_input_ids", "response_attention_mask",
        "logprobs", "ref_logprobs"
    ]
    for key, data_list in buffer_lists.items():
        if key in ["full_texts", "ground_truth_answers"]:
                collated_buffer[key] = data_list
        elif not data_list:
                collated_buffer[key] = torch.empty(0)
        elif key == "rewards":
                collated_buffer[key] = torch.cat(data_list, dim=0)
        elif key in keys_to_pad_and_cat:
                pad_val = 0.0
                for suffix, val in padding_value_map.items():
                    if key.endswith(suffix): pad_val = val; break
                collated_buffer[key] = pad_and_collate_tensors(data_list, padding_value=pad_val)
        else: logger.warning(f"Unexpected key '{key}' during buffer collation.")
    collation_time = time.time() - collation_start_time

    # --- Calculate Average Response Length ---
    individual_lengths = []
    for mask_batch in buffer_lists["response_attention_mask"]:
        if mask_batch.numel() > 0:
            lengths_in_batch = mask_batch.sum(dim=1)
            individual_lengths.extend(lengths_in_batch.cpu().numpy())
    avg_resp_len = np.mean(individual_lengths) if individual_lengths else 0.0
    if accelerator.is_main_process:
        logger.info(f"Average response length per sequence for this rollout: {avg_resp_len:.2f}")

    rollout_duration = time.time() - rollout_start_time
    collated_buffer["avg_response_length"] = avg_resp_len
    collated_buffer["rollout_duration_seconds"] = rollout_duration
    collated_buffer["timing/total_gen_time"] = timing_data["gen_time"]
    collated_buffer["timing/total_stats_time"] = timing_data["stats_time"]
    collated_buffer["timing/total_cpu_time"] = timing_data["cpu_time"]
    collated_buffer["timing/collation_time"] = collation_time
    if accelerator.is_main_process:
        logger.info(f"Rollout Timing Breakdown: Total={rollout_duration:.2f}s | Gen={timing_data['gen_time']:.2f}s | Stats={timing_data['stats_time']:.2f}s | CPU={timing_data['cpu_time']:.2f}s | Collate={collation_time:.2f}s")

    # --- Gather rewards across processes ---
    # Rewards were computed locally on CPU. Need to gather them for consistent advantage calculation.
    if accelerator.num_processes > 1:
        rewards_gathered = accelerator.gather(collated_buffer["rewards"].to(accelerator.device))
        # Trim padding potentially added by gather
        rewards_gathered = rewards_gathered[:num_total_samples]
        collated_buffer["rewards"] = rewards_gathered.cpu() # Move back to CPU if needed later

        # We also need to gather other stats used for advantage/update phase
        # to ensure consistency across processes
        keys_to_gather = ["logprobs", "ref_logprobs", "response_mask"] # Add others if needed by update
        for key in keys_to_gather:
            if key in collated_buffer and isinstance(collated_buffer[key], torch.Tensor):
                 tensor_to_gather = collated_buffer[key].to(accelerator.device)
                 gathered_tensor = accelerator.gather(tensor_to_gather)
                 # Trim padding
                 gathered_tensor = gathered_tensor[:num_total_samples] # Assumes first dim is batch
                 collated_buffer[key] = gathered_tensor.cpu() # Move back to CPU
            else:
                 logger.warning(f"Cannot gather key '{key}' - not found or not a tensor.")
    # Note: Prompts don't necessarily need gathering if each process handles its own subset
    # But response-related tensors used in the update phase DO need gathering.

    return collated_buffer


# ==============================================================================
# == 5. GRPO Update Phase Logic (Modified for Accelerate)
# ==============================================================================

def run_grpo_update_epoch(
    actor_model: PreTrainedModel, # Prepared model
    tokenizer,
    optimizer: torch.optim.Optimizer, # Prepared optimizer
    lr_scheduler, # Prepared scheduler
    collated_buffer: Dict[str, torch.Tensor],
    cfg: DictConfig,
    accelerator: Accelerator # Added accelerator
) -> Dict[str, float]:
    """Runs one GRPO epoch with mini-batch updates using Accelerate."""
    actor_model.train()
    aggregate_metrics = {}
    optimizer_steps_taken = 0

    prompt_ids = collated_buffer["prompt_input_ids"].to(accelerator.device)
    # prompt_mask = collated_buffer["prompt_attention_mask"].to(accelerator.device) # Not needed for fwd pass if handled below
    response_ids = collated_buffer["response_input_ids"].to(accelerator.device)
    response_mask = collated_buffer["response_attention_mask"].to(accelerator.device)
    logprobs_old = collated_buffer["logprobs"].to(accelerator.device)
    ref_logprobs = collated_buffer["ref_logprobs"].to(accelerator.device)
    rewards = collated_buffer["rewards"].to(accelerator.device) # Rewards needed on device for advantage calc
    group_size = cfg.grpo.group_size
    kl_coeff = cfg.ppo.kl_coeff

    with torch.no_grad():
        kl_per_token = logprobs_old - ref_logprobs
        advantages = compute_grpo_advantages(
            rewards, kl_per_token, response_mask, group_size, kl_coeff
        ) # Advantages are now on accelerator.device

    # --- Mini-batch Loop ---
    num_samples = response_ids.shape[0] # Total number of generated responses (prompts * G)
    num_prompts = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]

    if num_samples == 0 or num_prompts == 0:
         logger.warning("Empty buffer provided to run_grpo_update_epoch. Skipping update.")
         return {}

    # Shuffle prompts for mini-batching
    prompt_indices = np.arange(num_prompts)
    np.random.shuffle(prompt_indices)

    prompt_mini_batch_size = cfg.ppo.mini_batch_size
    grad_accum_steps = cfg.ppo.gradient_accumulation_steps

    # Iterate through prompts for mini-batching
    for i in range(0, num_prompts, prompt_mini_batch_size):
        prompt_batch_indices = prompt_indices[i:i + prompt_mini_batch_size]
        actual_mini_batch_size_prompts = len(prompt_batch_indices)
        if actual_mini_batch_size_prompts == 0: continue

        # Get prompts for this mini-batch (already on device)
        batch_prompt_ids = prompt_ids[prompt_batch_indices]
        # batch_prompt_mask = prompt_mask[prompt_batch_indices] # Recreate mask if needed

        # Expand prompts for group size for forward pass
        fwd_prompt_ids = batch_prompt_ids.repeat_interleave(group_size, dim=0)
        # Recreate attention mask for the forward pass input
        fwd_prompt_mask = (fwd_prompt_ids != tokenizer.pad_token_id).long() # Assuming tokenizer is accessible or pad_id passed

        # Get corresponding samples (responses, logprobs_old, advantages)
        sample_batch_indices = []
        for p_idx in prompt_batch_indices:
            start = p_idx * group_size
            end = start + group_size
            if start < num_samples and end <= num_samples:
                 sample_batch_indices.extend(range(start, end))
            else: logger.warning(f"Index calculation error. p_idx={p_idx}, G={group_size}, num_samples={num_samples}")

        if not sample_batch_indices: continue
        sample_batch_indices = torch.tensor(sample_batch_indices, dtype=torch.long, device=accelerator.device)
        if torch.any(sample_batch_indices >= num_samples): continue

        batch_response_ids = response_ids[sample_batch_indices]
        batch_response_mask = response_mask[sample_batch_indices]
        batch_logprobs_old = logprobs_old[sample_batch_indices]
        batch_advantages = advantages[sample_batch_indices]

        # Combine for forward pass
        batch_full_ids = torch.cat((fwd_prompt_ids, batch_response_ids), dim=1)
        batch_full_mask = torch.cat((fwd_prompt_mask, batch_response_mask), dim=1)

        # --- GRPO Mini-batch Update ---
        # Context manager for gradient accumulation
        # No context manager needed if scaling loss manually
        # with accelerator.accumulate(actor_model): # Use if NOT scaling loss manually

        # 1. Forward Pass
        # Ensure use_cache is False if using gradient checkpointing
        use_cache_fwd = not cfg.training.gradient_checkpointing
        actor_model_outputs = actor_model(
            batch_full_ids,
            attention_mask=batch_full_mask,
            use_cache=use_cache_fwd
        )
        logits_new = actor_model_outputs.logits
        
        # 2. Extract response parts and calculate new logprobs
        current_resp_len = batch_response_ids.shape[1]
        start_idx = prompt_len - 1
        end_idx = prompt_len + current_resp_len - 1
        
        logits_new_resp = logits_new[:, start_idx:end_idx, :]
        
        logprobs_all_new = F.log_softmax(logits_new_resp, dim=-1)
        logprobs_new = torch.gather(
            logprobs_all_new, 2,
            batch_response_ids.unsqueeze(-1)).squeeze(-1)
        logprobs_new = logprobs_new * batch_response_mask
    
        # 3. Calculate Losses
        policy_loss, p_clip_frac, approx_kl = compute_grpo_policy_loss(
            logprobs_new, batch_logprobs_old, batch_advantages,
        batch_response_mask, cfg.ppo.clip_ratio)
        entropy_loss = compute_grpo_entropy_loss(logits_new_resp, batch_response_mask)
        
        # 4. Combine Losses
        loss = policy_loss + cfg.ppo.entropy_coeff * entropy_loss
        
        # 5. Backward Pass & Gradient Accumulation
        scaled_loss = loss / grad_accum_steps
        # Use accelerator.backward
        accelerator.backward(scaled_loss)
        
        # 6. Store Metrics (Collect from all processes later if needed)
        # For simplicity, log metrics from each process, average later if desired
        current_metrics = {
            'loss/policy': policy_loss.item(),
            'loss/entropy': -entropy_loss.item(),
            'loss/total': loss.item(),
            'params/policy_clip_frac': p_clip_frac.item(),
            'params/approx_kl': approx_kl.item(),
        }
        for key, val in current_metrics.items():
            aggregate_metrics.setdefault(key, []).append(val)


        # 7. Optimizer Step (if accumulation cycle complete)
        is_last_batch_in_prompt_group = (i // prompt_mini_batch_size + 1) % grad_accum_steps == 0
        is_last_batch_overall = (i + prompt_mini_batch_size) >= num_prompts
        if is_last_batch_in_prompt_group or is_last_batch_overall:
            optimizer_steps_taken += 1
            # Check if grads exist (optional, step handles it)
            # Clip gradients across all processes
            if accelerator.sync_gradients: # Only clip when gradients are synchronized
                grad_norm = accelerator.clip_grad_norm_(
                    actor_model.parameters(), max_norm=cfg.ppo.max_grad_norm
                )
                aggregate_metrics.setdefault('params/grad_norm', []).append(grad_norm.item())
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True) # Zero grad AFTER step

    # --- End of Epoch ---
    # Average metrics collected on this process
    final_metrics = {key: np.mean(val) for key, val in aggregate_metrics.items() if val}
    logger.info(f"Process {accelerator.process_index}: Completed GRPO epoch. Optimizer steps taken: {optimizer_steps_taken}")
    return final_metrics


def perform_grpo_updates(
    actor_model: PreTrainedModel,
    tokenizer, 
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    rollout_buffer: Dict[str, Any],
    cfg: DictConfig,
    accelerator: Accelerator # Added accelerator
) -> Dict[str, float]:
    """Performs multiple GRPO epochs on the collected rollout data using Accelerate."""
    # Buffer data should be on CPU initially after collation/gathering
    # Move only necessary tensors to the accelerator device for the update phase
    buffer_on_device = {
        k: v.to(accelerator.device) if k in [
            "prompt_input_ids", "prompt_attention_mask", # Needed for recreating inputs
            "response_input_ids", "response_attention_mask",
            "logprobs", "ref_logprobs", "rewards" # Needed for loss/advantage
        ] and isinstance(v, torch.Tensor) else v
        for k, v in rollout_buffer.items()
    }

    if "response_input_ids" not in buffer_on_device or \
       not isinstance(buffer_on_device["response_input_ids"], torch.Tensor) or \
       buffer_on_device["response_input_ids"].numel() == 0:
        logger.warning("No response tokens found in buffer on device. Skipping GRPO update.")
        return {}

    all_epoch_metrics = []
    num_epochs = cfg.ppo.get("epochs", 1)
    for grpo_epoch in range(num_epochs):
        if accelerator.is_main_process:
            logger.info(f"--- Starting GRPO Update Epoch {grpo_epoch + 1}/{num_epochs} ---")
        # Pass accelerator to epoch runner
        epoch_metrics = run_grpo_update_epoch(
            actor_model, tokenizer, optimizer, lr_scheduler, buffer_on_device, cfg, accelerator
        )

    # Aggregate metrics across processes and epochs
    final_aggregated_metrics = {}
    if all_epoch_metrics: # If any epochs ran successfully
        # Average metrics from the last successful epoch across all processes
        last_epoch_metrics = all_epoch_metrics[-1]
        for key in last_epoch_metrics:
            # Gather metric value from all processes
            metric_tensor = torch.tensor(last_epoch_metrics[key], device=accelerator.device)
            gathered_metrics = accelerator.gather(metric_tensor)
            # Average on the main process
            if accelerator.is_main_process:
                 final_aggregated_metrics[key] = gathered_metrics.mean().item()

    return final_aggregated_metrics


# ==============================================================================
# == 6. Training Setup and Orchestration (Adapted for GRPO & Accelerate)
# ==============================================================================

def setup_training_grpo(cfg: DictConfig, accelerator: Accelerator) -> str:
    """Sets random seeds and output directory. Device handled by Accelerate."""
    # Set seed on all processes using accelerator
    accelerator.wait_for_everyone()
    seed = cfg.training.seed + accelerator.process_index # Ensure different seed per process if needed for dataloader
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed} for process {accelerator.process_index}")

    output_dir = cfg.training.output_dir
    # Create directory only on the main process
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    accelerator.wait_for_everyone() # Ensure directory exists before proceeding
    return output_dir

def load_models_and_tokenizer_grpo(
    cfg: DictConfig, accelerator: Accelerator
) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedModel], Optional[PreTrainedTokenizerBase]]:
    """Loads tokenizer, actor model, and reference model using Accelerate device."""
    # Tokenizer loading is usually fine on main process then broadcasted implicitly? Or load everywhere.
    # Let's load everywhere for simplicity.
    logger.info(f"Loading tokenizer: {cfg.model.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.info("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading models: {cfg.model.name}")
    model_kwargs = {}
    model_dtype_str = cfg.model.get("torch_dtype", "auto")
    if model_dtype_str != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, model_dtype_str)
    if cfg.model.get("trust_remote_code", False): model_kwargs["trust_remote_code"] = True

    # --- Load Actor Model ---
    # Load on CPU first or let accelerate handle placement via prepare?
    # Loading on CPU then preparing is safer for large models.
    # However, for simplicity here, load directly (may need adjustment for >1 node)
    actor_model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)
    if actor_model.config.pad_token_id is None:
        actor_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Actor model loaded.")
    if cfg.training.get("gradient_checkpointing", False):
        actor_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for actor model.")

    # --- Load Reference Model ---
    # Load ref model separately, don't prepare it with accelerator
    ref_model_kwargs = model_kwargs.copy()
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **ref_model_kwargs)
    # Move ref model to the correct device
    ref_model.to(accelerator.device)
    if ref_model.config.pad_token_id is None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
    for param in ref_model.parameters(): param.requires_grad = False
    ref_model.eval()
    logger.info("Reference model loaded, frozen, and moved to device.")

    return actor_model, ref_model, tokenizer


def setup_optimizer_grpo(cfg: DictConfig, model: nn.Module, accelerator: Accelerator) -> Tuple[Optional[torch.optim.Optimizer], Optional[Any]]:
    """Sets up the optimizer and LR scheduler for GRPO with Accelerate."""
    # (Code largely identical to previous version, just uses accelerator device type check)
    use_8bit = cfg.ppo.get("use_8bit_adam", True)
    lr = cfg.ppo.learning_rate

    if use_8bit and bnb_available and accelerator.distributed_type == accelerate.DistributedType.MULTI_GPU: # Check if on GPU via accelerator
        is_quantized = hasattr(model, 'quantization_config') and \
                        (model.quantization_config.load_in_8bit or model.quantization_config.load_in_4bit)
        if is_quantized:
                logger.warning("Using 8-bit AdamW with a quantized model. Consider standard AdamW.")
                optimizer = AdamW(model.parameters(), lr=lr)
        else:
                logger.info("Using 8-bit AdamW Optimizer (bitsandbytes)")
                optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        if use_8bit and accelerator.distributed_type == accelerate.DistributedType.MULTI_GPU: logger.warning("8-bit Adam not used (bnb not available or other issue). Using standard AdamW.")
        elif use_8bit: logger.info("8-bit Adam requires CUDA. Using standard AdamW.")
        else: logger.info("Using standard AdamW Optimizer")
        optimizer = AdamW(model.parameters(), lr=lr)

    # --- Scheduler Setup ---
    group_size = cfg.grpo.group_size
    num_prompts_per_rollout = cfg.ppo.rollout_samples
    prompt_mini_batch_size = max(1, cfg.ppo.mini_batch_size)
    grad_accum_steps = max(1, cfg.ppo.gradient_accumulation_steps)
    num_epochs = max(1, cfg.ppo.epochs)

    # Calculate number of updates PER PROCESS
    num_prompts_per_process = math.ceil(num_prompts_per_rollout / accelerator.num_processes)
    num_mini_batches_per_process = math.ceil(num_prompts_per_process / prompt_mini_batch_size)
    num_optimizer_steps_per_ppo_step_per_process = math.ceil(num_mini_batches_per_process / grad_accum_steps) * num_epochs
    num_training_steps = cfg.training.total_ppo_steps * num_optimizer_steps_per_ppo_step_per_process

    scheduler_name = cfg.ppo.get("scheduler", "linear")
    warmup_steps = cfg.ppo.get("warmup_steps", 0)

    if accelerator.is_main_process:
        logger.info(f"Setting up LR scheduler: {scheduler_name} with {warmup_steps} warmup steps.")
        logger.info(f"Total optimizer steps calculated (per process): {num_training_steps}")

    if scheduler_name == "cosine_with_min_lr":
        scheduler_name = "cosine"
        logger.warning("Scheduler 'cosine_with_min_lr' specified. Using 'cosine'. Min LR handling might need custom logic.")

    lr_scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler

# create_generation_config_grpo remains the same

def save_model_grpo(model: nn.Module, tokenizer: PreTrainedTokenizerBase, save_path: str, accelerator: Accelerator):
    """Saves the GRPO model and tokenizer using Accelerate."""
    accelerator.wait_for_everyone() # Ensure all processes are ready before saving
    if accelerator.is_main_process:
        logger.info(f"Saving model checkpoint to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        # Unwrap the model to get the base Hugging Face model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved by main process.")
    accelerator.wait_for_everyone() # Ensure main process finished saving before others continue


# ==============================================================================
# == 7. Main Training Orchestration (Adapted for GRPO & Accelerate)
# ==============================================================================

def train_grpo(cfg: DictConfig):
    """Main GRPO training loop using Accelerate."""
    # --- Initialize Accelerator ---
    # Handle find_unused_parameters for gradient checkpointing
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.training.gradient_checkpointing)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.ppo.gradient_accumulation_steps,
                              kwargs_handlers=[ddp_kwargs])

    # --- 1. Initial Setup (Pass accelerator) ---
    output_dir = setup_training_grpo(cfg, accelerator)
    # Save config only on main process
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(output_dir, "effective_config_grpo.yaml"))

    wandb_initialized = False
    if cfg.wandb.get("report_to_wandb", False) and accelerator.is_main_process:
        try:
            wandb.init(
                project=cfg.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.wandb.get("name", None)
            )
            wandb_initialized = True
        except Exception as e: logger.error(f"Failed to initialize WandB: {e}")

    # --- 2. Load Models and Tokenizer (Pass accelerator for device) ---
    actor_model, ref_model, tokenizer = load_models_and_tokenizer_grpo(cfg, accelerator)
    if actor_model is None or ref_model is None or tokenizer is None:
         logger.critical("Model or Tokenizer loading failed. Exiting."); return

    # --- 3. Load and Preprocess Dataset ---
    # Dataset loading/preprocessing usually done identically on all processes
    processed_dataset = load_and_preprocess_dataset(cfg, tokenizer)

    # --- 4. Setup Optimizer & Scheduler (Pass accelerator) ---
    optimizer, lr_scheduler = setup_optimizer_grpo(cfg, actor_model, accelerator)
    if optimizer is None or lr_scheduler is None:
         logger.critical("Optimizer/Scheduler setup failed. Exiting."); return

    # --- 5. Generation Config ---
    gen_config = create_generation_config_grpo(cfg, tokenizer)

    # --- 6. Collate Function ---
    # (collade_fn remains the same)
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        padded_inputs = tokenizer.pad({"input_ids": input_ids}, padding='longest', return_tensors="pt", return_attention_mask=True)
        ground_truths = [item['ground_truth_answer'] for item in batch]
        return {"prompt_input_ids": padded_inputs["input_ids"],
                "prompt_attention_mask": padded_inputs["attention_mask"],
                "ground_truth_answers": ground_truths}

    # --- Prepare components with Accelerator ---
    actor_model, optimizer, lr_scheduler = accelerator.prepare(
        actor_model, optimizer, lr_scheduler
    )
    # Note: Dataloader preparation happens inside the loop for subsetting

    # --- 7. Main GRPO Loop ---
    if accelerator.is_main_process: logger.info("\n--- Starting GRPO Training ---")
    group_size = cfg.grpo.group_size
    rollout_prompts = cfg.ppo.rollout_samples
    if accelerator.is_main_process: logger.info(f"Using Group Size (G): {group_size}")

    for grpo_step in range(cfg.training.total_ppo_steps):
        step_start_time = time.time()
        if accelerator.is_main_process:
            logger.info(f"\n===== GRPO Step {grpo_step + 1}/{cfg.training.total_ppo_steps} =====")

        # --- Phase 1: Rollout (Grouped) ---
        if accelerator.is_main_process: logger.info("Phase 1: Generating Rollouts...")
        # Select prompts for this rollout step - needs care in distributed setting
        # Simplest: Each process works on a shard of the dataset for rollouts
        # Or: Main process selects indices, broadcasts them?
        # Let's assume each process selects its own subset for now
        num_prompts_to_select = min(rollout_prompts, len(processed_dataset))
        # Ensure each process gets roughly equal work
        num_prompts_per_process = math.ceil(num_prompts_to_select / accelerator.num_processes)
        # Select a shard of the shuffled dataset for this process
        # This requires careful index handling or using dataset sharding capabilities
        # Simplified approach: select a range based on process index
        start_idx = accelerator.process_index * num_prompts_per_process
        end_idx = min(start_idx + num_prompts_per_process, num_prompts_to_select)

        if start_idx >= end_idx: # Handle cases where some processes get no data
             logger.info(f"Process {accelerator.process_index} has no prompts for this step.")
             # Need to participate in barriers later, maybe just create empty buffer?
             rollout_buffer = {} # Create empty buffer
        else:
            dataset_shard = processed_dataset.shuffle(seed=cfg.training.seed + grpo_step).select(range(start_idx, end_idx))
            if accelerator.is_main_process: logger.info(f"Total prompts for rollout: {num_prompts_to_select}, Prompts per process: ~{num_prompts_per_process}")

            prompt_dataloader = DataLoader(
                dataset_shard,
                batch_size=cfg.ppo.batch_size, # This is now prompts per batch *per process*
                shuffle=False,
                collate_fn=collate_fn
            )
            # Prepare dataloader for distributed sampling
            prompt_dataloader = accelerator.prepare(prompt_dataloader)

            rollout_buffer = perform_rollouts_grpo(
                actor_model, ref_model, tokenizer, prompt_dataloader, gen_config, group_size, accelerator
            )

        # --- Gather rollout buffers across processes ---
        # Need to gather the *entire buffer dictionary* or key components
        # This can be memory intensive. Alternative: save shards and load?
        # Simple approach: gather essential tensors needed for update phase.
        # Collation should ideally happen *after* gathering on the main process or per process?
        # Let's assume perform_rollouts returns CPU tensors and we gather them.

        # Gather necessary tensors (rewards, logprobs, ref_logprobs, response_mask, etc.)
        # This part requires careful implementation depending on how updates are handled (main process vs distributed)
        # For simplicity, let's assume updates run distributed and buffer contains gathered data from perform_rollouts
        # (The gather logic was added inside perform_rollouts for now)

        # Validate buffer (check on main process after potential gather)
        if accelerator.is_main_process:
            if not rollout_buffer or "rewards" not in rollout_buffer or \
               not isinstance(rollout_buffer["rewards"], torch.Tensor) or \
               rollout_buffer["rewards"].numel() == 0:
                logger.warning("Invalid or empty rollout buffer after gathering. Skipping update."); continue

            # Log aggregated stats from main process
            avg_reward = rollout_buffer["rewards"].mean().item()
            num_generated_samples = rollout_buffer["rewards"].shape[0]
            num_input_prompts = rollout_buffer["prompt_input_ids"].shape[0] # This might be wrong after gather
            avg_resp_len = rollout_buffer.get("avg_response_length", 0.0)
            rollout_duration = rollout_buffer.get("rollout_duration_seconds", 0.0) # Time from one process
            gen_time = rollout_buffer.get("timing/total_gen_time", 0.0)
            stats_time = rollout_buffer.get("timing/total_stats_time", 0.0)
            cpu_time = rollout_buffer.get("timing/total_cpu_time", 0.0)
            collation_time = rollout_buffer.get("timing/collation_time", 0.0)

            logger.info(
                f"Rollout complete (Total Samples: {num_generated_samples}). " # Log total samples
                f"Avg Reward: {avg_reward:.4f}, Avg Resp Len: {avg_resp_len:.2f}, "
                f"Max Rollout Duration: {rollout_duration:.2f}s " # This is max time across processes if gathered
                # Timing breakdown might be less meaningful after gather unless averaged
            )

        # --- Phase 2: Update ---
        if accelerator.is_main_process: logger.info("Phase 2: Performing GRPO Updates...")
        update_start_time = time.time()
        # Pass accelerator to update function
        metrics = perform_grpo_updates(
            actor_model, tokenizer, optimizer, lr_scheduler, rollout_buffer, cfg, accelerator
        )
        update_duration = time.time() - update_start_time
        if accelerator.is_main_process: logger.info(f"Update phase duration: {update_duration:.2f}s")


        # Log metrics (only on main process after aggregation in perform_grpo_updates)
        log_data = {}
        if accelerator.is_main_process:
            if metrics:
                log_data.update(metrics)
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Update Metrics (Avg over Epoch & Procs): {log_str}")
            else: logger.warning("GRPO update skipped or failed.")

            # Add rollout metrics for logging
            log_data["rollout/reward_mean"] = avg_reward # Use gathered avg reward
            log_data["rollout/avg_response_length"] = avg_resp_len
            log_data["rollout/duration_seconds"] = rollout_duration
            # Add timing if meaningful (e.g., max across processes)
            # ...

            log_data["update/duration_seconds"] = update_duration
            log_data["step_duration_seconds"] = time.time() - step_start_time

            log_data["params/learning_rate"] = lr_scheduler.get_last_lr()[0]

            if wandb_initialized and metrics:
                try: wandb.log(log_data, step=grpo_step)
                except Exception as e: logger.error(f"Failed to log to WandB: {e}")

        # --- Phase 3: Save Checkpoint ---
        if (grpo_step + 1) % cfg.training.save_interval == 0:
            # Pass accelerator to save function
            save_model_grpo(actor_model, tokenizer, os.path.join(output_dir, f"step_{grpo_step + 1}"), accelerator)

        # Barrier to ensure all processes finish step before next rollout
        accelerator.wait_for_everyone()

    # --- 8. Final Save ---
    if accelerator.is_main_process:
        if wandb_initialized:
            try: wandb.finish()
            except Exception as e: logger.error(f"Error finishing WandB run: {e}")
        logger.info("\n--- GRPO Training Finished ---")
        # Final save call
        save_model_grpo(actor_model, tokenizer, os.path.join(output_dir, "final"), accelerator)


# ==============================================================================
# == 8. Command-Line Interface Logic (Adapted for GRPO Config)
# ==============================================================================
# (load_config_with_cli_overrides_grpo remains the same)
def load_config_with_cli_overrides_grpo() -> Optional[DictConfig]:
    """Loads OmegaConf config, handling defaults and CLI overrides."""
    parser = argparse.ArgumentParser(description="GRPO RL Trainer")
    parser.add_argument("--config", required=True, help="Path to the GRPO config file (e.g., configs/grpo_config.yaml)")
    parser.add_argument("overrides", nargs="*", help="Key=value config overrides")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Error: Config file not found at '{config_path}'.")
        return None # Return None on failure

    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Apply command-line overrides
    if args.overrides:
        logger.info(f"Applying overrides: {args.overrides}")
        cli_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    # Add a default grpo section if missing, e.g., for group_size
    if 'grpo' not in cfg:
        logger.warning("Adding default 'grpo' section to config.")
        OmegaConf.set_struct(cfg, False) # Allow adding new keys
        cfg.grpo = {'group_size': 4} # Add default group size
        OmegaConf.set_struct(cfg, True)
    elif 'group_size' not in cfg.grpo:
         logger.warning("Adding default 'grpo.group_size=4' to config.")
         OmegaConf.set_struct(cfg.grpo, False)
         cfg.grpo.group_size = 4
         OmegaConf.set_struct(cfg.grpo, True)


    logger.info("--------- Final Configuration ---------")
    # Log config only on main process to avoid clutter
    if Accelerator().is_main_process: # Use a temporary accelerator instance just for the check
         logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("---------------------------------------")
    return cfg

# ==============================================================================
# == 9. Entry Point
# ==============================================================================

if __name__ == "__main__":
    config = load_config_with_cli_overrides_grpo()
    if config:
        train_grpo(config)
    else:
        # Logger might not be configured if config loading failed early
        print("CRITICAL: Failed to load configuration. Exiting.")
        sys.exit(1)
