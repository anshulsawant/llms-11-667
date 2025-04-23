# grpo_trainer.py
# -*- coding: utf-8 -*-
"""
Minimal GRPO Trainer script, adapted from PPO trainer.
Focuses on core GRPO logic: group generation and relative advantage.
Omits value function and GAE. Avoids extensive defensive programming.
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

from ppo_trainer_solutions import(
    masked_mean,
    masked_whiten,
    compute_gsm8k_reward,
    extract_gsm8k_solution,
    load_and_preprocess_dataset,
    pad_and_collate_tensors
)

from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM, # Use standard LM model
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
import time # For timing

# ==============================================================================
# == 2. Core GRPO Algorithm Components
# ==============================================================================

def compute_grpo_advantages(
    rewards: torch.Tensor, # Shape (num_prompts * group_size,) or (num_prompts, group_size)
    kl_penalties: torch.Tensor, # Shape (num_prompts * group_size, resp_len)
    response_mask: torch.Tensor, # Shape (num_prompts * group_size, resp_len)
    group_size: int,
    kl_coeff: float
) -> torch.Tensor:
    """
    Computes GRPO advantages by normalizing rewards within each group.
    Applies KL penalty *before* normalization.
    """
    with torch.no_grad():
        num_samples = rewards.shape[0]
        num_prompts = num_samples // group_size
        if num_samples % group_size != 0:
             logger.warning(f"Total samples ({num_samples}) not divisible by group_size ({group_size}).")
             # Adjust num_prompts if needed, or handle remainder
             num_prompts = num_samples // group_size


        # Reshape rewards to (num_prompts, group_size)
        if rewards.dim() == 1:
            rewards = rewards.view(num_prompts, group_size)

        # --- Apply KL Penalty to Rewards ---
        # Calculate mean KL penalty per sequence (masked mean over resp_len)
        # kl_penalties has shape (num_samples, resp_len)
        mean_kl_penalty_per_seq = masked_mean(kl_penalties, response_mask, dim=1) # Shape: (num_samples,)
        mean_kl_penalty_per_seq = mean_kl_penalty_per_seq.view(num_prompts, group_size) # Shape: (num_prompts, group_size)

        # Subtract KL penalty from the reward
        # Note: Original GRPO paper might integrate KL differently (e.g., in loss),
        # but subtracting from reward is common in PPO/RLHF. Adjust if needed.
        adjusted_rewards = rewards - kl_coeff * mean_kl_penalty_per_seq # Shape: (num_prompts, group_size)
        # ------------------------------------

        # Calculate mean and std dev *within each group*
        group_mean = adjusted_rewards.mean(dim=1, keepdim=True) # Shape: (num_prompts, 1)
        group_std = adjusted_rewards.std(dim=1, keepdim=True)   # Shape: (num_prompts, 1)

        # Normalize rewards to get advantages
        advantages = (adjusted_rewards - group_mean) / (group_std + 1e-8) # Add epsilon

        # Reshape advantages back to (num_prompts * group_size, 1) and expand to match response length
        advantages = advantages.view(num_prompts * group_size, 1)
        advantages = advantages.expand(-1, response_mask.shape[1]) # Shape: (num_samples, resp_len)

        # Apply response mask (optional but good practice)
        advantages = advantages * response_mask.float()

    return advantages


def compute_grpo_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor, # GRPO advantages
    response_mask: torch.Tensor,
    clip_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes GRPO policy loss (clipped surrogate objective). Identical logic to PPO policy loss, just uses GRPO advantages."""
    with torch.no_grad():
        mask = response_mask.bool()
        if advantages.shape != log_probs_old.shape:
            raise ValueError(f"Shape mismatch: advantages {advantages.shape}, log_probs {log_probs_old.shape}")
    log_ratio = (log_probs_new - log_probs_old).clamp(-20, 20)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

    with torch.no_grad():
        clip_frac = masked_mean(torch.gt(torch.abs(ratio - 1.0), clip_ratio).float(), mask)
        # Note: KL calculation here is KL(old || new). Sometimes KL(new || old) is used.
        approx_kl = masked_mean(log_probs_old - log_probs_new, mask)

    return policy_loss, clip_frac, approx_kl


def compute_grpo_entropy_loss(logits_new: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Computes entropy loss. Identical logic to PPO."""
    mask = response_mask.bool()
    dist = torch.distributions.Categorical(logits=logits_new.float())
    entropy = dist.entropy()
    entropy_loss = -masked_mean(entropy, mask) # Maximize entropy
    return entropy_loss

# ==============================================================================
# == 3. Actor Model Definition (No Value Head)
# ==============================================================================

# We can just use AutoModelForCausalLM directly, or wrap it if needed later.
# For simplicity, we'll load AutoModelForCausalLM directly in setup.

# ==============================================================================
# == 4. Rollout Phase Logic (Modified for Group Generation)
# ==============================================================================

def generate_responses_grouped(
    model: PreTrainedModel, # Standard LM model
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_config: GenerationConfig,
    group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates G responses for each prompt in a batch."""
    model.eval()
    batch_size = prompt_ids.shape[0]

    gen_config.do_sample = True # Ensure sampling for diversity in group

    with torch.no_grad():
        # Input needs repeating for generation: (B, L) -> (B*G, L)
        expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

        generated_output = model.generate(
            input_ids=expanded_prompt_ids,
            attention_mask=expanded_prompt_mask,
            generation_config=gen_config,
            pad_token_id=tokenizer.pad_token_id
        )
        # Output shape: (batch_size * group_size, full_len)

        # Extract only generated tokens
        prompt_len = prompt_ids.shape[1]
        response_ids = generated_output[:, prompt_len:] # Shape: (B*G, resp_len)

        # Create response mask
        response_mask = (response_ids != tokenizer.pad_token_id).long()

    return response_ids, response_mask # Return grouped results


def calculate_rollout_stats_grpo(
    actor_model: PreTrainedModel, # Standard LM model
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,      # Shape (batch, prompt_len)
    prompt_mask: torch.Tensor,     # Shape (batch, prompt_len)
    response_ids: torch.Tensor,    # Shape (batch*G, resp_len)
    response_mask: torch.Tensor,   # Shape (batch*G, resp_len)
    group_size: int
) -> Dict[str, torch.Tensor]:
    """Calculates logprobs, ref_logprobs for grouped responses."""
    actor_model.eval()
    ref_model.eval()

    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]

    # Expand prompts to match grouped responses: (B, L) -> (B*G, L)
    expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

    # Combine expanded prompt and response for forward passes
    full_ids = torch.cat((expanded_prompt_ids, response_ids), dim=1)
    full_mask = torch.cat((expanded_prompt_mask, response_mask), dim=1)
    full_len = full_ids.shape[1]

    with torch.no_grad():
        # Get actor logits
        actor_logits = actor_model(full_ids, attention_mask=full_mask).logits
        # Get reference model logits
        ref_logits = ref_model(full_ids, attention_mask=full_mask).logits

        # --- Calculate Logprobs for the RESPONSE part ---
        start_idx = prompt_len - 1
        end_idx = full_len - 1

        if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
            logprobs = torch.empty((batch_size * group_size, 0), dtype=torch.float, device=prompt_ids.device)
            ref_logprobs = torch.empty((batch_size * group_size, 0), dtype=torch.float, device=prompt_ids.device)
        else:
            logits_resp = actor_logits[:, start_idx:end_idx, :]
            ref_logits_resp = ref_logits[:, start_idx:end_idx, :]
            target_ids = response_ids # Shape: (B*G, resp_len)

            # Ensure shapes match before gather
            current_resp_len = logits_resp.shape[1]
            if current_resp_len != target_ids.shape[1]:
                 min_len = min(current_resp_len, target_ids.shape[1])
                 logits_resp = logits_resp[:, :min_len, :]
                 ref_logits_resp = ref_logits_resp[:, :min_len, :]
                 target_ids = target_ids[:, :min_len]
                 response_mask_adjusted = response_mask[:,:min_len]
            else:
                 response_mask_adjusted = response_mask

            # Calculate log probabilities
            logprobs_all = F.log_softmax(logits_resp, dim=-1)
            ref_logprobs_all = F.log_softmax(ref_logits_resp, dim=-1)
            logprobs = torch.gather(logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            ref_logprobs = torch.gather(ref_logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Apply mask
            logprobs = logprobs * response_mask_adjusted
            ref_logprobs = ref_logprobs * response_mask_adjusted

    return {
        "logprobs": logprobs, # Shape: (B*G, resp_len)
        "ref_logprobs": ref_logprobs, # Shape: (B*G, resp_len)
        # No 'values' needed for GRPO
    }


def perform_rollouts_grpo(
    actor_model: PreTrainedModel, # Standard LM model
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_dataloader: DataLoader,
    gen_config: GenerationConfig,
    group_size: int, # Added parameter
    device: torch.device
) -> Dict[str, Any]:
    """Generates groups of responses and computes stats for GRPO update."""
    rollout_start_time = time.time()
    buffer_lists = {
        "prompt_input_ids": [], "prompt_attention_mask": [],
        "response_input_ids": [], "response_attention_mask": [],
        "logprobs": [], "ref_logprobs": [], # No values
        "rewards": [], "full_texts": [], "ground_truth_answers": []
    }
    timing_data = {"gen_time": 0.0, "stats_time": 0.0, "cpu_time": 0.0}

    progress_bar = tqdm(prompt_dataloader, desc="GRPO Rollout", leave=False)
    for batch in progress_bar:
        if batch is None: continue
        cpu_start_time = time.time()
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"] # List of strings, len B
        cpu_prep_time = time.time() - cpu_start_time

        # 1. Generate G responses per prompt
        gen_start_time = time.time()
        response_ids, response_mask = generate_responses_grouped(
            actor_model, tokenizer, prompt_ids, prompt_mask, gen_config, group_size
        ) # Shapes: (B*G, R_i), (B*G, R_i)
        timing_data["gen_time"] += (time.time() - gen_start_time)

        # 2. Calculate stats (logprobs, ref_logprobs)
        stats_start_time = time.time()
        stats = calculate_rollout_stats_grpo(
            actor_model, ref_model, tokenizer,
            prompt_ids, prompt_mask, response_ids, response_mask, group_size
        ) # Dict of tensors (B*G, R_i)
        timing_data["stats_time"] += (time.time() - stats_start_time)

        # 3. Decode texts and calculate rewards
        cpu_work_start_time = time.time()
        # Expand prompts to match responses for decoding: (B, L) -> (B*G, L)
        expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        full_ids = torch.cat((expanded_prompt_ids, response_ids), dim=1)
        full_decoded_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True) # List len B*G

        # Expand ground truths to match responses: [gt1, gt2] -> [gt1]*G + [gt2]*G
        expanded_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]

        rewards = torch.tensor(
            [compute_gsm8k_reward(txt, gt) for txt, gt in zip(full_decoded_texts, expanded_ground_truths)],
            dtype=torch.float32, device='cpu'
        ) # Shape: (B*G,)

        # 4. Append results to buffer lists
        # Store original prompts once per group
        buffer_lists["prompt_input_ids"].append(prompt_ids.cpu())
        buffer_lists["prompt_attention_mask"].append(prompt_mask.cpu())
        # Store grouped responses and stats
        buffer_lists["response_input_ids"].append(response_ids.cpu())
        buffer_lists["response_attention_mask"].append(response_mask.cpu())
        buffer_lists["logprobs"].append(stats["logprobs"].cpu())
        buffer_lists["ref_logprobs"].append(stats["ref_logprobs"].cpu())
        buffer_lists["rewards"].append(rewards) # Shape (B*G,)
        buffer_lists["full_texts"].extend(full_decoded_texts) # List len B*G
        buffer_lists["ground_truth_answers"].extend(expanded_ground_truths) # List len B*G
        timing_data["cpu_time"] += (time.time() - cpu_work_start_time) + cpu_prep_time

    # --- Collate the buffer lists ---
    collation_start_time = time.time()
    collated_buffer = {}
    # Keep track of number of prompts vs total samples (prompts * group_size)
    num_prompts_processed = sum(p.shape[0] for p in buffer_lists["prompt_input_ids"])
    num_total_samples = sum(r.shape[0] for r in buffer_lists["response_input_ids"])

    # Collate prompts (B, P_len) -> (TotalPrompts, Max_P_len)
    collated_buffer["prompt_input_ids"] = pad_and_collate_tensors(buffer_lists["prompt_input_ids"], tokenizer.pad_token_id or 0)
    collated_buffer["prompt_attention_mask"] = pad_and_collate_tensors(buffer_lists["prompt_attention_mask"], 0)

    # Collate grouped responses/stats (B*G, R_len) -> (TotalSamples, Max_R_len)
    collated_buffer["response_input_ids"] = pad_and_collate_tensors(buffer_lists["response_input_ids"], tokenizer.pad_token_id or 0)
    collated_buffer["response_attention_mask"] = pad_and_collate_tensors(buffer_lists["response_attention_mask"], 0)
    collated_buffer["logprobs"] = pad_and_collate_tensors(buffer_lists["logprobs"], 0.0)
    collated_buffer["ref_logprobs"] = pad_and_collate_tensors(buffer_lists["ref_logprobs"], 0.0)

    # Concatenate rewards (B*G,) -> (TotalSamples,)
    collated_buffer["rewards"] = torch.cat(buffer_lists["rewards"], dim=0) if buffer_lists["rewards"] else torch.empty(0)

    # Store lists as is
    collated_buffer["full_texts"] = buffer_lists["full_texts"]
    collated_buffer["ground_truth_answers"] = buffer_lists["ground_truth_answers"]
    collation_time = time.time() - collation_start_time

    # --- Calculate Average Response Length (Corrected) ---
    individual_lengths = []
    # Iterate through the list of batch masks (B*G, R_len)
    for mask_batch in buffer_lists["response_attention_mask"]:
        if mask_batch.numel() > 0:
            lengths_in_batch = mask_batch.sum(dim=1) # Length per sequence
            individual_lengths.extend(lengths_in_batch.cpu().numpy())
    avg_resp_len = np.mean(individual_lengths) if individual_lengths else 0.0
    logger.info(f"Average response length per sequence for this rollout: {avg_resp_len:.2f}")

    rollout_duration = time.time() - rollout_start_time
    collated_buffer["avg_response_length"] = avg_resp_len
    collated_buffer["rollout_duration_seconds"] = rollout_duration
    collated_buffer["timing/total_gen_time"] = timing_data["gen_time"]
    collated_buffer["timing/total_stats_time"] = timing_data["stats_time"]
    collated_buffer["timing/total_cpu_time"] = timing_data["cpu_time"]
    collated_buffer["timing/collation_time"] = collation_time
    logger.info(f"Rollout Timing Breakdown: Total={rollout_duration:.2f}s | Gen={timing_data['gen_time']:.2f}s | Stats={timing_data['stats_time']:.2f}s | CPU={timing_data['cpu_time']:.2f}s | Collate={collation_time:.2f}s")

    return collated_buffer


# ==============================================================================
# == 5. GRPO Update Phase Logic
# ==============================================================================

def run_grpo_update_epoch(
    actor_model: PreTrainedModel, # Standard LM model
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    collated_buffer: Dict[str, torch.Tensor],
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Runs one GRPO epoch with mini-batch updates."""
    actor_model.train()
    aggregate_metrics = {}
    ppo_step_count = 0 # Renamed from ppo_step_count, but tracks optimizer steps

    # Load data from buffer (already collated and on device)
    prompt_ids = collated_buffer["prompt_input_ids"]
    prompt_mask = collated_buffer["prompt_attention_mask"]
    response_ids = collated_buffer["response_input_ids"] # Shape (TotalSamples, R_len)
    response_mask = collated_buffer["response_attention_mask"] # Shape (TotalSamples, R_len)
    logprobs_old = collated_buffer["logprobs"] # Shape (TotalSamples, R_len)
    ref_logprobs = collated_buffer["ref_logprobs"] # Shape (TotalSamples, R_len)
    rewards = collated_buffer["rewards"] # Shape (TotalSamples,)
    group_size = cfg.grpo.group_size # Get group size from config

    # --- Calculate GRPO Advantages (Once per epoch) ---
    # Need KL penalties first
    with torch.no_grad():
        kl_per_token = logprobs_old - ref_logprobs # Shape (TotalSamples, R_len)
        # Calculate advantages using grouped rewards and KL penalties
        advantages = compute_grpo_advantages(
            rewards, kl_per_token, response_mask, group_size, cfg.ppo.kl_coeff # Use ppo.kl_coeff for now
        ) # Shape (TotalSamples, R_len)

    # --- Mini-batch Loop ---
    num_samples = response_ids.shape[0] # Total number of generated responses (prompts * G)
    num_prompts = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]

    # We need to map samples back to prompts for the forward pass
    # Create indices that shuffle prompts, then expand for groups
    prompt_indices = np.arange(num_prompts)
    np.random.shuffle(prompt_indices)

    # Iterate through prompts for mini-batching
    for i in range(0, num_prompts, cfg.ppo.mini_batch_size): # Use ppo.mini_batch_size for prompt batching
        prompt_batch_indices = prompt_indices[i:i + cfg.ppo.mini_batch_size]
        actual_mini_batch_size_prompts = len(prompt_batch_indices)
        if actual_mini_batch_size_prompts == 0: continue

        # Get prompts for this mini-batch
        batch_prompt_ids = prompt_ids[prompt_batch_indices]
        batch_prompt_mask = prompt_mask[prompt_batch_indices]

        # Expand prompts for group size for forward pass
        fwd_prompt_ids = batch_prompt_ids.repeat_interleave(group_size, dim=0)
        fwd_prompt_mask = batch_prompt_mask.repeat_interleave(group_size, dim=0)

        # Get corresponding samples (responses, logprobs_old, advantages)
        # Indices need careful calculation: prompt_idx * group_size + group_offset
        sample_batch_indices = []
        for p_idx in prompt_batch_indices:
            sample_batch_indices.extend(range(p_idx * group_size, (p_idx + 1) * group_size))

        batch_response_ids = response_ids[sample_batch_indices]
        batch_response_mask = response_mask[sample_batch_indices]
        batch_logprobs_old = logprobs_old[sample_batch_indices]
        batch_advantages = advantages[sample_batch_indices]

        # Combine for forward pass
        batch_full_ids = torch.cat((fwd_prompt_ids, batch_response_ids), dim=1)
        batch_full_mask = torch.cat((fwd_prompt_mask, batch_response_mask), dim=1)

        # --- GRPO Mini-batch Update ---
        # 1. Forward Pass (No value head needed)
        logits_new = actor_model(batch_full_ids, attention_mask=batch_full_mask).logits

        # 2. Extract response parts and calculate new logprobs
        start_idx = prompt_len - 1
        end_idx = prompt_len + resp_len - 1
        if start_idx < 0 or end_idx <= start_idx or end_idx > logits_new.shape[1]:
            logger.warning(f"Invalid slice indices in GRPO update. Skipping mini-batch chunk.")
            continue

        logits_new_resp = logits_new[:, start_idx:end_idx, :]

        if logits_new_resp.shape[1] != batch_response_ids.shape[1]:
             logger.warning(f"Mismatch logits/response len in GRPO update. Skipping mini-batch chunk.")
             continue

        logprobs_all_new = F.log_softmax(logits_new_resp, dim=-1)
        logprobs_new = torch.gather(
            logprobs_all_new, 2,
            batch_response_ids.unsqueeze(-1)).squeeze(-1)

        # Apply mask
        logprobs_new = logprobs_new * batch_response_mask

        # 3. Calculate Losses (Policy and Entropy only)
        policy_loss, p_clip_frac, approx_kl = compute_grpo_policy_loss(
            logprobs_new, batch_logprobs_old, batch_advantages,
            batch_response_mask, cfg.ppo.clip_ratio) # Use ppo.clip_ratio

        entropy_loss = compute_grpo_entropy_loss(logits_new_resp, batch_response_mask)

        # 4. Combine Losses
        # Use ppo coeffs for now, adjust config later if needed
        loss = policy_loss + cfg.ppo.entropy_coeff * entropy_loss

        # 5. Backward Pass & Gradient Accumulation
        # Accumulate based on prompt mini-batch size
        scaled_loss = loss / cfg.ppo.gradient_accumulation_steps
        scaled_loss.backward()
        ppo_step_count += 1 # Increment steps within this prompt batch

        # 6. Store Metrics
        current_metrics = {
            'loss/policy': policy_loss.item(),
            'loss/entropy': -entropy_loss.item(), # Store positive entropy
            'loss/total': loss.item(),
            'params/policy_clip_frac': p_clip_frac.item(),
            'params/approx_kl': approx_kl.item(),
        }
        for key, val in current_metrics.items():
            aggregate_metrics.setdefault(key, []).append(val)

        # 7. Optimizer Step (if accumulation cycle complete)
        if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
            grads_exist = any(p.grad is not None for p in actor_model.parameters() if p.requires_grad)
            if grads_exist:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_model.parameters(), max_norm=cfg.ppo.max_grad_norm) # Use ppo.max_grad_norm
                aggregate_metrics.setdefault('params/grad_norm', []).append(grad_norm.item())
                optimizer.step()
                lr_scheduler.step() # Step scheduler with optimizer
            else:
                raise Exception('No gradients found during training.')
            optimizer.zero_grad(set_to_none=True)

    # --- End of Epoch ---
    final_metrics = {key: np.mean(val) for key, val in aggregate_metrics.items() if val}
    return final_metrics


def perform_grpo_updates(
    actor_model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    rollout_buffer: Dict[str, Any],
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Performs multiple GRPO epochs on the collected rollout data."""
    buffer_on_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in rollout_buffer.items()
    }

    if "response_input_ids" not in buffer_on_device or \
       not isinstance(buffer_on_device["response_input_ids"], torch.Tensor) or \
       buffer_on_device["response_input_ids"].numel() == 0:
        logger.warning("No response tokens found in buffer on device. Skipping GRPO update.")
        return {}

    all_epoch_metrics = {}
    for grpo_epoch in range(cfg.ppo.epochs): # Reuse ppo.epochs for now
        epoch_metrics = run_grpo_update_epoch(
            actor_model, optimizer, lr_scheduler, buffer_on_device, cfg, device
        )
        all_epoch_metrics = epoch_metrics # Store last epoch's metrics

    return all_epoch_metrics


# ==============================================================================
# == 6. Training Setup and Orchestration (Adapted for GRPO)
# ==============================================================================

def setup_training_grpo(cfg: DictConfig) -> Tuple[torch.device, str]:
    """Sets random seeds, device, and output directory."""
    # Identical to PPO setup
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.training.seed)
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        if cfg.training.device == "cuda": logger.warning("CUDA requested but unavailable, using CPU.")
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    return device, output_dir

def load_models_and_tokenizer_grpo(cfg: DictConfig, device: torch.device) -> Tuple[
    PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase
]:
    """Loads tokenizer, actor model (standard LM), and reference model."""
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
    # Add quantization if needed (similar to PPO setup)
    # ...

    # --- Load Actor Model (Standard AutoModelForCausalLM) ---
    actor_model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)
    actor_model.to(device)
    if actor_model.config.pad_token_id is None:
        actor_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Actor model loaded.")
    # Enable gradient checkpointing if configured
    if cfg.training.get("gradient_checkpointing", False):
        actor_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for actor model.")

    # --- Load Reference Model (Identical to PPO setup) ---
    ref_model_kwargs = model_kwargs.copy()
    ref_model_kwargs.pop("quantization_config", None)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **ref_model_kwargs)
    ref_model.to(device)
    if ref_model.config.pad_token_id is None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
    for param in ref_model.parameters(): param.requires_grad = False
    ref_model.eval()
    logger.info("Reference model loaded and frozen.")

    return actor_model, ref_model, tokenizer


def setup_optimizer_grpo(cfg: DictConfig, model: nn.Module) -> Tuple[torch.optim.Optimizer, Any]:
    """Sets up the optimizer and LR scheduler for GRPO."""
    use_8bit = cfg.ppo.get("use_8bit_adam", True) # Reuse PPO config for now
    lr = cfg.ppo.learning_rate

    if use_8bit and bnb_available and isinstance(next(model.parameters()).device, torch.device) and next(model.parameters()).device.type == "cuda":
        logger.info("Using 8-bit AdamW Optimizer (bitsandbytes)")
        optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        if use_8bit: logger.warning("8-bit Adam not used (requirements not met). Using standard AdamW.")
        else: logger.info("Using standard AdamW Optimizer")
        optimizer = AdamW(model.parameters(), lr=lr)

    # --- Scheduler Setup ---
    # Calculate total training steps (optimizer steps)
    # Use cfg.grpo.group_size if defined, else maybe default or error
    group_size = cfg.grpo.get("group_size", 4) # Default group size if not in config
    num_prompts_per_rollout = cfg.ppo.rollout_samples # Assume this means number of prompts
    num_mini_batches_per_update = math.ceil(num_prompts_per_rollout / cfg.ppo.mini_batch_size) # Mini-batching over prompts
    num_optimizer_steps_per_ppo_step = math.ceil(num_mini_batches_per_update / cfg.ppo.gradient_accumulation_steps) * cfg.ppo.epochs
    num_training_steps = cfg.training.total_ppo_steps * num_optimizer_steps_per_ppo_step

    # Use scheduler params from ppo config section for now
    scheduler_name = cfg.ppo.get("scheduler", "linear") # Default to linear if not specified
    warmup_steps = cfg.ppo.get("warmup_steps", 0)
    min_lr_ratio = cfg.ppo.get("min_lr_ratio", 0.0) # Ratio of initial LR for min_lr
    min_lr_abs = cfg.ppo.get("min_lr", None) # Absolute min_lr takes precedence

    scheduler_kwargs = {}
    if scheduler_name == "cosine_with_min_lr" and min_lr_abs is not None:
         scheduler_name = "cosine" # Transformers uses 'cosine'
         # min_lr might need custom handling or wrapper if get_scheduler doesn't support it directly
         # For now, we just use cosine decay towards 0
         logger.warning("Cosine scheduler with specific min_lr might need custom handling. Using standard cosine decay.")
    elif scheduler_name == "cosine_with_min_lr":
         scheduler_name = "cosine"
         logger.warning("Cosine scheduler specified but no min_lr. Using standard cosine decay.")


    logger.info(f"Setting up LR scheduler: {scheduler_name} with {warmup_steps} warmup steps.")
    logger.info(f"Total optimizer steps calculated: {num_training_steps}")

    lr_scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
        # Add scheduler_specific_kwargs if needed and supported
    )
    return optimizer, lr_scheduler

def create_generation_config_grpo(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase) -> GenerationConfig:
     """Creates the GenerationConfig object for GRPO."""
     # Identical to PPO, just ensure sampling is enabled
     return GenerationConfig(
        max_new_tokens=cfg.generation.max_new_tokens,
        min_new_tokens=cfg.generation.min_new_tokens,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        do_sample=True, # Ensure sampling for group generation
        pad_token_id=tokenizer.pad_token_id,
        # num_return_sequences will be set dynamically in generate_responses_grouped
    )

def save_model_grpo(model: nn.Module, tokenizer: PreTrainedTokenizerBase, save_path: str):
    """Saves the GRPO model and tokenizer."""
    # Identical to PPO saving
    logger.info(f"Saving model checkpoint to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    # No base_model attribute if using AutoModelForCausalLM directly
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved.")


# ==============================================================================
# == 7. Main Training Orchestration (Adapted for GRPO)
# ==============================================================================

def train_grpo(cfg: DictConfig):
    """Main GRPO training loop."""
    # --- 1. Initial Setup ---
    device, output_dir = setup_training_grpo(cfg)
    OmegaConf.save(cfg, os.path.join(output_dir, "effective_config_grpo.yaml"))

    if cfg.wandb.get("report_to_wandb", False): # Check existence safely
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.get("name", None)
        )

    # --- 2. Load Models and Tokenizer ---
    actor_model, ref_model, tokenizer = load_models_and_tokenizer_grpo(cfg, device)

    # --- 3. Load and Preprocess Dataset ---
    processed_dataset = load_and_preprocess_dataset(cfg, tokenizer) # Reusable

    # --- 4. Setup Optimizer & Scheduler ---
    optimizer, lr_scheduler = setup_optimizer_grpo(cfg, actor_model)

    # --- 5. Generation Config ---
    gen_config = create_generation_config_grpo(cfg, tokenizer)

    # --- 6. Collate Function (Reusable from PPO) ---
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        padded_inputs = tokenizer.pad({"input_ids": input_ids}, padding='longest', return_tensors="pt", return_attention_mask=True)
        ground_truths = [item['ground_truth_answer'] for item in batch]
        return {"prompt_input_ids": padded_inputs["input_ids"],
                "prompt_attention_mask": padded_inputs["attention_mask"],
                "ground_truth_answers": ground_truths}

    # --- 7. Main GRPO Loop ---
    logger.info("\n--- Starting GRPO Training ---")
    group_size = cfg.grpo.get("group_size", 4) # Get group size from config (add default)
    rollout_prompts = cfg.ppo.get("rollout_samples", 512) # Number of prompts per rollout
    logger.info(f"Using Group Size (G): {group_size}")

    for grpo_step in range(cfg.training.total_ppo_steps): # Reuse total_ppo_steps
        logger.info(f"\n===== GRPO Step {grpo_step + 1}/{cfg.training.total_ppo_steps} =====")

        # --- Phase 1: Rollout (Grouped) ---
        logger.info("Phase 1: Generating Rollouts...")
        # Select prompts for this rollout step
        # Ensure we don't select more prompts than available in the dataset
        num_prompts_to_select = min(rollout_prompts, len(processed_dataset))
        if num_prompts_to_select < rollout_prompts:
            logger.warning(f"Requested {rollout_prompts} rollout samples, but dataset only has {len(processed_dataset)}. Using {num_prompts_to_select}.")

        # Create dataloader for selected prompts
        prompt_dataloader = DataLoader(
            processed_dataset.shuffle(seed=cfg.training.seed + grpo_step).select(range(num_prompts_to_select)),
            batch_size=cfg.ppo.batch_size, # Batch size for prompts
            shuffle=False, # Already shuffled by select
            collate_fn=collate_fn
        )

        rollout_buffer = perform_rollouts_grpo(
            actor_model, ref_model, tokenizer, prompt_dataloader, gen_config, group_size, device
        )

        # Validate rollout buffer
        if not rollout_buffer or "rewards" not in rollout_buffer or \
           not isinstance(rollout_buffer["rewards"], torch.Tensor) or \
           rollout_buffer["rewards"].numel() == 0:
            logger.warning("Invalid rollout buffer generated. Skipping update."); continue

        # Calculate average reward across all generated samples (B*G)
        avg_reward = rollout_buffer["rewards"].mean().item()
        num_generated_samples = rollout_buffer["rewards"].shape[0]
        num_input_prompts = rollout_buffer["prompt_input_ids"].shape[0]
        avg_resp_len = rollout_buffer.get("avg_response_length", 0.0)
        rollout_duration = rollout_buffer.get("rollout_duration_seconds", 0.0)
        # Log timing breakdown
        gen_time = rollout_buffer.get("timing/total_gen_time", 0.0)
        stats_time = rollout_buffer.get("timing/total_stats_time", 0.0)
        cpu_time = rollout_buffer.get("timing/total_cpu_time", 0.0)
        collation_time = rollout_buffer.get("timing/collation_time", 0.0)

        logger.info(
            f"Rollout complete ({num_input_prompts} prompts -> {num_generated_samples} samples). "
            f"Avg Reward: {avg_reward:.4f}, Avg Resp Len: {avg_resp_len:.2f}, "
            f"Duration: {rollout_duration:.2f}s "
            f"(Gen: {gen_time:.2f}s, Stats: {stats_time:.2f}s, CPU: {cpu_time:.2f}s, Collate: {collation_time:.2f}s)"
        )

        # --- Phase 2: Update ---
        logger.info("Phase 2: Performing GRPO Updates...")
        metrics = perform_grpo_updates(
            actor_model, optimizer, lr_scheduler, rollout_buffer, cfg, device
        )

        # Log metrics
        log_data = {}
        if metrics:
            log_data.update(metrics)
            log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Update Metrics (Avg over Epoch): {log_str}")
            logger.info(f"  Rollout Reward (Avg over Samples): {avg_reward:.4f}")
        else: logger.warning("GRPO update skipped or failed.")

        # Add rollout metrics for logging
        log_data["rollout/reward_mean"] = avg_reward
        log_data["rollout/avg_response_length"] = avg_resp_len
        log_data["rollout/duration_seconds"] = rollout_duration
        log_data["rollout/timing_gen_seconds"] = gen_time
        log_data["rollout/timing_stats_seconds"] = stats_time
        log_data["rollout/timing_cpu_seconds"] = cpu_time
        log_data["rollout/timing_collate_seconds"] = collation_time

        if cfg.wandb.get("report_to_wandb", False) and metrics:
            wandb.log(log_data, step=grpo_step)

        # --- Phase 3: Save Checkpoint ---
        if (grpo_step + 1) % cfg.training.save_interval == 0:
            save_model_grpo(actor_model, tokenizer, os.path.join(output_dir, f"step_{grpo_step + 1}"))

    # --- 8. Final Save ---
    if cfg.wandb.get("report_to_wandb", False):
        wandb.finish()
    logger.info("\n--- GRPO Training Finished ---")
    save_model_grpo(actor_model, tokenizer, os.path.join(output_dir, "final"))


# ==============================================================================
# == 8. Command-Line Interface Logic (Adapted for GRPO Config)
# ==============================================================================

def load_config_with_cli_overrides_grpo() -> DictConfig:
    """Loads OmegaConf config, handling defaults and CLI overrides."""
    # Basic CLI parsing (can reuse PPO's if structure is similar)
    parser = argparse.ArgumentParser(description="GRPO RL Trainer")
    parser.add_argument("--config", required=True, help="Path to the GRPO config file (e.g., configs/grpo_config.yaml)")
    parser.add_argument("overrides", nargs="*", help="Key=value config overrides")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Error: Config file not found at '{config_path}'.")
        sys.exit(1)

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
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("---------------------------------------")
    return cfg


# ==============================================================================
# == 9. Entry Point
# ==============================================================================

if __name__ == "__main__":
    config = load_config_with_cli_overrides_grpo()
    train_grpo(config)
