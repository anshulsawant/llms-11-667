# src/ppo_trainer_refactored_exercise.py
# -*- coding: utf-8 -*-
"""
Refactored PPO Trainer script - STUDENT EXERCISE VERSION.
Focuses on modularity and clarity. Students need to implement
the core PPO algorithm components marked with 'EXERCISE'.
"""

import torch
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
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding # Can be useful if not using custom collate
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
# == 1. Helper Functions (Masking, Reward, Padding) - PROVIDED
# ==============================================================================

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    """Calculates mean of tensor elements specified by mask."""
    if mask is None:
        return torch.mean(tensor, dim=dim)
    mask = mask.bool()
    # Expand mask dimensions if necessary
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor) # Ensure shapes match

    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8) # Add epsilon for stability
    return mean

def masked_whiten(tensor: torch.Tensor, mask: Optional[torch.Tensor], shift_mean: bool = True) -> torch.Tensor:
    """Whitens the tensor values specified by the mask."""
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_variance = torch.where(mask, (tensor - mean)**2, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    variance = masked_mean(masked_tensor_variance, mask, dim=None)
    std = torch.sqrt(variance + 1e-8) # Add epsilon for stability

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(mask, whitened, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))

def extract_gsm8k_solution(solution_str: str) -> Optional[str]:
    """Extracts the numerical answer from the #### format."""
    solution = re.search(r"####\s*([-+]?\s*[\d\.\,]+)", solution_str)
    if solution:
        return solution.group(1).replace(',', '').replace(' ', '')
    else: # Fallback: try finding the last number
        answer = re.findall(r"([-+]?\s*[\d\.\,]+)", solution_str)
        if answer:
            final_answer_str = answer[-1].replace(',', '').replace(' ', '')
            try: # Validate if it's a number
                float(final_answer_str)
                return final_answer_str
            except ValueError: return None
        else: return None

def compute_gsm8k_reward(generated_text: str, ground_truth_str: str) -> float:
    """Computes reward: 1.0 if extracted answer matches ground truth, 0 otherwise."""
    extracted_answer_str = extract_gsm8k_solution(generated_text)
    if extracted_answer_str is None: return 0.0
    try:
        extracted_answer = float(extracted_answer_str)
        ground_truth = float(ground_truth_str)
        return 1.0 if math.isclose(extracted_answer, ground_truth) else 0.0
    except ValueError: return 0.0 # Handle non-numeric cases

def pad_and_collate_tensors(
    tensor_list: List[torch.Tensor],
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pads tensors in a list to the maximum length of the second dimension
    and concatenates them along the first dimension.

    Assumes input tensors are at least 2D.
    """
    if not tensor_list:
        return torch.empty(0) # Or determine appropriate shape based on context

    if not all(isinstance(t, torch.Tensor) for t in tensor_list):
         raise TypeError("All elements in tensor_list must be PyTorch tensors.")

    if tensor_list[0].dim() < 2:
        # If tensors are 1D (like rewards), just concatenate
        if all(t.dim() == 1 for t in tensor_list):
            return torch.cat(tensor_list, dim=0)
        else:
            raise ValueError("Cannot collate tensors with mixed dimensions < 2.")

    # Find max length in the second dimension (sequence length)
    max_len = 0
    for t in tensor_list:
        if t.dim() < 2:
             raise ValueError(f"Tensor with shape {t.shape} has fewer than 2 dimensions.")
        max_len = max(max_len, t.shape[1])

    if max_len == 0: # Handle cases where all sequences have length 0
        total_batch_size = sum(t.shape[0] for t in tensor_list)
        # Return shape (TotalB, 0, ...) matching original dims > 1
        original_shape = tensor_list[0].shape
        return torch.empty((total_batch_size, 0) + original_shape[2:],
                            dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    # Pad each tensor and collect in a new list
    padded_list = []
    for t in tensor_list:
        current_len = t.shape[1]
        padding_needed = max_len - current_len
        if padding_needed > 0:
            # Pad only the second dimension (dim=1) on the right
            num_dims = t.dim()
            # F.pad format: (pad_last_dim_left, pad_last_dim_right, pad_next_to_last_left, ...)
            pad_tuple = [0] * (2 * num_dims)
            pad_idx_right = 2 * (num_dims - 1 - 1) + 1 # Index for right padding of dim 1
            pad_tuple[pad_idx_right] = padding_needed
            padded_t = F.pad(t, tuple(pad_tuple), mode='constant', value=padding_value)
            padded_list.append(padded_t)
        else:
            padded_list.append(t) # No padding needed or already max length

    # Concatenate the padded tensors along the batch dimension (dim=0)
    try:
        return torch.cat(padded_list, dim=0)
    except Exception as e:
        print(f"Error during final concatenation: {e}")
        # Print shapes for debugging
        for i, p_t in enumerate(padded_list): print(f"  Tensor {i} shape: {p_t.shape}")
        raise # Re-raise the exception after printing info


# ==============================================================================
# == 2. Core PPO Algorithm Components - EXERCISES
# ==============================================================================

# === Recommended Order for Implementing PPO Exercises & The Big Picture ===
#
# This comment block outlines a suggested order for the PPO exercises below
# and explains how these components interconnect within the overall RL training loop.
#
# --- Recommended Implementation Order ---
#
# 1. `compute_policy_loss`:
#    - Why first? This is the core PPO objective function, dictating how the actor learns.
#      Understanding the clipped surrogate objective is fundamental.
#    - Focus: Implement the ratio calculation, clipped/unclipped objectives, and masked mean.
#
# 2. `compute_value_loss`:
#    - Why second? This trains the critic, whose value estimates V(s) are needed for
#      stable and efficient policy updates (via advantage calculation). Learning how V(s)
#      is trained against the target 'returns' is key.
#    - Focus: Implement value clipping and the MSE calculation against returns.
#
# 3. `compute_entropy_loss`:
#    - Why third? A simpler but important component for encouraging exploration.
#    - Focus: Calculate policy distribution entropy from logits and take the masked mean.
#
# 4. `compute_gae_advantages`:
#    - Why fourth? This calculates the 'advantages' (how good were the actions?) and
#      'returns' (what's the target for the value function?) needed by the policy and
#      value loss functions. Implementing it after the losses provides context for why these
#      inputs are calculated this way (incorporating rewards, KL penalty, and value estimates).
#    - Focus: Implement the backward GAE loop, combining rewards/KL/values. Whiten advantages.
#
# 5. `run_ppo_update_epoch` (The Mini-Batch Update Logic):
#    - Why last? This orchestrates the learning step within an epoch, bringing together the rollout
#      data, GAE calculation (done before epoch), and all the loss functions you've implemented.
#    - Focus: Implement the forward pass, slicing, loss calculations, and backward pass
#      within the mini-batch loop inside this function.
#
# --- The Big Picture: How It Fits Together ---
#
# The PPO algorithm iterates through a cycle of experience gathering and policy improvement:
#
# A. Rollout Phase (`perform_rollouts`):
#    - The current Actor model generates responses (trajectories) based on input prompts.
#    - During generation, we store crucial data: prompts, responses, `logprobs` (Actor),
#      `values` (Critic), `ref_logprobs` (Reference), `rewards` (Task).
#
# B. Advantage Calculation Phase (within `run_ppo_update_epoch`'s caller):
#    - The collected rollout data (`rewards`, `values`, `logprobs`, `ref_logprobs`) is processed.
#    - First, KL penalties are calculated (`logprobs - ref_logprobs`).
#    - Then, `compute_gae_advantages` uses the rewards, KL penalties, and values to estimate:
#      - `advantages`: How much better were actions than expected? (Incorporates KL penalty).
#      - `returns`: What was the actual observed discounted reward-to-go? (Target for Critic).
#
# C. Update Phase (`perform_ppo_updates` calls `run_ppo_update_epoch`):
#    - Uses rollout data and calculated advantages/returns to update the model.
#    - Loops for multiple `ppo_epochs` over the *same* rollout data.
#    - `run_ppo_update_epoch` iterates over mini-batches:
#      - Re-evaluates sequences with the *current* Actor/Critic (`logprobs_new`, `values_new`).
#      - Calculates the three core losses (`compute_policy_loss`, `compute_value_loss`, `compute_entropy_loss`).
#      - Combines losses into a single objective.
#      - Performs gradient descent (`backward()`, `optimizer.step()`) to update Actor/Critic.
#
# D. Repeat: The entire cycle (Rollout -> GAE -> Update) repeats.
# =========================================================

# --- EXERCISE 1 START ---
def compute_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EXERCISE: Implement PPO policy loss (clipped surrogate objective).
    WHY Clipped Objective?: Standard policy gradient updates `log_prob * Advantage`. If the advantage
              or the change in policy (log_prob ratio) is large, this can lead to destructively
              large updates. PPO introduces clipping the probability ratio `r = exp(log_new - log_old)`
              within `[1-clip_ratio, 1+clip_ratio]`. This prevents the new policy from moving
              too far from the old policy in a single update, stabilizing learning. The loss
              takes the minimum of the clipped and unclipped objectives, meaning we only take
              the pessimistic (smaller improvement or larger decrease) estimate when the policy change is large.

    Args:
        log_probs_new (torch.Tensor): Log probabilities from current policy. Shape (batch, resp_len).
        log_probs_old (torch.Tensor): Log probabilities from rollout policy. Shape (batch, resp_len).
        advantages (torch.Tensor): Calculated GAE advantages. Shape (batch, resp_len).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len).
        clip_ratio (float): PPO clipping parameter (epsilon).

    Steps:
    1. Calculate the probability ratio: `ratio = exp(log_probs_new - log_probs_old)`. Clamp log_ratio first for stability.
    2. Calculate surrogate objective 1: `surr1 = ratio * advantages`.
    3. Calculate surrogate objective 2: `surr2 = clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages`.
    4. The policy loss is the negative mean of the minimum of surr1 and surr2.
       - Use `masked_mean` with `response_mask` to average only over valid tokens.
    5. (Optional but Recommended) Calculate `clip_frac` (fraction of samples clipped) and `approx_kl` (KL divergence) for logging.

    Returns:
        policy_loss (torch.Tensor): Scalar policy loss.
        clip_frac (torch.Tensor): Scalar fraction of clipped samples.
        approx_kl (torch.Tensor): Scalar approximate KL divergence.
    """
    policy_loss = torch.tensor(0.0, device=log_probs_new.device, requires_grad=True) # Ensure grad enabled
    clip_frac = torch.tensor(0.0, device=log_probs_new.device)
    approx_kl = torch.tensor(0.0, device=log_probs_new.device)

    # <<<< YOUR POLICY LOSS IMPLEMENTATION HERE >>>>
    # Hints: Use torch.exp, torch.clamp, torch.min, masked_mean.
    # Remember the negative sign for the final loss.
    # Use torch.no_grad() context for clip_frac and approx_kl calculation.
    # Ensure advantages shape matches log_probs_old shape.
    if advantages.shape != log_probs_old.shape:
         print(f"Warning: Shape mismatch in compute_policy_loss. Advantages: {advantages.shape}, LogProbs Old: {log_probs_old.shape}")
         # Handle shape mismatch or raise error if necessary

    print("Warning: compute_policy_loss not implemented!") # Remove this line

    return policy_loss, clip_frac, approx_kl
# --- EXERCISE 1 END ---


# --- EXERCISE 2 START ---
def compute_value_loss(
    values_new: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range_value: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EXERCISE: Implement PPO value loss (clipped).
    WHY Clipped Value Loss?: Similar to the policy loss, clipping the value function update helps
               stabilize training. It prevents the value function (critic) from changing too
               drastically based on potentially noisy return estimates from a single batch.
               The loss is based on the squared error between the predicted value `values_new`
               and the target `returns`, but the prediction is clipped around the `values_old`
               estimate from the rollout phase.

    Args:
        values_new (torch.Tensor): Values predicted by the current critic. Shape (batch, resp_len).
        values_old (torch.Tensor): Values predicted by the critic during rollout. Shape (batch, resp_len).
        returns (torch.Tensor): Calculated returns (target for value function). Shape (batch, resp_len).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len).
        clip_range_value (float): Clipping parameter for value loss.

    Steps:
    1. Clip the `values_new` based on `values_old`:
       `values_pred_clipped = values_old + clamp(values_new - values_old, -clip_range_value, clip_range_value)`.
    2. Calculate squared error loss for unclipped values: `vf_loss1 = (values_new - returns)^2`.
    3. Calculate squared error loss for clipped values: `vf_loss2 = (values_pred_clipped - returns)^2`.
    4. The value loss is `0.5 * mean(max(vf_loss1, vf_loss2))`.
       - Use `masked_mean` with `response_mask` for averaging.
    5. (Optional) Calculate `vf_clip_frac` (fraction of samples where clipped loss was used) for logging.

    Returns:
        value_loss (torch.Tensor): Scalar value loss.
        vf_clip_frac (torch.Tensor): Scalar fraction of clipped value samples.
    """
    value_loss = torch.tensor(0.0, device=values_new.device, requires_grad=True) # Ensure grad enabled
    vf_clip_frac = torch.tensor(0.0, device=values_new.device)

    # <<<< YOUR VALUE LOSS IMPLEMENTATION HERE >>>>
    # Hints: Use torch.clamp, torch.max, masked_mean. Remember the 0.5 factor.
    # Use torch.no_grad() context for vf_clip_frac calculation.

    print("Warning: compute_value_loss not implemented!") # Remove this line

    return value_loss, vf_clip_frac
# --- EXERCISE 2 END ---


# --- EXERCISE 3 START ---
def compute_entropy_loss(logits_new: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """
    EXERCISE: Implement entropy loss calculation.
    WHY Entropy Bonus?: Entropy measures the randomness or uncertainty of the policy's action
             distribution. Adding an entropy bonus (or subtracting an entropy loss) encourages
             the policy to explore by making its action choices less deterministic. This can
             prevent the policy from collapsing prematurely to a suboptimal strategy. The
             `entropy_coeff` controls the strength of this exploration incentive.

    Args:
        logits_new (torch.Tensor): Logits predicted by the current policy. Shape (batch, resp_len, vocab_size).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len).

    Steps:
    1. Calculate the categorical distribution from `logits_new`. `torch.distributions.Categorical` is recommended.
       - Ensure logits are float32 for stability: `logits_new.float()`.
    2. Compute the entropy of the distribution. The result should have shape (batch, resp_len).
    3. Calculate the masked mean of the entropy using `response_mask`.
    4. The entropy loss is the negative of the mean entropy (since we want to maximize entropy).

    Returns:
        entropy_loss (torch.Tensor): Scalar entropy loss.
    """
    entropy_loss = torch.tensor(0.0, device=logits_new.device, requires_grad=True) # Ensure grad enabled

    # <<<< YOUR ENTROPY LOSS IMPLEMENTATION HERE >>>>
    # Hints: Use torch.distributions.Categorical(logits=...).entropy().
    # Remember to use masked_mean and the negative sign.

    print("Warning: compute_entropy_loss not implemented!") # Remove this line

    return entropy_loss
# --- EXERCISE 3 END ---


# --- EXERCISE 4 START ---
def compute_gae_advantages(
    final_rewards: torch.Tensor, # Shape (batch_size,)
    kl_penalties: torch.Tensor, # Shape (batch_size, resp_len)
    values: torch.Tensor,       # Shape (batch_size, resp_len) - V(s_t) used for return G_t
    response_mask: torch.Tensor,# Shape (batch_size, resp_len)
    gamma: float,
    lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EXERCISE: Implement Generalized Advantage Estimation (GAE).
    WHY GAE?: Standard reward-to-go sums rewards but can have high variance. Value functions
             (like our critic) estimate expected future rewards, reducing variance but potentially
             adding bias. GAE provides a tunable knob (lambda) to balance this trade-off,
             aiming for lower variance advantage estimates than reward-to-go while being less
             biased than using only the value function difference.
    WHY incorporate KL?: In LLM PPO, we often add a KL penalty (difference between actor and
             reference model logprobs) to the reward signal *before* calculating advantages.
             This discourages the actor from deviating too much from the (presumably safer)
             reference policy during exploration, stabilizing training.

    Args:
        final_rewards (torch.Tensor): Shape (batch_size,) - Task reward (e.g., 1.0 for correct GSM8K answer).
        kl_penalties (torch.Tensor): Shape (batch_size, response_length) - KL penalty per token (kl_coeff * kl_divergence).
        values (torch.Tensor): Shape (batch_size, response_length) - Critic's state value estimates (V(s_t)).
        response_mask (torch.Tensor): Shape (batch_size, response_length) - Mask for valid response tokens.
        gamma (float): Discount factor for future rewards.
        lam (float): GAE lambda parameter for bias-variance trade-off.

    Steps:
    1. Initialize `advantages` tensor.
    2. Construct `token_level_rewards`: Start with zeros, apply `final_rewards` at the last valid token position
       (use `response_mask` to find it) and subtract `kl_penalties` at each step.
    3. Iterate backwards through the sequence length (`t` from `response_length - 1` down to `0`).
    4. Calculate `delta_t = token_level_rewards_t + gamma * V(s_{t+1}) * mask_{t+1} - V(s_t) * mask_t`.
       - Handle boundary condition: `V(s_T) = 0` if `t` is the last token (`t+1` is out of bounds).
       - Use `response_mask` to ensure values/rewards are zero after sequence end.
    5. Calculate `advantage_t = delta_t + gamma * lambda * advantage_{t+1} * mask_{t+1}`. (This is `last_gae_lam` in the loop).
       - Use `response_mask` at `t+1` to reset advantage contribution after sequence end.
    6. Store calculated advantages (append to a list and reverse later, or fill tensor).
    7. Compute `returns = advantages + values`. (Make sure to use the original `values` here).
    8. Whiten (normalize) the calculated `advantages` using the `masked_whiten` helper function and `response_mask`.

    Returns:
        advantages (torch.Tensor): Shape (batch_size, response_length) - Whitened advantages.
        returns (torch.Tensor): Shape (batch_size, response_length) - Calculated returns.
    """
    advantages = torch.zeros_like(values) # Placeholder
    returns = torch.zeros_like(values)    # Placeholder

    # --- IMPORTANT: GAE calculation should not track gradients ---
    with torch.no_grad():
        # <<<< YOUR GAE IMPLEMENTATION HERE >>>>
        # Hints:
        # - Loop backwards from `response_length - 1` down to `0`.
        # - Keep track of `last_gae_lam` (advantage estimate for t+1).
        # - Get `next_values` carefully, handling the edge case `t == response_length - 1`.
        # - Use `response_mask` correctly when accessing values and propagating `last_gae_lam`.
        # - Remember to whiten the final advantages.

        print("Warning: compute_gae_advantages not implemented!") # Remove this line

        # Dummy implementation for shape consistency during testing:
        advantages = masked_whiten(torch.randn_like(values), response_mask)
        returns = advantages + values

    return advantages, returns
# --- EXERCISE 4 END ---


# ==============================================================================
# == 3. Actor Model Definition - PROVIDED
# ==============================================================================

class ActorModelWithValueHead(nn.Module):
    """
    Wraps a pre-trained transformer, adding a value head and generation method.
    Computes per-token values.
    """
    def __init__(self, model_name_or_path: str, **kwargs_model_load):
        """Initializes the model, loading the base transformer and value head."""
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs_model_load)
        self.config = self.base_model.config # Store config
        # Value head maps hidden states to scalar value
        self.value_head = nn.Linear(self.config.hidden_size, 1)
        # Basic initialization for value head
        self.value_head.weight.data.normal_(mean=0.0, std=0.01)
        self.value_head.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: computes logits and per-token values."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, # Need hidden states for value head
            **kwargs
        )
        logits = outputs.logits
        # Get last hidden state (batch, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        # Compute value for each token's hidden state
        values = self.value_head(last_hidden_state).squeeze(-1) # Shape: (batch, seq_len)
        return logits, values

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Forwards generate call to the base model."""
        return self.base_model.generate(*args, **kwargs)


# ==============================================================================
# == 4. Rollout Phase Logic - PROVIDED (Uses Modular Functions)
# ==============================================================================

def generate_responses(
    model: ActorModelWithValueHead,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates responses for a batch of prompts."""
    model.eval() # Ensure model is in eval mode for generation
    with torch.no_grad():
        generated_output = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=gen_config,
            pad_token_id=tokenizer.pad_token_id # Important for generation
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
    prompt_ids: torch.Tensor,      # Shape (batch, prompt_len)
    prompt_mask: torch.Tensor,     # Shape (batch, prompt_len)
    response_ids: torch.Tensor,    # Shape (batch, resp_len)
    response_mask: torch.Tensor    # Shape (batch, resp_len)
) -> Dict[str, torch.Tensor]:
    """Calculates logprobs, ref_logprobs, values for a batch."""
    actor_model.eval()
    ref_model.eval()
    with torch.no_grad():
        # Combine prompt and response for forward passes
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        full_mask = torch.cat((prompt_mask, response_mask), dim=1)

        # Get actor logits and values
        actor_logits, actor_values = actor_model(full_ids, attention_mask=full_mask)
        # Get reference model logits
        ref_logits = ref_model(full_ids, attention_mask=full_mask).logits

        # --- Calculate Logprobs and Values for the RESPONSE part ---
        prompt_len = prompt_ids.shape[1]
        resp_len = response_ids.shape[1]
        full_len = full_ids.shape[1]

        # Logits/Values indices: We need state BEFORE generating token R_t
        # Corresponds to indices from prompt_len-1 to full_len-2
        start_idx = prompt_len - 1
        end_idx = full_len - 1 # Slice up to (but not including) this index

        if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
            # Handle cases with empty response or invalid indices
            logprobs = torch.empty((prompt_ids.shape[0], 0), dtype=torch.float, device=prompt_ids.device)
            ref_logprobs = torch.empty((prompt_ids.shape[0], 0), dtype=torch.float, device=prompt_ids.device)
            values = torch.empty((prompt_ids.shape[0], 0), dtype=torch.float, device=prompt_ids.device)
        else:
            logits_resp = actor_logits[:, start_idx:end_idx, :]
            ref_logits_resp = ref_logits[:, start_idx:end_idx, :]
            values = actor_values[:, start_idx:end_idx] # Values for states BEFORE response tokens

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
                response_mask_adjusted = response_mask[:,:min_len]
            else:
                 response_mask_adjusted = response_mask


            # Calculate log probabilities
            logprobs_all = F.log_softmax(logits_resp, dim=-1)
            ref_logprobs_all = F.log_softmax(ref_logits_resp, dim=-1)
            logprobs = torch.gather(logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            ref_logprobs = torch.gather(ref_logprobs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Apply mask (mask should match the potentially adjusted length)
            logprobs = logprobs * response_mask_adjusted
            ref_logprobs = ref_logprobs * response_mask_adjusted
            values = values * response_mask_adjusted

    return {
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
        "values": values,
    }


def perform_rollouts(
    actor_model: ActorModelWithValueHead,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_dataloader: DataLoader,
    gen_config: GenerationConfig,
    device: torch.device
) -> Dict[str, Any]:
    """Generates responses and computes stats for PPO update."""
    # Temporary buffer to store results from each batch before collation
    buffer_lists = {
        "prompt_input_ids": [], "prompt_attention_mask": [],
        "response_input_ids": [], "response_attention_mask": [],
        "logprobs": [], "ref_logprobs": [], "values": [],
        "rewards": [], "full_texts": [], "ground_truth_answers": []
    }

    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        if batch is None: # Handle potential error from collate_fn
             print("Warning: Skipping None batch from dataloader.")
             continue
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"] # List of strings

        # 1. Generate responses
        response_ids, response_mask = generate_responses(
            actor_model, tokenizer, prompt_ids, prompt_mask, gen_config
        ) # Shapes: (B, R_i), (B, R_i)

        # 2. Calculate stats (logprobs, values, etc.)
        stats = calculate_rollout_stats(
            actor_model, ref_model, tokenizer,
            prompt_ids, prompt_mask, response_ids, response_mask
        ) # Dict of tensors (B, R_i)

        # 3. Decode texts and calculate rewards
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        full_decoded_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True)
        rewards = torch.tensor(
            [compute_gsm8k_reward(txt, gt) for txt, gt in zip(full_decoded_texts, ground_truths)],
            dtype=torch.float32, device='cpu' # Calculate reward on CPU
        ) # Shape: (B,)

        # 4. Append results to buffer lists (moving tensors to CPU)
        buffer_lists["prompt_input_ids"].append(prompt_ids.cpu())
        buffer_lists["prompt_attention_mask"].append(prompt_mask.cpu())
        buffer_lists["response_input_ids"].append(response_ids.cpu())
        buffer_lists["response_attention_mask"].append(response_mask.cpu())
        buffer_lists["logprobs"].append(stats["logprobs"].cpu())
        buffer_lists["ref_logprobs"].append(stats["ref_logprobs"].cpu())
        buffer_lists["values"].append(stats["values"].cpu())
        buffer_lists["rewards"].append(rewards) # Already on CPU
        buffer_lists["full_texts"].extend(full_decoded_texts)
        buffer_lists["ground_truth_answers"].extend(ground_truths)

    # --- Collate the buffer lists into single tensors ---
    collated_buffer = {}
    padding_value_map = {
        "input_ids": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "attention_mask": 0,
        "logprobs": 0.0, "ref_logprobs": 0.0, "values": 0.0,
    }
    keys_to_pad_and_cat = [
        "prompt_input_ids", "prompt_attention_mask",
        "response_input_ids", "response_attention_mask",
        "logprobs", "ref_logprobs", "values"
    ]

    for key, data_list in buffer_lists.items():
        if key in ["full_texts", "ground_truth_answers"]:
            collated_buffer[key] = data_list # Keep as list
        elif not data_list:
            collated_buffer[key] = torch.empty(0) # Handle empty list
        elif key == "rewards":
            collated_buffer[key] = torch.cat(data_list, dim=0) # Simple concat for 1D rewards
        elif key in keys_to_pad_and_cat:
            # Determine padding value
            pad_val = 0.0
            for suffix, val in padding_value_map.items():
                if key.endswith(suffix): pad_val = val; break
            # Pad list elements to max seq len and concatenate
            try:
                collated_buffer[key] = pad_and_collate_tensors(data_list, padding_value=pad_val)
            except (TypeError, ValueError) as e:
                 print(f"Error during collation padding/concat for key '{key}': {e}")
                 collated_buffer[key] = torch.empty(0) # Assign empty on error
        else:
            print(f"Warning: Unexpected key '{key}' in buffer collation.")
            collated_buffer[key] = data_list # Keep as is

    return collated_buffer


# ==============================================================================
# == 5. PPO Update Phase Logic - EXERCISE in run_ppo_update_epoch
# ==============================================================================

# --- EXERCISE 5 START (Inside run_ppo_update_epoch) ---
def run_ppo_update_epoch(
    actor_model: ActorModelWithValueHead,
    optimizer: torch.optim.Optimizer,
    collated_buffer: Dict[str, torch.Tensor], # Assumes tensors are on correct device
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Runs one PPO epoch with mini-batch updates."""
    actor_model.train()
    aggregate_metrics = {}
    ppo_step_count = 0 # For gradient accumulation tracking

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
        # --- USES EXERCISE 4 ---
        advantages, returns = compute_gae_advantages(
            final_rewards, kl_penalties, values_old, response_mask,
            cfg.ppo.gamma, cfg.ppo.lam
        )

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

        # <<<< YOUR PPO MINI-BATCH UPDATE IMPLEMENTATION HERE >>>>
        # Steps:
        # 1. Forward Pass: Get new logits and values from actor_model for the mini-batch.
        #    logits_new, values_new = actor_model(batch_full_ids, attention_mask=batch_full_mask)
        #
        # 2. Calculate New Logprobs: Extract logits corresponding to response tokens.
        #    Be careful with slicing indices (use prompt_len, resp_len).
        #    Calculate log_softmax and gather logprobs for `batch_response_tokens`.
        #    logprobs_new = gather(log_softmax(logits_new_resp), labels=batch_response_tokens)
        #    Shape: (mini_batch, resp_len)
        #
        # 3. Extract New Values: Slice `values_new` to get values corresponding to response states.
        #    values_new_response = values_new[:, start_idx:end_idx]
        #    Shape: (mini_batch, resp_len)
        #
        # 4. Apply Mask: Apply `batch_response_mask` to `logprobs_new` and `values_new_response`.
        #
        # 5. Calculate Losses: Call your implemented loss functions (Exercises 1, 2, 3).
        #    - policy_loss, p_clip_frac, approx_kl = compute_policy_loss(...)
        #    - value_loss, v_clip_frac = compute_value_loss(...)
        #    - entropy_loss = compute_entropy_loss(...)
        #    *CRITICAL*: Ensure masks and tensor slices passed to loss functions are correct.
        #
        # 6. Combine Losses: Use coefficients from `cfg.ppo`.
        #    loss = policy_loss + cfg.ppo.vf_coeff * value_loss + cfg.ppo.entropy_coeff * entropy_loss
        #
        # 7. Backward Pass & Grad Accumulation:
        #    scaled_loss = loss / cfg.ppo.gradient_accumulation_steps
        #    scaled_loss.backward()
        #
        # 8. Store Metrics: Append .item() of losses/stats for logging.
        #    current_metrics = {'loss/policy': policy_loss.item(), ...}
        #    aggregate_metrics.setdefault(key, []).append(val)
        #
        # 9. Optimizer Step (Handled outside this section based on grad accum count)

        print(f"Warning: PPO Mini-batch update logic {i // cfg.ppo.mini_batch_size} not implemented!") # Remove this line
        # Dummy forward/backward to allow script execution without full implementation
        try:
            logits_new, values_new = actor_model(batch_full_ids, attention_mask=batch_full_mask)
            dummy_loss = logits_new.mean() * 0.0 + values_new.mean() * 0.0 # Dummy loss based on outputs
            scaled_loss = dummy_loss / cfg.ppo.gradient_accumulation_steps
            scaled_loss.backward()
            # Store dummy metrics
            current_metrics = {
                'loss/policy': [0.0], 'loss/value': [0.0], 'loss/entropy': [0.0], 'loss/total': [0.0],
                'params/policy_clip_frac': [0.0], 'params/value_clip_frac': [0.0], 'params/approx_kl': [0.0]
            }
            for key, val_list in current_metrics.items():
                aggregate_metrics.setdefault(key, []).extend(val_list)
        except Exception as e_dummy:
            print(f"Dummy forward/backward failed in update loop: {e_dummy}")


        # --- Optimizer Step (Handled outside the exercise block) ---
        if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
            grads_exist = any(p.grad is not None for p in actor_model.parameters() if p.requires_grad)
            if grads_exist:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_model.parameters(), max_norm=cfg.ppo.max_grad_norm)
                aggregate_metrics.setdefault('params/grad_norm', []).append(grad_norm.item())
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # --- End of Epoch ---
    final_metrics = {key: np.mean(val) for key, val in aggregate_metrics.items() if val}
    return final_metrics
# --- EXERCISE 5 END ---


def perform_ppo_updates(
    actor_model: ActorModelWithValueHead,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: Dict[str, Any], # Can contain lists or tensors
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Performs multiple PPO epochs on the collected rollout data."""
    # Move collated tensors from buffer to the training device
    try:
        buffer_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in rollout_buffer.items()
        }
    except AttributeError as e:
         print(f"Error moving buffer to device, likely due to non-tensor data: {e}")
         print("Rollout Buffer Contents (Keys and Types):")
         for k, v in rollout_buffer.items(): print(f"  {k}: {type(v)}")
         return {} # Cannot proceed

    # Basic validation after moving to device
    if "response_input_ids" not in buffer_on_device or \
       not isinstance(buffer_on_device["response_input_ids"], torch.Tensor) or \
       buffer_on_device["response_input_ids"].numel() == 0:
        print("Warning: No response tokens found in buffer on device. Skipping PPO update.")
        return {}

    all_epoch_metrics = {}
    for ppo_epoch in range(cfg.ppo.epochs):
        # --- Calls EXERCISE 5 logic ---
        epoch_metrics = run_ppo_update_epoch(
            actor_model, optimizer, buffer_on_device, cfg, device
        )
        # Store metrics from the last epoch
        all_epoch_metrics = epoch_metrics

    return all_epoch_metrics


# ==============================================================================
# == 6. Training Setup and Orchestration - PROVIDED
# ==============================================================================

def setup_training(cfg: DictConfig) -> Tuple[torch.device, str]:
    """Sets random seeds, device, and output directory."""
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.training.seed)
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        if cfg.training.device == "cuda": print("Warning: CUDA requested but unavailable, using CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    return device, output_dir

def load_models_and_tokenizer(cfg: DictConfig, device: torch.device) -> Tuple[
    ActorModelWithValueHead, PreTrainedModel, PreTrainedTokenizerBase
]:
    """Loads tokenizer, actor model (with value head), and reference model."""
    print(f"Loading tokenizer: {cfg.model.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    # --- Set Padding Token ---
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Explicitly set padding side (optional, default is often right)
    # tokenizer.padding_side = 'left' # Uncomment to use left padding

    print(f"Loading models: {cfg.model.name}")
    # --- Model Kwargs (dtype, quantization, etc.) ---
    model_kwargs = {}
    model_dtype_str = cfg.model.get("torch_dtype", "auto")
    if model_dtype_str != "auto":
        try: model_kwargs["torch_dtype"] = getattr(torch, model_dtype_str)
        except AttributeError: print(f"Warning: Invalid torch_dtype '{model_dtype_str}'. Using auto.")
    if cfg.model.get("trust_remote_code", False): model_kwargs["trust_remote_code"] = True
    if cfg.model.get("quantization"):
        q_cfg = cfg.model.quantization
        print(f"Applying quantization: {q_cfg}")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
             load_in_8bit=q_cfg.get("load_in_8bit", False), load_in_4bit=q_cfg.get("load_in_4bit", False),
             bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
             bnb_4bit_compute_dtype=getattr(torch, q_cfg.get("bnb_4bit_compute_dtype", "float16")),
             bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", False),
        )

    # --- Load Actor Model ---
    actor_model = ActorModelWithValueHead(cfg.model.name, **model_kwargs)
    if not cfg.model.get("quantization"): actor_model.to(device) # Move if not quantized
    # Ensure pad token ID is set in model config
    if actor_model.config.pad_token_id is None:
        actor_model.config.pad_token_id = tokenizer.pad_token_id
    print("Actor model loaded.")

    # --- Load Reference Model ---
    ref_model_kwargs = model_kwargs.copy()
    ref_model_kwargs.pop("quantization_config", None) # No quantization for ref model
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **ref_model_kwargs)
    ref_model.to(device) # Move ref model to device
    if ref_model.config.pad_token_id is None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
    # Freeze reference model
    for param in ref_model.parameters(): param.requires_grad = False
    ref_model.eval()
    print("Reference model loaded and frozen.")

    return actor_model, ref_model, tokenizer


def load_and_preprocess_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Loads the dataset and preprocesses it."""
    print(f"Loading dataset: {cfg.dataset.name}")
    try:
        dataset = load_dataset(cfg.dataset.name, cfg.dataset.get("config"), split=cfg.dataset.split)
    except Exception as e:
         print(f"Error loading dataset '{cfg.dataset.name}': {e}")
         raise # Re-raise critical error

    # --- Subsetting ---
    num_samples = cfg.training.get("num_samples")
    if num_samples is not None and num_samples > 0 and num_samples <= len(dataset):
        print(f"Subsetting dataset to {num_samples} samples.")
        dataset = dataset.select(range(num_samples))

    # --- Preprocessing Function ---
    def preprocess_function(example):
        try:
            example["prompt"] = cfg.dataset.prompt_format.format(question=example["question"])
            example["ground_truth_answer"] = example["answer"].split("####")[-1].strip()
        except KeyError as e:
            print(f"Error processing example: Missing key {e}. Skipping prompt/answer.")
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
            remove_columns=dataset.column_names # Keep only processed columns
        )
        processed_dataset.set_format(type="torch") # Set format for DataLoader
        print(f"Dataset preprocessed. Samples: {len(processed_dataset)}")
        return processed_dataset
    except Exception as e:
         print(f"Error during dataset mapping: {e}")
         raise # Re-raise critical error


def setup_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Sets up the optimizer based on configuration."""
    use_8bit = cfg.ppo.get("use_8bit_adam", False)
    lr = cfg.ppo.learning_rate

    if use_8bit and bnb_available and isinstance(next(model.parameters()).device, torch.device) and next(model.parameters()).device.type == "cuda":
        is_quantized = hasattr(model, 'quantization_config') and \
                       (model.quantization_config.load_in_8bit or model.quantization_config.load_in_4bit)
        if is_quantized:
             print("Warning: Using 8-bit AdamW with a quantized model. Consider standard AdamW.")
             optimizer = AdamW(model.parameters(), lr=lr)
        else:
             print("Using 8-bit AdamW Optimizer (bitsandbytes)")
             optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        if use_8bit: print("Info: 8-bit Adam not used (requirements not met). Using standard AdamW.")
        else: print("Using standard AdamW Optimizer")
        optimizer = AdamW(model.parameters(), lr=lr)
    return optimizer


def create_generation_config(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase) -> GenerationConfig:
     """Creates the GenerationConfig object."""
     return GenerationConfig(
        max_new_tokens=cfg.generation.max_new_tokens,
        min_new_tokens=cfg.generation.min_new_tokens,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        do_sample=cfg.generation.do_sample,
        pad_token_id=tokenizer.pad_token_id
    )

def save_model(model: nn.Module, tokenizer: PreTrainedTokenizerBase, save_path: str):
    """Saves the model and tokenizer."""
    print(f"Saving model checkpoint to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    try:
        unwrapped_model = getattr(model, "base_model", model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved.")
    except Exception as e:
        print(f"Error saving model: {e}")


# ==============================================================================
# == 7. Main Training Orchestration - PROVIDED
# ==============================================================================

def train(cfg: DictConfig):
    """Main PPO training loop."""
    # --- 1. Initial Setup ---
    device, output_dir = setup_training(cfg)
    try: OmegaConf.save(cfg, os.path.join(output_dir, "effective_config.yaml"))
    except Exception as e: print(f"Error saving final config: {e}")

    # --- 2. Load Models and Tokenizer ---
    try: actor_model, ref_model, tokenizer = load_models_and_tokenizer(cfg, device)
    except Exception as e: print(f"Failed to load models/tokenizer: {e}"); return

    # --- 3. Load and Preprocess Dataset ---
    try: processed_dataset = load_and_preprocess_dataset(cfg, tokenizer)
    except Exception as e: print(f"Failed to load/preprocess dataset: {e}"); return

    # --- 4. Setup Optimizer ---
    optimizer = setup_optimizer(cfg, actor_model)

    # --- 5. Generation Config ---
    gen_config = create_generation_config(cfg, tokenizer)

    # --- 6. Collate Function for DataLoader ---
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        try:
            padded_inputs = tokenizer.pad({"input_ids": input_ids},
                                          padding='longest', return_tensors="pt", return_attention_mask=True)
        except Exception as e: print(f"Error during tokenizer.pad: {e}"); return None
        ground_truths = [item['ground_truth_answer'] for item in batch]
        return {"prompt_input_ids": padded_inputs["input_ids"],
                "prompt_attention_mask": padded_inputs["attention_mask"],
                "ground_truth_answers": ground_truths}

    # --- 7. Main PPO Loop ---
    print("\n--- Starting PPO Training ---")
    for ppo_step in range(cfg.training.total_ppo_steps):
        print(f"\n===== PPO Step {ppo_step + 1}/{cfg.training.total_ppo_steps} =====")

        # --- Phase 1: Rollout ---
        print("Phase 1: Generating Rollouts...")
        prompt_dataloader = DataLoader(
            processed_dataset, batch_size=cfg.ppo.batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        try:
            rollout_buffer = perform_rollouts(
                actor_model, ref_model, tokenizer, prompt_dataloader, gen_config, device
            )
        except Exception as e:
            print(f"Error during rollout phase: {e}"); import traceback; traceback.print_exc(); continue

        # Validate rollout buffer
        if not rollout_buffer or "rewards" not in rollout_buffer or \
           not isinstance(rollout_buffer["rewards"], torch.Tensor) or \
           rollout_buffer["rewards"].numel() == 0:
            print("Warning: Invalid rollout buffer generated. Skipping update."); continue

        avg_reward = rollout_buffer["rewards"].mean().item()
        num_rollouts = rollout_buffer["rewards"].shape[0]
        print(f"Rollout complete ({num_rollouts} samples). Average reward: {avg_reward:.4f}")

        # --- Phase 2: Update ---
        print("Phase 2: Performing PPO Updates...")
        try:
            # --- Calls EXERCISE 5 logic indirectly via run_ppo_update_epoch ---
            metrics = perform_ppo_updates(
                actor_model, optimizer, rollout_buffer, cfg, device
            )
        except Exception as e:
             print(f"Error during update phase: {e}"); import traceback; traceback.print_exc(); metrics = {}

        # Log metrics
        if metrics:
            log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Update Metrics (Avg over Epoch): {log_str}")
            print(f"  Rollout Reward (for this step): {avg_reward:.4f}")
        else: print("PPO update skipped or failed.")

        # --- Phase 3: Save Checkpoint ---
        if (ppo_step + 1) % cfg.training.save_interval == 0:
            save_model(actor_model, tokenizer, os.path.join(output_dir, f"step_{ppo_step + 1}"))

    # --- 8. Final Save ---
    print("\n--- PPO Training Finished ---")
    save_model(actor_model, tokenizer, os.path.join(output_dir, "final"))


# ==============================================================================
# == 8. Command-Line Interface Logic - PROVIDED
# ==============================================================================

def load_config_with_cli_overrides(
    default_config_path: str = "configs",
    default_config_name: str = "config.yaml"
) -> DictConfig:
    """Loads OmegaConf config, handling defaults and CLI overrides."""
    parser = argparse.ArgumentParser(description="PPO RL Trainer (Refactored Exercise)")
    parser.add_argument("--config-path", type=str, default=default_config_path)
    parser.add_argument("--config-name", type=str, default=default_config_name)
    parser.add_argument("overrides", nargs="*", help="Key=value config overrides")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes script is in src/
    config_dir_abs = os.path.join(project_root, args.config_path)

    conf_path = os.path.join(config_dir_abs, args.config_name)
    if not os.path.exists(conf_path): # Fallback to relative path
        conf_path = os.path.join(args.config_path, args.config_name)

    if not os.path.exists(conf_path):
        print(f"Error: Config file '{args.config_name}' not found in '{config_dir_abs}' or '{args.config_path}'.")
        sys.exit(1)

    print(f"Loading config from: {conf_path}")
    cfg = OmegaConf.load(conf_path)

    # Handle 'defaults' for base config merging (simplified)
    if 'defaults' in cfg and cfg.defaults:
        base_conf_name = cfg.defaults[0] + ".yaml" # Assumes first default is base
        base_conf_path = os.path.join(config_dir_abs, base_conf_name)
        if not os.path.exists(base_conf_path): base_conf_path = os.path.join(args.config_path, base_conf_name) # Fallback

        if os.path.exists(base_conf_path):
            print(f"Loading base config from: {base_conf_path}")
            base_cfg = OmegaConf.load(base_conf_path)
            cfg = OmegaConf.merge(base_cfg, cfg) # Merge base first
        else: print(f"Warning: Base config '{base_conf_name}' not found.")
        OmegaConf.set_struct(cfg, False); cfg.pop('defaults', None); OmegaConf.set_struct(cfg, True)

    # Apply command-line overrides
    if args.overrides:
        print(f"Applying overrides: {args.overrides}")
        cli_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    # Resolve interpolations
    try: OmegaConf.resolve(cfg)
    except Exception as e: print(f"Warning: Config resolution error: {e}")

    print("--------- Final Configuration ---------")
    print(OmegaConf.to_yaml(cfg, resolve=True)) # Print resolved config
    print("---------------------------------------")
    return cfg


# ==============================================================================
# == 9. Entry Point - PROVIDED
# ==============================================================================

if __name__ == "__main__":
    config = load_config_with_cli_overrides()
    train(config)
