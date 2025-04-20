# src/ppo_trainer.py
# -*- coding: utf-8 -*-
import torch
import wandb
import torch.nn.functional as F
from torch.optim import AdamW
# Try importing 8-bit AdamW from bitsandbytes
try:
    import bitsandbytes.optim as bnb_optim
    bnb_available = True
except ImportError:
    print(
        "Warning: bitsandbytes not found. 8-bit Adam optimizer will not be available."
    )
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


# --- Core PPO Logic (Exercise Placeholders - REORDERED) ---
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
# 5. `perform_ppo_update` (The Update Loop Logic):
#    - Why last? This orchestrates the entire learning step, bringing together the rollout
#      data, GAE calculation, and all the loss functions you've implemented.
#    - Focus: Structure the PPO epoch and mini-batch loops, correctly call GAE and loss
#      functions, combine losses, and perform the backward pass/optimizer step.
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
# B. Advantage Calculation Phase (within `perform_ppo_update`):
#    - The collected rollout data (`rewards`, `values`, `logprobs`, `ref_logprobs`) is processed.
#    - First, KL penalties are calculated (`logprobs - ref_logprobs`).
#    - Then, `compute_gae_advantages` uses the rewards, KL penalties, and values to estimate:
#      - `advantages`: How much better were actions than expected? (Incorporates KL penalty).
#      - `returns`: What was the actual observed discounted reward-to-go? (Target for Critic).
#
# C. Update Phase (`perform_ppo_update`):
#    - Uses rollout data and calculated advantages/returns to update the model.
#    - Loops for multiple `ppo_epochs` over the *same* rollout data.
#    - Iterates over mini-batches:
#      - Re-evaluates sequences with the *current* Actor/Critic (`logprobs_new`, `values_new`).
#      - Calculates the three core losses (`compute_policy_loss`, `compute_value_loss`, `compute_entropy_loss`).
#      - Combines losses into a single objective.
#      - Performs gradient descent (`backward()`, `optimizer.step()`) to update Actor/Critic.
#
# D. Repeat: The entire cycle (Rollout -> GAE -> Update) repeats.
# =========================================================

# --- EXERCISE START: Implement Core RL Functions (Reordered) ---


# Exercise 1: Policy Loss
def compute_policy_loss(log_probs_new, log_probs_old, advantages,
                        response_mask, clip_ratio):
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
    policy_loss = torch.tensor(0.0, device=log_probs_new.device)  # Placeholder
    clip_frac = torch.tensor(0.0, device=log_probs_new.device)  # Placeholder
    approx_kl = torch.tensor(0.0, device=log_probs_new.device)  # Placeholder

    # <<<< YOUR POLICY LOSS IMPLEMENTATION HERE >>>>
    # Ensure advantages and logprobs shapes align before calculating ratio/surrogates.
    if advantages.shape == log_probs_old.shape:
        print("Warning: Policy loss calculation not implemented!"
              )  # Remove this line
    else:
        print(
            f"Warning: Shape mismatch for policy loss calculation. Advantages: {advantages.shape}, LogProbs Old: {log_probs_old.shape}"
        )
        # Fallback to avoid crash during testing with dummy GAE:
        try:  # Try slicing first
            advantages_aligned = advantages[:, :log_probs_old.shape[1]]
            response_mask_aligned = response_mask[:, :log_probs_old.shape[1]]
            if advantages_aligned.shape == log_probs_old.shape:
                print(
                    "Warning: Policy loss calculation not implemented! (using fallback shapes)"
                )
            else:
                pass  # Let it potentially fail later or return dummy 0 loss
        except IndexError:  # Handle cases where slicing might fail (e.g. empty dims)
            print(
                "Warning: Slicing failed during policy loss shape alignment.")
            pass  # Let it potentially fail later or return dummy 0 loss

    return policy_loss, clip_frac, approx_kl


# Exercise 2: Value Loss
def compute_value_loss(values_new, values_old, returns, response_mask,
                       clip_range_value):
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
    value_loss = torch.tensor(0.0, device=values_new.device)  # Placeholder
    vf_clip_frac = torch.tensor(0.0, device=values_new.device)  # Placeholder

    # <<<< YOUR VALUE LOSS IMPLEMENTATION HERE >>>>
    print(
        "Warning: Value loss calculation not implemented!")  # Remove this line

    return value_loss, vf_clip_frac


# Exercise 3: Entropy Loss
def compute_entropy_loss(logits_new, response_mask):
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
    2. Compute the entropy of the distribution.
    3. Calculate the masked mean of the entropy using `response_mask`.
    4. The entropy loss is the negative of the mean entropy (since we want to maximize entropy).

    Returns:
        entropy_loss (torch.Tensor): Scalar entropy loss.
    """
    entropy_loss = torch.tensor(0.0, device=logits_new.device)  # Placeholder

    # <<<< YOUR ENTROPY LOSS IMPLEMENTATION HERE >>>>
    # Ensure mask shape aligns with entropy shape, which should be (batch, resp_len)
    print("Warning: Entropy loss calculation not implemented!"
          )  # Remove this line

    return entropy_loss


# Exercise 4: GAE Advantages
def compute_gae_advantages(final_rewards, kl_penalties, values, response_mask,
                           gamma, lam):
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
    2. Construct `token_level_rewards`: Apply `final_rewards` at the last valid token position
       (use `response_mask` to find it) and subtract `kl_penalties` at each step.
    3. Iterate backwards through the sequence length (`response_length`).
    4. Calculate `delta = token_level_rewards_t + gamma * V(s_{t+1}) - V(s_t)`.
       - Handle boundary condition: `V(s_T) = 0` if `t` is the last token.
       - Use `response_mask` to ensure values/rewards are zero after sequence end.
    5. Calculate `advantage_t = delta_t + gamma * lambda * advantage_{t+1}`.
       - Use `response_mask` to reset advantage after sequence end.
    6. Store calculated advantages.
    7. Compute `returns = advantages + values`.
    8. Whiten (normalize) the `advantages` using the `masked_whiten` helper function.

    Returns:
        advantages (torch.Tensor): Shape (batch_size, response_length) - Whitened advantages.
        returns (torch.Tensor): Shape (batch_size, response_length) - Calculated returns.
    """
    advantages = torch.zeros_like(values)  # Placeholder
    returns = torch.zeros_like(values)  # Placeholder

    # <<<< YOUR GAE IMPLEMENTATION HERE >>>>
    # Remember to use torch.no_grad() context

    print("Warning: GAE calculation not implemented!"
          )  # Remove this line after implementation
    # Dummy implementation for shape consistency:
    advantages = masked_whiten(torch.randn_like(values), response_mask)
    returns = advantages + values

    return advantages, returns


# --- EXERCISE END ---


# --- Rollout Phase ---
def perform_rollouts(actor_model, ref_model, tokenizer, prompt_dataloader,
                     gen_config, device):
    """
    Generates responses and computes necessary data for PPO update.
    (Full implementation included below)
    """
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
    actor_model.eval()  # Set models to evaluation mode for generation
    ref_model.eval()

    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"]

        with torch.no_grad():  # Disable gradient calculations during rollout
            # Step 1: Generate responses using the current actor policy.
            generated_output = actor_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=
                True,  # Setting to True might not be strictly needed if we do separate forward pass
            )
            response_ids = generated_output.sequences[:, prompt_ids.shape[1]:]

            # Step 2: Prepare full sequences (prompt + response) for analysis.
            full_ids = torch.cat((prompt_ids, response_ids), dim=1)
            # Create attention mask for the response part, ensuring it's on the correct device
            # Note: The mask calculation here assumes PAD token is used for eos/padding in response.
            # If generation stops *before* max_length without EOS/PAD, this mask might extend beyond actual generated tokens.
            # A more robust way might involve tracking actual lengths, but this is common.
            response_mask = (response_ids
                             != tokenizer.pad_token_id).long().to(device)
            full_mask = torch.cat((prompt_mask, response_mask), dim=1)

            # Step 3: Perform forward passes to get needed values for PPO.
            outputs = actor_model(full_ids, attention_mask=full_mask)
            logits = outputs.logits
            values = outputs.value.squeeze(-1)  # Shape: (batch, full_seq_len)

            ref_outputs = ref_model(full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits

            # Step 4: Calculate log probabilities of the *generated* tokens.
            prompt_len = prompt_ids.shape[1]
            response_len = response_ids.shape[1]

            if response_len > 0:
                # Logits for predicting token t are at index t-1
                # Need logits for tokens 0...response_len-1 (indices prompt_len-1 to prompt_len+response_len-2)
                logits_for_logprobs = logits[:, prompt_len - 1:prompt_len +
                                             response_len - 1, :]
                ref_logits_for_logprobs = ref_logits[:, prompt_len -
                                                     1:prompt_len +
                                                     response_len - 1, :]
                # Target tokens are response_ids (token 0 to response_len-1)
                target_ids = response_ids  # Shape: (batch, response_len)

                logprobs_all_vocab = F.log_softmax(logits_for_logprobs, dim=-1)
                ref_logprobs_all_vocab = F.log_softmax(ref_logits_for_logprobs,
                                                       dim=-1)

                logprobs = torch.gather(logprobs_all_vocab, 2,
                                        target_ids.unsqueeze(-1)).squeeze(-1)
                ref_logprobs = torch.gather(
                    ref_logprobs_all_vocab, 2,
                    target_ids.unsqueeze(-1)).squeeze(-1)

                # Values corresponding to states *before* generating the token
                # Indices: (prompt_len - 1) to (prompt_len + response_len - 1)
                values_response = values[:, prompt_len - 1:prompt_len +
                                         response_len - 1]

                # Apply mask to logprobs and values where response_mask is 0
                logprobs = logprobs * response_mask
                ref_logprobs = ref_logprobs * response_mask
                values_response = values_response * response_mask

            else:  # Handle sequences with no response tokens generated
                logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                ref_logprobs = torch.zeros((prompt_ids.shape[0], 0),
                                           device=device)
                values_response = torch.zeros((prompt_ids.shape[0], 0),
                                              device=device)

            # Step 5: Calculate the final reward for the generated sequence based on the task.
            full_decoded_texts = tokenizer.batch_decode(
                full_ids, skip_special_tokens=True)
            rewards = torch.tensor([
                compute_gsm8k_reward(txt, gt)
                for txt, gt in zip(full_decoded_texts, ground_truths)
            ],
                                   dtype=torch.float32,
                                   device=device)  # Shape: (batch,)

            # Step 6: Store all computed data in the rollout buffer (on CPU).
            rollout_buffer["prompt_input_ids"].append(prompt_ids.cpu())
            rollout_buffer["prompt_attention_mask"].append(prompt_mask.cpu())
            rollout_buffer["response_input_ids"].append(response_ids.cpu())
            rollout_buffer["response_attention_mask"].append(
                response_mask.cpu())  # Mask for response tokens
            rollout_buffer["logprobs"].append(
                logprobs.cpu())  # Shape: (batch, response_len)
            rollout_buffer["ref_logprobs"].append(
                ref_logprobs.cpu())  # Shape: (batch, response_len)
            rollout_buffer["values"].append(
                values_response.cpu())  # Shape: (batch, response_len)
            rollout_buffer["rewards"].append(rewards.cpu())  # Shape: (batch,)
            rollout_buffer["full_texts"].extend(full_decoded_texts)
            rollout_buffer["ground_truth_answers"].extend(ground_truths)

    # Collate the buffer after collecting data from all batches.
    collated_buffer = {}
    padding_value_map = {
        "input_ids":
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "attention_mask":
        0,
        # Add other keys if they need specific padding values (e.g., logprobs, values often padded with 0)
    }
    for key, data_list in rollout_buffer.items():
        if key in ["full_texts", "ground_truth_answers"]:
            collated_buffer[key] = data_list  # Keep as list
        elif key == "rewards":  # Rewards are per-sequence
            if data_list: collated_buffer[key] = torch.cat(data_list, dim=0)
            else: collated_buffer[key] = torch.empty(0)
        else:  # Handle tensors, padding needed for sequences
            if data_list:
                # Use pad_sequence for robust padding
                if data_list[0].dim(
                ) > 0:  # Only pad tensors with sequence dim
                    # Determine padding value based on key content
                    pad_val_key = key.split(
                        "_"
                    )[-1]  # Heuristic: use last part like 'ids', 'mask', 'logprobs'
                    pad_val = padding_value_map.get(pad_val_key,
                                                    0.0)  # Default pad 0.0

                    # Ensure all tensors in the list are tensors before padding
                    if all(isinstance(t, torch.Tensor) for t in data_list):
                        try:
                            collated_buffer[
                                key] = torch.nn.utils.rnn.pad_sequence(
                                    data_list,
                                    batch_first=True,
                                    padding_value=pad_val)
                        except Exception as e_pad:
                            print(
                                f"Error during padding for key '{key}': {e_pad}"
                            )
                            print(
                                f"Attempting simple concat (may fail if lengths differ)."
                            )
                            try:
                                collated_buffer[key] = torch.cat(data_list,
                                                                 dim=0)
                            except Exception as e_cat:
                                print(
                                    f"Concatenation also failed for key '{key}': {e_cat}"
                                )
                                collated_buffer[key] = []  # Mark as failed

                    else:
                        print(
                            f"Warning: Non-tensor data found in list for key '{key}'. Skipping collation."
                        )
                        collated_buffer[key] = []  # Or handle appropriately
                else:  # Scalars or 1D tensors per item
                    collated_buffer[key] = torch.cat(data_list, dim=0)
            else:
                collated_buffer[key] = torch.empty(0)  # Handle empty list case
    return collated_buffer


# --- Update Phase (Exercise Placeholder) ---
def perform_ppo_update(actor_model, optimizer, rollout_buffer, cfg: DictConfig,
                       device):
    """
    EXERCISE: Implement the PPO optimization loop.
    WHY Update Phase?: After collecting experience (rollouts), we use that data to improve
                 the actor (policy) and critic (value function) models.
    WHY Multiple Epochs?: PPO is an on-policy algorithm, meaning updates should ideally use
                 data generated by the *current* policy. However, generating rollouts is expensive.
                 PPO allows for multiple update epochs over the *same* rollout data by using
                 importance sampling (via the logprob ratio) and clipping to keep updates stable,
                 improving sample efficiency compared to vanilla policy gradients.
    WHY Mini-batches?: Processing the entire rollout buffer at once might require too much memory.
                 Using mini-batches allows for updates with manageable memory usage, similar to
                 standard supervised learning.
    """
    actor_model.train()  # Set model to training mode
    metrics = {}
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

        # Basic validity check
        if response_ids.numel() == 0 or response_ids.shape[1] == 0:
            print("Warning: Skipping update. Empty responses in buffer.")
            return {}
        # Add more shape consistency checks if needed after collation fixes
        # Example: Check if shapes match expected response length derived from mask
        if logprobs_old.shape[1] != response_ids.shape[1] or values_old.shape[
                1] != response_ids.shape[1]:
            print(
                f"Warning: Shape mismatch after collation. Logprobs: {logprobs_old.shape}, Values: {values_old.shape}, Response: {response_ids.shape}. Skipping update."
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
        # Calculate KL divergence penalty per token
        kl_per_token = logprobs_old - ref_logprobs  # Shape: (batch, resp_len)
        # Scale by coefficient to get KL penalty term used in reward signal
        kl_penalties = cfg.ppo.kl_coeff * kl_per_token

        # Compute GAE using the final sequence reward, KL penalties, and critic values
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

            # <<<< YOUR PPO MINI-BATCH UPDATE IMPLEMENTATION HERE >>>>
            # Steps:
            # 1. Forward Pass: Get new logits and values from actor_model
            #    outputs = actor_model(batch_full_ids, attention_mask=batch_full_mask)
            #    logits_new = outputs.logits
            #    values_new = outputs.value.squeeze(-1)
            #
            # 2. Calculate New Logprobs: Extract logits for response part and calculate logprobs
            #    for the actual response tokens (`batch_response_tokens`). Careful with slicing!
            #    logprobs_new = gather(log_softmax(logits_new[:, prompt_len-1:prompt_len+resp_len-1]), labels=batch_response_tokens)
            #    Shape: (mini_batch, resp_len)
            #
            # 3. Extract New Values: Slice `values_new` to match the response length.
            #    values_new_response = values_new[:, prompt_len-1 : prompt_len+resp_len-1]
            #    Shape: (mini_batch, resp_len)
            #
            # 4. Calculate Losses: Call your implemented loss functions.
            #    - policy_loss, p_clip_frac, approx_kl = compute_policy_loss(logprobs_new, batch_logprobs_old, batch_advantages, batch_response_mask, cfg.ppo.clip_ratio)
            #    - value_loss, v_clip_frac = compute_value_loss(values_new_response, batch_values_old, batch_returns, batch_response_mask, cfg.ppo.clip_range_value)
            #    - entropy_loss = compute_entropy_loss(logits_new[:, prompt_len-1:prompt_len+resp_len-1], batch_response_mask)
            #    *CRITICAL*: Ensure masks align with the tensor dimensions used in each loss (e.g., policy/entropy often use resp_len, value uses resp_len).
            #
            # 5. Combine Losses:
            #    loss = policy_loss + cfg.ppo.vf_coeff * value_loss + cfg.ppo.entropy_coeff * entropy_loss
            #
            # 6. Backward Pass:
            #    scaled_loss = loss / cfg.ppo.gradient_accumulation_steps
            #    scaled_loss.backward()
            #
            # 7. Store Metrics: Append .item() of losses/stats to a dictionary for logging.
            #    metrics['loss/policy'] = policy_loss.item() ... etc.
            #
            # 8. Optimizer Step:
            #    if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
            #        # Optional gradient clipping
            #        # torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
            #        optimizer.step()
            #        optimizer.zero_grad()
            #

            print(f"Warning: PPO Update step {ppo_step_count} not implemented!"
                  )  # Remove this line
            # Dummy backward pass to prevent errors when exercises aren't filled
            try:  # Add try-except for dummy backward pass
                dummy_loss = actor_model(
                    batch_full_ids,
                    attention_mask=batch_full_mask).logits.mean() * 0.0
                (dummy_loss / cfg.ppo.gradient_accumulation_steps).backward()
            except Exception as e_dummy:
                print(f"Dummy backward failed: {e_dummy}")
            # Store dummy metrics
            metrics = {
                'loss/policy': [0.0],
                'loss/value': [0.0],
                'loss/entropy': [0.0],
                'loss/total': [0.0],
                'params/policy_clip_frac': [0.0],
                'params/value_clip_frac': [0.0],
                'params/approx_kl': [0.0]
            }

            # Aggregate metrics immediately for simplicity here
            for key, val_list in metrics.items():
                aggregate_metrics.setdefault(key, []).extend(val_list)

            # Optimizer step after accumulating gradients
            if ppo_step_count % cfg.ppo.gradient_accumulation_steps == 0:
                if any(p.grad is not None for p in
                       actor_model.parameters()):  # Only step if grads exist
                    # Optional: Gradient Clipping (common for LLM training)
                    # torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

    # Aggregate metrics over the PPO epoch
    final_metrics = {
        key: np.mean(val)
        for key, val in aggregate_metrics.items() if val
    }  # Avoid division by zero if no steps ran
    return final_metrics


# --- Main Training Function ---
def train(cfg: DictConfig):
    """Main training loop, now takes OmegaConf DictConfig object"""
    # --- Setup ---
    if cfg.wandb.report_to_wandb:
        wandb.init(project=cfg.wandb.project,
                   config=OmegaConf.to_container(cfg, resolve=True),
                   name=cfg.wandb.name)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    # Determine device based on config and availability
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

    output_dir = cfg.training.output_dir  # Use output_dir from config
    os.makedirs(output_dir, exist_ok=True)
    # Save the final merged config
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
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Needed for model config

    print(f"Loading model: {cfg.model.name}")
    model_kwargs = {}
    # Set up dtype based on config
    model_dtype_str = cfg.model.get("torch_dtype", "auto")
    model_dtype = getattr(
        torch, model_dtype_str) if model_dtype_str != "auto" else "auto"
    if model_dtype != "auto": print(f"Setting model dtype to: {model_dtype}")
    model_kwargs["torch_dtype"] = model_dtype

    # Add trust_remote_code if specified in config
    if cfg.model.get("trust_remote_code", False):
        model_kwargs["trust_remote_code"] = True

    try:
        # Load models
        actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model.name, **model_kwargs)
        actor_model.to(device)  # Move model after loading
        # Update model config pad token id AFTER loading tokenizer
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
        print(
            f"Check model name, dtype ({model_dtype_str}), and necessary dependencies."
        )
        return  # Exit if models can't load

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
    use_8bit = cfg.ppo.get("use_8bit_adam",
                           False)  # Get flag, default to False
    if use_8bit and bnb_available and device.type == "cuda":
        print("Using 8-bit AdamW Optimizer (bitsandbytes)")
        optimizer = bnb_optim.AdamW8bit(actor_model.parameters(),
                                        lr=cfg.ppo.learning_rate)
    elif use_8bit and not bnb_available:
        print(
            "Warning: use_8bit_adam=True but bitsandbytes not available. Falling back to standard AdamW."
        )
        optimizer = AdamW(actor_model.parameters(), lr=cfg.ppo.learning_rate)
    elif use_8bit and device.type != "cuda":
        print(
            "Warning: use_8bit_adam=True ignored because device is not CUDA. Falling back to standard AdamW."
        )
        optimizer = AdamW(actor_model.parameters(), lr=cfg.ppo.learning_rate)
    else:
        print("Using standard AdamW Optimizer")
        optimizer = AdamW(actor_model.parameters(), lr=cfg.ppo.learning_rate)

    # --- Generation Config ---
    gen_config = GenerationConfig(
        max_new_tokens=cfg.generation.max_new_tokens,
        min_new_tokens=cfg.generation.min_new_tokens,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        do_sample=cfg.generation.do_sample,
        # Pass eos/pad during generate() call using tokenizer settings
    )

    # --- Main Training Loop ---
    print("--- Starting PPO Training (Exercise Mode) ---")
    for ppo_step in range(cfg.training.total_ppo_steps):
        print(f"\nPPO Step {ppo_step + 1}/{cfg.training.total_ppo_steps}")

        # --- Rollout Phase ---
        print("Phase 1: Generating Rollouts...")
        dataloader = torch.utils.data.DataLoader(tokenized_dataset,
                                                 batch_size=cfg.ppo.batch_size,
                                                 shuffle=True,
                                                 collate_fn=collate_fn)
        rollout_buffer = perform_rollouts(actor_model, ref_model, tokenizer,
                                          dataloader, gen_config, device)

        # Basic Rollout Stats & Logging
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

        # --- Update Phase ---
        print("Phase 2: Performing PPO Updates...")
        is_buffer_valid = (
            rollout_buffer and num_rollouts > 0
            and all(k in rollout_buffer
                    for k in ["response_input_ids", "logprobs", "values"])
            and rollout_buffer["response_input_ids"].numel() > 0
            and rollout_buffer["response_input_ids"].shape[1] >
            0  # Need at least 1 response token
        )
        if is_buffer_valid:
            metrics = perform_ppo_update(actor_model, optimizer,
                                         rollout_buffer, cfg, device)
            if cfg.wandb.report_to_wandb:
                wandb.login(metrics, ppo_step)
            if metrics and (ppo_step + 1) % cfg.training.log_interval == 0:
                print(f"PPO Step {ppo_step+1} Metrics (from update):")
                log_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(log_str)
                print(f"  Reward (mean from rollout): {avg_reward:.4f}")
            elif not metrics:
                print("Update function returned empty metrics.")
        else:
            print("Skipping update step because rollout buffer is invalid.")
        if cfg.wandb.report_to_wandb:
            # Prepare rollout metrics
            rollout_metrics_log = {"rollout/reward_mean": avg_reward}
            if num_rollouts > 0: # Avoid logging samples if buffer was invalid
                rollout_metrics_log["rollout/num_samples"] = num_rollouts

                # Combine all metrics for this step and log
                step_log_data = {**rollout_metrics_log, **update_metrics} # Combine dicts
                if step_log_data: # Only log if there's something to log
                    wandb.log(step_log_data, step=ppo_step)

        # --- Save Model Checkpoint ---
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

    # --- Final Model Saving ---
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

    # Construct absolute path to config directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        script_dir)  # Assumes src is one level down from root
    config_dir_abs = os.path.join(project_root, args.config_path)
    conf_path = os.path.join(config_dir_abs, args.config_name)

    if not os.path.exists(conf_path):
        # Try relative path from current working directory as fallback
        alt_conf_path = os.path.join(args.config_path, args.config_name)
        if os.path.exists(alt_conf_path):
            print(f"Found config at relative path: {alt_conf_path}")
            conf_path = alt_conf_path
        else:
            print(
                f"Error: Config file not found at {conf_path} or {alt_conf_path}"
            )
            sys.exit(1)

    print(f"Loading config from: {conf_path}")
    cfg = OmegaConf.load(conf_path)

    # Handle defaults inheritance (e.g., config_debug loading config first)
    if 'defaults' in cfg and isinstance(cfg.defaults, list) and cfg.defaults:
        base_conf_name = cfg.defaults[0] + ".yaml"
        base_conf_path_abs = os.path.join(config_dir_abs, base_conf_name)
        base_conf_path_rel = os.path.join(args.config_path, base_conf_name)
        base_path_to_load = None
        if os.path.exists(base_conf_path_abs):
            base_path_to_load = base_conf_path_abs
        elif os.path.exists(base_conf_path_rel):
            base_path_to_load = base_conf_path_rel

        if base_path_to_load:
            print(f"Loading base config from: {base_path_to_load}")
            base_cfg = OmegaConf.load(base_path_to_load)
            # Merge base first, then the specific config overrides the base
            cfg = OmegaConf.merge(base_cfg, cfg)
        else:
            print(
                f"Warning: Base config {base_conf_name} specified in defaults not found."
            )

        # Remove defaults key after potentially merging
        if 'defaults' in cfg:
            try:
                OmegaConf.set_struct(cfg,
                                     False)  # Allow popping keys temporarily
                cfg.pop('defaults')
                OmegaConf.set_struct(cfg,
                                     True)  # Restore struct mode if desired
            except Exception as e_pop:
                print(f"Note: Could not pop 'defaults' key: {e_pop}")

    # Apply command-line overrides AFTER merging defaults
    if args.overrides:
        print(f"Applying overrides: {args.overrides}")
        try:
            cli_conf = OmegaConf.from_cli(args.overrides)
            cfg = OmegaConf.merge(cfg, cli_conf)
        except Exception as e_cli:
            print(f"Error applying CLI overrides: {e_cli}")
            sys.exit(1)

    print("--------- Final Configuration ---------")
    # Resolve interpolations (like ${dataset.max_gen_length}) before printing/using
    try:
        OmegaConf.resolve(cfg)
        print(OmegaConf.to_yaml(cfg))
    except Exception as e_resolve:
        print(f"Error resolving OmegaConf interpolations: {e_resolve}")
        print("Using potentially unresolved config.")
        print(OmegaConf.to_yaml(cfg, resolve=False))  # Print unresolved

    print("---------------------------------------")

    # Start training
    train(cfg)


if __name__ == "__main__":
    main_cli()
