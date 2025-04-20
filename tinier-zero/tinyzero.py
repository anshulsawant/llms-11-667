# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.optim import AdamW # Use AdamW for LLMs
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from trl.models import AutoModelForCausalLMWithValueHead # Helper for value head
from datasets import load_dataset
import numpy as np
import random
import re
import math
from tqdm.auto import tqdm

# --- Configuration ---
config = {
    # Model & Tokenizer
    "model_name": "Qwen/Qwen1.5-1.8B-Chat", # ~1.8B parameter model
    "tokenizer_name": "Qwen/Qwen1.5-1.8B-Chat",
    "load_in_4bit": True,  # Use 4-bit quantization to fit on more GPUs

    # Dataset & Task
    "dataset_name": "gsm8k",
    "dataset_split": "train", # Use train split for prompts
    "prompt_format": "Question: {question}\nAnswer:", # How to format the prompt
    "max_prompt_length": 512, # Max tokens for prompt
    "max_gen_length": 256,  # Max tokens to generate for answer

    # PPO Hyperparameters (tune these based on model/task/results)
    "learning_rate": 1e-6, # Lower LR often needed for larger models & RL
    "ppo_epochs": 4,       # Optimization epochs per rollout
    "batch_size": 8,       # Rollout batch size (adjust based on VRAM)
    "mini_batch_size": 2,  # Update mini-batch size (adjust based on VRAM)
    "gradient_accumulation_steps": 4, # Effective batch size = batch_size * grad_acc_steps
    "kl_coeff": 0.1,       # KL penalty coefficient (beta)
    "clip_ratio": 0.2,     # PPO policy objective clipping
    "clip_range_value": 0.2,# PPO value function clipping
    "vf_coeff": 0.1,       # Value function loss weight
    "entropy_coeff": 0.01, # Entropy bonus weight
    "gamma": 1.0,          # Discount factor (1.0 often used for non-episodic LLM tasks)
    "lam": 0.95,           # GAE lambda

    # Training Control
    "total_ppo_steps": 100, # Number of PPO steps (Rollout -> Update)
    "seed": 42,
    "log_interval": 1,     # Log metrics every N PPO steps
    "save_interval": 10,   # Save model every N PPO steps
    "output_dir": "ppo_gsm8k_qwen1.8b",
}

# --- Setup ---
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model, Tokenizer ---

print(f"Loading model: {config['model_name']}")
quantization_config = None
if config["load_in_4bit"]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Or float16 if bf16 not supported
    )

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
# Qwen uses pad_token='<|endoftext|>'
if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
     print("Warning: Setting pad_token to eos_token.")
     tokenizer.pad_token = tokenizer.eos_token
     tokenizer.pad_token_id = tokenizer.eos_token_id


# Load base model for Actor and Reference
# Actor needs a value head. Reference doesn't but using the same class is convenient.
model_kwargs = {"quantization_config": quantization_config} if config["load_in_4bit"] else {}
actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"],
    trust_remote_code=True, # Needed for some models like Qwen
    **model_kwargs
).to(device)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"],
    trust_remote_code=True,
    **model_kwargs
).to(device)

# Ensure ref model parameters are not updated
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()
print("Models loaded.")

# --- 2. Load Dataset & Preprocess ---
print(f"Loading dataset: {config['dataset_name']}")
dataset = load_dataset(config["dataset_name"], config["dataset_split"])

def preprocess_dataset(example):
    """Formats and tokenizes prompts."""
    example["prompt"] = config["prompt_format"].format(question=example["question"])
    tokenized_prompt = tokenizer(
        example["prompt"],
        max_length=config["max_prompt_length"],
        truncation=True,
        padding=False, # Padding handled later by data collator or during batching
        return_tensors=None # Return list of IDs
    )
    example["input_ids"] = tokenized_prompt["input_ids"]
    example["attention_mask"] = tokenized_prompt["attention_mask"]
    # Keep ground truth answer for reward calculation
    example["ground_truth_answer"] = example["answer"].split("####")[-1].strip()
    return example

tokenized_dataset = dataset.map(preprocess_dataset, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch")
print(f"Dataset preprocessed. Number of prompts: {len(tokenized_dataset)}")
print(f"Example prompt: {tokenized_dataset[0]['prompt']}")
print(f"Example input_ids length: {len(tokenized_dataset[0]['input_ids'])}")
print(f"Example ground truth: {tokenized_dataset[0]['ground_truth_answer']}")

# Simple data collator for batching prompts
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    # Pad prompts to the longest in the batch
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding='longest',
        return_tensors="pt",
        return_attention_mask=True,
    )
    # Keep track of ground truth for reward
    ground_truths = [item['ground_truth_answer'] for item in batch]
    return {
        "prompt_input_ids": padded_inputs["input_ids"],
        "prompt_attention_mask": padded_inputs["attention_mask"],
        "ground_truth_answers": ground_truths
    }

# --- 3. GSM8K Reward Function (based on TinyZero/verl/utils/reward_score/gsm8k.py) ---
def extract_gsm8k_solution(solution_str):
    """Extracts the numerical answer from the #### format."""
    # Use strict method from TinyZero
    solution = re.search(r"####\s*([-+]?\s*[\d\.\,]+)", solution_str)
    if solution is None:
        # Try finding the last number if strict format fails (more flexible)
         answer = re.findall(r"([-+]?\s*[\d\.\,]+)", solution_str)
         if len(answer) > 0:
              final_answer_str = answer[-1].replace(',', '').replace(' ', '')
              try:
                  # Validate if it's a number
                  float(final_answer_str)
                  return final_answer_str
              except ValueError:
                   return None
         else:
             return None # No number found
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


# --- 4. Core PPO Logic (adapted from TinyZero/verl/trainer/ppo/core_algos.py) ---

# Helper for masked operations (similar to verl.utils.torch_functional)
def masked_mean(tensor, mask, dim=None):
    """Calculates mean of tensor elements specified by mask."""
    if mask is None:
        return torch.mean(tensor, dim=dim)
    # Ensure mask is boolean
    mask = mask.bool()
    # Expand mask dims if necessary to match tensor
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8) # Add epsilon for stability
    return mean

def masked_whiten(tensor, mask, shift_mean=True):
    """Whitens the tensor values specified by the mask."""
    # Ensure mask is boolean and expanded
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_variance = torch.where(mask, (tensor - mean)**2, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    variance = masked_mean(masked_tensor_variance, mask, dim=None)
    std = torch.sqrt(variance + 1e-8) # Add epsilon for numerical stability

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(mask, whitened, tensor) # Return original values where mask is False


def compute_gae_advantages(final_rewards, kl_penalties, values, response_mask, gamma, lam):
    """
    Computes GAE advantages based on TinyZero/verl/trainer/ppo/core_algos.py.
    KL penalty is incorporated into the rewards here.

    Args:
        final_rewards (torch.Tensor): Shape (batch_size,) - Reward for the whole sequence.
        kl_penalties (torch.Tensor): Shape (batch_size, response_length) - KL penalty per token.
        values (torch.Tensor): Shape (batch_size, response_length) - Critic values for response tokens.
        response_mask (torch.Tensor): Shape (batch_size, response_length) - Mask for response tokens.
        gamma (float): Discount factor.
        lam (float): GAE lambda.

    Returns:
        advantages (torch.Tensor): Shape (batch_size, response_length)
        returns (torch.Tensor): Shape (batch_size, response_length)
    """
    with torch.no_grad():
        response_length = values.shape[1]
        advantages_reversed = []
        last_gae_lam = 0

        # Construct token-level rewards: 0 everywhere except last token (gets final reward), minus KL penalty
        token_level_rewards = torch.zeros_like(values)
        # Find the index of the last *actual* response token for each sequence using the mask
        sequence_lengths = response_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1

        # Apply final reward at the last valid token position
        # Use scatter_ for efficient assignment based on indices
        # Ensure last_token_indices are within bounds [0, response_length-1]
        valid_indices = last_token_indices >= 0
        if valid_indices.any():
           batch_indices = torch.arange(values.shape[0], device=values.device)[valid_indices]
           indices_to_update = last_token_indices[valid_indices]
           rewards_to_apply = final_rewards[valid_indices]
           token_level_rewards[batch_indices, indices_to_update] = rewards_to_apply

        # Subtract KL penalty at each step
        token_level_rewards -= kl_penalties

        # GAE calculation loop (from core_algos.compute_gae_advantage_return)
        for t in reversed(range(response_length)):
            # Value of next state V(s_{t+1}) - if t is the last step, next value is 0
            next_values = values[:, t + 1] if t < response_length - 1 else torch.zeros_like(values[:, 0])
            # Mask ensures we only consider valid steps
            current_mask = response_mask[:, t].float()
            # Delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * next_values * current_mask - values[:, t]
            # Accumulate GAE: delta_t + gamma * lambda * A(s_{t+1})
            # Mask ensures GAE resets after sequence ends
            last_gae_lam = delta + gamma * lam * last_gae_lam * current_mask
            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values # Return = Advantage + Value

        # Whiten advantages (normalize) - crucial for stability
        # Use the adapted masked_whiten function
        advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


def compute_policy_loss(log_probs_new, log_probs_old, advantages, response_mask, clip_ratio):
    """Computes PPO policy loss (clipped surrogate objective)."""
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2), response_mask) # Negative because we want to maximize
    # Calculate clip fraction for logging (optional)
    clip_frac = masked_mean(torch.gt(torch.abs(ratio - 1.0), clip_ratio).float(), response_mask)
    # Calculate approximate KL for logging (optional)
    approx_kl = masked_mean(log_probs_old - log_probs_new, response_mask) # KL(old || new)
    return policy_loss, clip_frac, approx_kl


def compute_value_loss(values_new, values_old, returns, response_mask, clip_range_value):
    """Computes PPO value loss (clipped)."""
    # Clip predicted values based on old values
    values_pred_clipped = values_old + torch.clamp(
        values_new - values_old, -clip_range_value, clip_range_value
    )
    # Calculate MSE loss for both clipped and unclipped predictions
    vf_loss1 = (values_new - returns) ** 2
    vf_loss2 = (values_pred_clipped - returns) ** 2
    # Use the max of the two losses (more stable) and take masked mean
    vf_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), response_mask)
    # Calculate clip fraction for logging (optional)
    vf_clip_frac = masked_mean(torch.gt(vf_loss2, vf_loss1).float(), response_mask)
    return vf_loss, vf_clip_frac


def compute_entropy_loss(logits_new, response_mask):
    """Computes entropy loss to encourage exploration."""
    # Calculate entropy from logits
    # Use torch.distributions.Categorical for stable entropy calculation
    dist = torch.distributions.Categorical(logits=logits_new)
    entropy = dist.entropy()
    # Apply mask and compute mean
    entropy_loss = -masked_mean(entropy, response_mask) # Negative because we maximize entropy
    return entropy_loss

# --- 5. Rollout Phase ---
generation_config = GenerationConfig(
    max_new_tokens=config["max_gen_length"],
    min_new_tokens=5, # Ensure a minimum length response
    temperature=0.7,  # Use temperature sampling for exploration
    top_k=50,
    top_p=0.95,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

def perform_rollouts(actor_model, ref_model, tokenizer, prompt_dataloader, gen_config):
    """Generates responses and computes necessary data for PPO update."""
    rollout_buffer = {
        "prompt_input_ids": [], "prompt_attention_mask": [],
        "response_input_ids": [], "response_attention_mask": [],
        "logprobs": [], "ref_logprobs": [], "values": [],
        "rewards": [], "full_texts": [], "ground_truth_answers": []
    }
    actor_model.eval()
    ref_model.eval()

    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"]

        with torch.no_grad():
            # Generate sequences with the actor model
            generated_output = actor_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True, # Needed for logprobs calculation
            )
            # Extract generated part (response)
            # Sequences includes prompt + response
            response_ids = generated_output.sequences[:, prompt_ids.shape[1]:]

            # Combine prompt and response for forward passes
            full_ids = torch.cat((prompt_ids, response_ids), dim=1)
            # Create attention mask for the full sequence
            # Assume full attention if not padding (or handle padding correctly)
            response_mask = (response_ids != tokenizer.pad_token_id).long()
            full_mask = torch.cat((prompt_mask, response_mask), dim=1)

            # Forward pass with Actor to get logprobs and values
            outputs = actor_model(full_ids, attention_mask=full_mask)
            logits = outputs.logits
            values = outputs.value.squeeze(-1) # Shape: (batch, full_seq_len)

            # Forward pass with Reference model to get ref_logprobs
            ref_outputs = ref_model(full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits

            # Calculate logprobs for the *response* tokens only
            # Shift logits and labels for next token prediction logic
            # Logits shape: (batch, full_seq_len, vocab_size)
            # We need logprobs for response_ids[:, 1:] using logits[:, prompt_len-1:-1]
            prompt_len = prompt_ids.shape[1]
            response_len = response_ids.shape[1]

            # Ensure lengths match for gather
            # Logits needed: from index (prompt_len - 1) up to (prompt_len + response_len - 2)
            # Labels needed: response_ids[:, 1:] (from index 1 up to response_len - 1)
            logprobs_logits = F.log_softmax(logits[:, prompt_len - 1 : prompt_len + response_len - 1, :], dim=-1)
            ref_logprobs_logits = F.log_softmax(ref_logits[:, prompt_len - 1 : prompt_len + response_len - 1, :], dim=-1)

            # Target tokens are response_ids shifted left (ignore first token, use up to second-to-last)
            target_ids = response_ids[:, 1:] # Shape: (batch, response_len - 1)

            # Gather logprobs corresponding to actual generated tokens
            # Need to handle potential length mismatch if response_len=0 or 1
            if response_len > 1:
                 logprobs = torch.gather(logprobs_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                 ref_logprobs = torch.gather(ref_logprobs_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                 # Extract values corresponding to the states *before* generating the response tokens
                 # Shape: (batch, response_len) -> includes value for state before 1st resp token
                 values_response = values[:, prompt_len -1 : prompt_len + response_len -1]
            else:
                 # Handle cases with very short responses (e.g., length 0 or 1)
                 logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                 ref_logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                 values_response = torch.zeros((prompt_ids.shape[0], 0), device=device)


            # Decode generated text (full sequence)
            full_decoded_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True)

            # Calculate rewards based on decoded text and ground truth
            rewards = torch.tensor(
                [compute_gsm8k_reward(txt, gt) for txt, gt in zip(full_decoded_texts, ground_truths)],
                dtype=torch.float32, device=device
            )

            # Store rollout data
            rollout_buffer["prompt_input_ids"].append(prompt_ids.cpu())
            rollout_buffer["prompt_attention_mask"].append(prompt_mask.cpu())
            # Store response *excluding* potentially added EOS/PAD by generate if not part of target
            # Use the response mask derived earlier
            rollout_buffer["response_input_ids"].append(response_ids.cpu())
            rollout_buffer["response_attention_mask"].append(response_mask.cpu()) # Crucial mask for loss calc
            rollout_buffer["logprobs"].append(logprobs.cpu()) # Shape: (batch, response_len - 1)
            rollout_buffer["ref_logprobs"].append(ref_logprobs.cpu()) # Shape: (batch, response_len - 1)
            rollout_buffer["values"].append(values_response.cpu()) # Shape: (batch, response_len)
            rollout_buffer["rewards"].append(rewards.cpu()) # Shape: (batch,)
            rollout_buffer["full_texts"].extend(full_decoded_texts)
            rollout_buffer["ground_truth_answers"].extend(ground_truths)

    # Collate the buffer
    # Requires careful padding, especially for logprobs, values, response_ids
    # This basic collation assumes relatively uniform response lengths or handles padding implicitly
    # A more robust implementation would use a dedicated padding function.
    collated_buffer = {}
    for key, data_list in rollout_buffer.items():
        if key in ["full_texts", "ground_truth_answers"]:
             collated_buffer[key] = data_list # Keep as list
        elif key == "rewards":
             collated_buffer[key] = torch.cat(data_list, dim=0)
        else:
             # Pad tensors in the list to the max length within that list
             # This simplistic padding might not be perfect if lengths vary wildly
             if data_list: # Check if list is not empty
                 # Use rnn.pad_sequence for variable length sequences if needed
                 # Simple cat assumes fixed length or implicit padding
                 try:
                     collated_buffer[key] = torch.cat(data_list, dim=0)
                 except RuntimeError as e:
                      print(f"Warning: Error during collation for key '{key}', likely due to length mismatch: {e}. Requires proper padding.")
                      # Fallback or error handling needed here
                      # For simplicity, we might drop problematic batches or implement padding
                      # Let's try padding (requires identifying max length per key)
                      if data_list[0].dim() > 0: # Only pad tensors with sequence dim
                           max_len = max(t.shape[1] for t in data_list)
                           padded_list = []
                           for t in data_list:
                                pad_size = max_len - t.shape[1]
                                if pad_size > 0:
                                     # Pad dim 1 (sequence length)
                                     padded_t = F.pad(t, (0, 0, 0, pad_size)) # Pads last dim=0, second last=pad_size
                                     if t.dim() > 2 : # Handle cases like (batch, seq, features)
                                         padded_t = F.pad(t, (0, 0, 0, pad_size))
                                     elif t.dim() == 2: # Handle (batch, seq)
                                          padded_t = F.pad(t, (0, pad_size))
                                     else: # Should not happen for these keys
                                          padded_t = t
                                     padded_list.append(padded_t)

                                else:
                                     padded_list.append(t)

                           collated_buffer[key] = torch.cat(padded_list, dim=0)

                      else: # Scalars or 1D tensors (like rewards, handled above)
                           collated_buffer[key] = torch.cat(data_list, dim=0)

             else:
                 collated_buffer[key] = torch.empty(0) # Handle empty list case

    return collated_buffer

# --- 6. Update Phase ---
def perform_ppo_update(actor_model, optimizer, rollout_buffer, config):
    """Performs PPO optimization steps on the collected rollout data."""
    actor_model.train()
    metrics = {}
    total_loss = 0.0
    ppo_step_count = 0

    # Move necessary data to device and prepare for iteration
    prompt_ids = rollout_buffer["prompt_input_ids"].to(device)
    prompt_mask = rollout_buffer["prompt_attention_mask"].to(device)
    response_ids = rollout_buffer["response_input_ids"].to(device)
    response_mask = rollout_buffer["response_attention_mask"].to(device) # Use stored mask
    logprobs_old = rollout_buffer["logprobs"].to(device)
    ref_logprobs = rollout_buffer["ref_logprobs"].to(device)
    values_old = rollout_buffer["values"].to(device)
    final_rewards = rollout_buffer["rewards"].to(device)

    # Check for empty tensors which can happen with collation issues or empty rollouts
    if response_ids.shape[0] == 0 or response_ids.shape[1] <= 1:
         print("Warning: Skipping update due to empty or too short responses in buffer.")
         return {}


    # Compute KL penalties and advantages ONCE before PPO epochs
    with torch.no_grad():
        # KL(Policy || Reference) - Shape: (batch, response_len - 1)
        kl_per_token = logprobs_old - ref_logprobs
        # Need to align KL shape with values shape for GAE
        # KL is for transitions (t -> t+1), Values are for states (t)
        # Pad KL to match values length (response_len) - typically pad start or end with 0
        # Let's assume KL penalty applies starting from the first generated token's transition
        # Pad KL at the beginning to align with values: [0, kl_token1, kl_token2, ...]
        kl_padded = F.pad(kl_per_token, (1, 0), value=0.0) # Pad start of seq dim

        advantages, returns = compute_gae_advantages(
            final_rewards, config["kl_coeff"] * kl_padded, values_old,
            response_mask, config["gamma"], config["lam"]
        )

    # Combine prompt and response for forward pass during update
    full_input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    full_attention_mask = torch.cat((prompt_mask, response_mask), dim=1)

    # Data indices for mini-batch sampling
    num_samples = prompt_ids.shape[0]
    indices = np.arange(num_samples)
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]

    # PPO Update Epochs
    for ppo_epoch in range(config["ppo_epochs"]):
        np.random.shuffle(indices)
        # Mini-batch iteration
        for i in range(0, num_samples, config["mini_batch_size"]):
            ppo_step_count += 1
            batch_indices = indices[i : i + config["mini_batch_size"]]
            # Slice all necessary tensors using batch_indices
            batch_full_ids = full_input_ids[batch_indices]
            batch_full_mask = full_attention_mask[batch_indices]
            batch_response_ids = response_ids[batch_indices]
            batch_response_mask = response_mask[batch_indices] # Crucial for masking losses
            batch_logprobs_old = logprobs_old[batch_indices]
            batch_values_old = values_old[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]


            # Forward pass with current actor model
            outputs = actor_model(batch_full_ids, attention_mask=batch_full_mask)
            logits_new = outputs.logits
            values_new = outputs.value.squeeze(-1) # Shape: (mini_batch, full_seq_len)

            # Calculate logprobs for *response* part using new logits
            logprobs_logits_new = F.log_softmax(logits_new[:, prompt_len - 1 : prompt_len + response_len - 1, :], dim=-1)
            target_ids = batch_response_ids[:, 1:] # Shape: (mini_batch, response_len - 1)

            # Handle potential length mismatch if response_len=0 or 1
            if response_len > 1:
                 logprobs_new = torch.gather(logprobs_logits_new, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                 # Extract corresponding values
                 values_new_response = values_new[:, prompt_len -1 : prompt_len + response_len -1]
            else:
                 # Should have been caught earlier, but safety check
                 logprobs_new = torch.zeros((len(batch_indices), 0), device=device)
                 values_new_response = torch.zeros((len(batch_indices), 0), device=device)


            # Compute PPO losses using the core logic functions
            policy_loss, p_clip_frac, approx_kl = compute_policy_loss(
                logprobs_new, batch_logprobs_old, batch_advantages,
                batch_response_mask[:, 1:], config["clip_ratio"] # Mask needs align with logprobs length
            )
            value_loss, v_clip_frac = compute_value_loss(
                values_new_response, batch_values_old, batch_returns,
                batch_response_mask, config["clip_range_value"] # Mask aligns with values length
            )
            entropy_loss = compute_entropy_loss(
                logits_new[:, prompt_len - 1 : prompt_len + response_len - 1, :], # Logits for response part
                batch_response_mask[:, 1:] # Mask aligns with logprobs length
            )

            # Combine losses
            loss = policy_loss + config["vf_coeff"] * value_loss + config["entropy_coeff"] * entropy_loss

            # Scale loss for gradient accumulation and perform backward pass
            scaled_loss = loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()

            # Store metrics (average over mini-batches within epoch)
            # Use .item() and detach()
            metrics.setdefault("loss/policy", []).append(policy_loss.item())
            metrics.setdefault("loss/value", []).append(value_loss.item())
            metrics.setdefault("loss/entropy", []).append(-entropy_loss.item()) # Store positive entropy
            metrics.setdefault("loss/total", []).append(loss.item())
            metrics.setdefault("params/policy_clip_frac", []).append(p_clip_frac.item())
            metrics.setdefault("params/value_clip_frac", []).append(v_clip_frac.item())
            metrics.setdefault("params/approx_kl", []).append(approx_kl.item())

            # Optimizer step after accumulating gradients
            if ppo_step_count % config["gradient_accumulation_steps"] == 0:
                # Optional: Gradient Clipping (common for LLM training)
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()


    # Aggregate metrics over the PPO epoch
    final_metrics = {key: np.mean(val) for key, val in metrics.items()}
    return final_metrics


# --- 7. Main Training Loop ---
optimizer = AdamW(actor_model.parameters(), lr=config["learning_rate"])
dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
total_steps = len(dataloader) // config["gradient_accumulation_steps"] # Steps per 'epoch' over dataset

print("--- Starting PPO Training ---")
for ppo_step in range(config["total_ppo_steps"]):
    print(f"\nPPO Step {ppo_step + 1}/{config['total_ppo_steps']}")

    # --- Rollout Phase ---
    print("Phase 1: Generating Rollouts...")
    # Re-create dataloader to ensure shuffling if desired per PPO step
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    rollout_buffer = perform_rollouts(actor_model, ref_model, tokenizer, dataloader, generation_config)

    # Basic Rollout Stats
    avg_reward = rollout_buffer["rewards"].mean().item()
    print(f"Rollout complete. Average reward: {avg_reward:.4f}")
    print("Sample Generations:")
    num_samples_to_show = 3
    for i in range(min(num_samples_to_show, len(rollout_buffer["full_texts"]))):
         print("-" * 20)
         print(f"Prompt: {tokenizer.decode(rollout_buffer['prompt_input_ids'][i], skip_special_tokens=True)}")
         print(f"Generated: ...{rollout_buffer['full_texts'][i].split('Answer:')[-1]}")
         print(f"Ground Truth: {rollout_buffer['ground_truth_answers'][i]}")
         print(f"Reward: {rollout_buffer['rewards'][i].item():.1f}")
    print("-" * 20)

    # --- Update Phase ---
    print("Phase 2: Performing PPO Updates...")
    # Check buffer validity
    if rollout_buffer["prompt_input_ids"].shape[0] > 0 and rollout_buffer["response_input_ids"].shape[1] > 1 :
         metrics = perform_ppo_update(actor_model, optimizer, rollout_buffer, config)

         # Log metrics
         if (ppo_step + 1) % config["log_interval"] == 0:
             print(f"PPO Step {ppo_step+1} Metrics:")
             log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
             print(log_str)
             print(f"  Reward (mean): {avg_reward:.4f}") # Log reward too
    else:
         print("Skipping update step because rollout buffer is invalid (empty or too short responses).")

    # --- Save Model Checkpoint ---
    if (ppo_step + 1) % config["save_interval"] == 0:
        print(f"Saving model checkpoint at step {ppo_step + 1}...")
        output_path = f"{config['output_dir']}/step_{ppo_step + 1}"
        actor_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

print("--- PPO Training Finished ---")

# --- Final Model Saving ---
print("Saving final model...")
output_path = f"{config['output_dir']}/final"
actor_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Final model saved to {output_path}")

# --- Example Inference (Optional) ---
print("\n--- Running Inference with Final Model ---")
actor_model.eval()
test_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
test_prompt = config["prompt_format"].format(question=test_question)
test_input = tokenizer(test_prompt, return_tensors="pt").to(device)

gen_config_greedy = GenerationConfig( # Use greedy decoding for testing
    max_new_tokens=config["max_gen_length"],
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False
)

with torch.no_grad():
    output_sequences = actor_model.generate(
        input_ids=test_input["input_ids"],
        attention_mask=test_input["attention_mask"],
        generation_config=gen_config_greedy
    )

print(f"Test Prompt:\n{test_prompt}")
print("\nGenerated Response:")
print(tokenizer.decode(output_sequences[0, test_input['input_ids'].shape[1]:], skip_special_tokens=True)) # Decode only generated part

