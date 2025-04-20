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
    "kl_coeff": 0.1,       # KL penalty coefficient (beta) - Controls deviation from ref model
    "clip_ratio": 0.2,     # PPO policy objective clipping - Prevents large policy updates
    "clip_range_value": 0.2,# PPO value function clipping - Stabilizes value function updates
    "vf_coeff": 0.1,       # Value function loss weight - Balances policy and value learning
    "entropy_coeff": 0.01, # Entropy bonus weight - Encourages exploration
    "gamma": 1.0,          # Discount factor (1.0 often used for non-episodic LLM tasks)
    "lam": 0.95,           # GAE lambda - Controls bias-variance trade-off in advantage estimation

    # Training Control
    "total_ppo_steps": 100, # Number of PPO steps (Rollout -> Update)
    "seed": 42,
    "log_interval": 1,     # Log metrics every N PPO steps
    "save_interval": 10,   # Save model every N PPO steps
    "output_dir": "ppo_gsm8k_qwen1.8b_exercise",
}

# --- Setup ---
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model, Tokenizer ---
# We need three models:
# 1. Actor: The model we are actively training with PPO. It generates responses.
# 2. Critic: Estimates the expected future reward (value) from a given state (sequence).
#    Often implemented as a value head on top of the Actor's base model.
# 3. Reference (Ref): A frozen copy of the initial model (e.g., SFT model).
#    Used to calculate KL divergence penalty, keeping the Actor from deviating too much.

print(f"Loading model: {config['model_name']}")
quantization_config = None
if config["load_in_4bit"]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Or float16 if bf16 not supported
    )

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
     print("Warning: Setting pad_token to eos_token.")
     tokenizer.pad_token = tokenizer.eos_token
     tokenizer.pad_token_id = tokenizer.eos_token_id

# Load Actor model with Value Head using TRL's helper class for convenience
model_kwargs = {"quantization_config": quantization_config} if config["load_in_4bit"] else {}
actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"],
    trust_remote_code=True,
    **model_kwargs
).to(device)

# Load Reference model (also with value head class for simplicity, though head isn't used)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"],
    trust_remote_code=True,
    **model_kwargs
).to(device)

# Freeze the reference model - its weights must not change during PPO training.
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()
print("Models loaded.")

# --- 2. Load Dataset & Preprocess ---
# (Dataset loading remains the same)
print(f"Loading dataset: {config['dataset_name']}")
dataset = load_dataset(config["dataset_name"], config["dataset_split"])

def preprocess_dataset(example):
    example["prompt"] = config["prompt_format"].format(question=example["question"])
    tokenized_prompt = tokenizer(
        example["prompt"],
        max_length=config["max_prompt_length"],
        truncation=True,
        padding=False,
        return_tensors=None
    )
    example["input_ids"] = tokenized_prompt["input_ids"]
    example["attention_mask"] = tokenized_prompt["attention_mask"]
    example["ground_truth_answer"] = example["answer"].split("####")[-1].strip()
    return example

tokenized_dataset = dataset.map(preprocess_dataset, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch")
print(f"Dataset preprocessed. Number of prompts: {len(tokenized_dataset)}")

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding='longest',
        return_tensors="pt",
        return_attention_mask=True,
    )
    ground_truths = [item['ground_truth_answer'] for item in batch]
    return {
        "prompt_input_ids": padded_inputs["input_ids"],
        "prompt_attention_mask": padded_inputs["attention_mask"],
        "ground_truth_answers": ground_truths
    }

# --- 3. GSM8K Reward Function ---
# (Reward function remains the same)
def extract_gsm8k_solution(solution_str):
    solution = re.search(r"####\s*([-+]?\s*[\d\.\,]+)", solution_str)
    if solution is None:
         answer = re.findall(r"([-+]?\s*[\d\.\,]+)", solution_str)
         if len(answer) > 0:
              final_answer_str = answer[-1].replace(',', '').replace(' ', '')
              try: float(final_answer_str); return final_answer_str
              except ValueError: return None
         else: return None
    else: return solution.group(1).replace(',', '').replace(' ', '')

def compute_gsm8k_reward(generated_text, ground_truth_str):
    extracted_answer_str = extract_gsm8k_solution(generated_text)
    if extracted_answer_str is None: return 0.0
    try:
        if math.isclose(float(extracted_answer_str), float(ground_truth_str)): return 1.0
        else: return 0.0
    except ValueError: return 0.0


# --- 4. Core PPO Logic (Exercise: Implement these functions with Understanding) ---

# Helper for masked operations (keep these)
def masked_mean(tensor, mask, dim=None):
    if mask is None: return torch.mean(tensor, dim=dim)
    mask = mask.bool(); mask = mask.expand_as(tensor)
    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    return masked_tensor.sum(dim=dim) / (mask.sum(dim=dim).float() + 1e-8)

def masked_whiten(tensor, mask, shift_mean=True):
    mask = mask.bool(); mask = mask.expand_as(tensor)
    mean = masked_mean(tensor, mask, dim=None)
    var = masked_mean(torch.where(mask, (tensor - mean)**2, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)), mask, dim=None)
    std = torch.sqrt(var + 1e-8)
    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(mask, whitened, tensor)


# --- EXERCISE START: Implement Core RL Functions ---

def compute_gae_advantages(final_rewards, kl_penalties, values, response_mask, gamma, lam):
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
        kl_penalties (torch.Tensor): Shape (batch_size, response_length) - KL penalty (logprobs_actor - logprobs_ref).
        values (torch.Tensor): Shape (batch_size, response_length) - Critic's state value estimates (V(s_t)).
        response_mask (torch.Tensor): Shape (batch_size, response_length) - Mask for valid response tokens.
        gamma (float): Discount factor for future rewards.
        lam (float): GAE lambda parameter for bias-variance trade-off.

    Steps:
    1. Initialize `advantages` tensor.
    2. Construct `token_level_rewards`: Base reward is 0 everywhere except the last valid token,
       which gets the `final_rewards`. Then, subtract the `kl_penalties` (scaled by kl_coeff later)
       at each step. This combines the task objective with the KL constraint.
    3. Iterate backwards through the sequence (`t` from `response_length-1` down to `0`).
    4. Calculate the TD error (delta) for step `t`: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`.
       - `r_t` is the token-level reward (including KL penalty) at step `t`.
       - `V(s_{t+1})` is the value of the *next* state (use `values[:, t + 1]`). Handle boundary: `V(s_T) = 0`.
       - Use `response_mask` to ensure calculations only happen for valid tokens.
    5. Calculate the GAE advantage for step `t`: `A(s_t) = delta_t + gamma * lambda * A(s_{t+1})`.
       - Accumulate the advantage recursively. Use `response_mask` to reset advantage after sequence ends.
    6. Store advantages.
    7. Compute returns: `Returns(s_t) = A(s_t) + V(s_t)`. Returns are the target values for the critic update.
    8. Whiten (normalize) advantages: This stabilizes training by ensuring advantages have zero mean and unit variance. Use `masked_whiten`.

    Returns:
        advantages (torch.Tensor): Shape (batch_size, response_length) - Whitened GAE advantages.
        returns (torch.Tensor): Shape (batch_size, response_length) - Target values for the critic.
    """
    advantages = torch.zeros_like(values) # Placeholder
    returns = torch.zeros_like(values) # Placeholder

    # <<<< YOUR GAE IMPLEMENTATION HERE >>>>
    # Remember to use torch.no_grad() context

    print("Warning: GAE calculation not implemented!") # Remove this line
    # Dummy implementation for shape consistency:
    advantages = masked_whiten(torch.randn_like(values), response_mask)
    returns = advantages + values

    return advantages, returns

def compute_policy_loss(log_probs_new, log_probs_old, advantages, response_mask, clip_ratio):
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
        log_probs_new (torch.Tensor): Log probabilities from current policy. Shape (batch, resp_len-1).
        log_probs_old (torch.Tensor): Log probabilities from rollout policy. Shape (batch, resp_len-1).
        advantages (torch.Tensor): GAE advantages. Shape (batch, resp_len).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len-1).
        clip_ratio (float): PPO clipping parameter (epsilon).

    Steps:
    1. Calculate probability ratio: `ratio = exp(log_probs_new - log_probs_old)`.
    2. Align advantages: Ensure `advantages` tensor corresponds to the states *before* the actions whose
       logprobs are given. Often `advantages[:, :-1]` aligns with `log_probs_new`/`log_probs_old`.
    3. Calculate unclipped objective: `surr1 = ratio * advantages_aligned`.
    4. Calculate clipped objective: `surr2 = clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_aligned`.
    5. Policy loss = `- mean(min(surr1, surr2))`. Use `masked_mean` with the appropriate mask. The negative sign
       turns maximization of the objective into minimization of the loss.
    6. (Optional) Calculate `clip_frac` and `approx_kl` for logging.

    Returns:
        policy_loss (torch.Tensor): Scalar policy loss.
        clip_frac (torch.Tensor): Scalar fraction of clipped samples.
        approx_kl (torch.Tensor): Scalar approximate KL divergence.
    """
    policy_loss = torch.tensor(0.0, device=log_probs_new.device) # Placeholder
    clip_frac = torch.tensor(0.0, device=log_probs_new.device) # Placeholder
    approx_kl = torch.tensor(0.0, device=log_probs_new.device) # Placeholder

    # Align advantages with logprobs (assuming logprobs correspond to actions a_1 to a_{T-1})
    advantages_aligned = advantages[:, :-1] if advantages.shape[1] == log_probs_old.shape[1] + 1 else advantages

    if advantages_aligned.shape == log_probs_old.shape:
        # <<<< YOUR POLICY LOSS IMPLEMENTATION HERE >>>>
        print("Warning: Policy loss calculation not implemented!") # Remove this line
    else:
        print(f"Warning: Shape mismatch for policy loss calculation. Advantages: {advantages.shape}, LogProbs Old: {log_probs_old.shape}")

    return policy_loss, clip_frac, approx_kl

def compute_value_loss(values_new, values_old, returns, response_mask, clip_range_value):
    """
    EXERCISE: Implement PPO value loss (clipped).
    WHY Clipped Value Loss?: Similar to the policy loss, clipping the value function update helps
               stabilize training. It prevents the value function (critic) from changing too
               drastically based on potentially noisy return estimates from a single batch.
               The loss is based on the squared error between the predicted value `values_new`
               and the target `returns`, but the prediction is clipped around the `values_old`
               estimate from the rollout phase.

    Args:
        values_new (torch.Tensor): Values predicted by current critic. Shape (batch, resp_len).
        values_old (torch.Tensor): Values predicted by rollout critic. Shape (batch, resp_len).
        returns (torch.Tensor): Target values (calculated from GAE). Shape (batch, resp_len).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len).
        clip_range_value (float): Clipping parameter for value loss.

    Steps:
    1. Clip `values_new` around `values_old`: `values_pred_clipped = values_old + clamp(...)`.
    2. Calculate squared error for unclipped prediction: `vf_loss1 = (values_new - returns)^2`.
    3. Calculate squared error for clipped prediction: `vf_loss2 = (values_pred_clipped - returns)^2`.
    4. Value loss = `0.5 * mean(max(vf_loss1, vf_loss2))`. Use `masked_mean` with `response_mask`.
    5. (Optional) Calculate `vf_clip_frac` for logging.

    Returns:
        value_loss (torch.Tensor): Scalar value loss.
        vf_clip_frac (torch.Tensor): Scalar fraction of clipped value samples.
    """
    value_loss = torch.tensor(0.0, device=values_new.device) # Placeholder
    vf_clip_frac = torch.tensor(0.0, device=values_new.device) # Placeholder

    # <<<< YOUR VALUE LOSS IMPLEMENTATION HERE >>>>
    print("Warning: Value loss calculation not implemented!") # Remove this line

    return value_loss, vf_clip_frac

def compute_entropy_loss(logits_new, response_mask):
    """
    EXERCISE: Implement entropy loss calculation.
    WHY Entropy Bonus?: Entropy measures the randomness or uncertainty of the policy's action
             distribution. Adding an entropy bonus (or subtracting an entropy loss) encourages
             the policy to explore by making its action choices less deterministic. This can
             prevent the policy from collapsing prematurely to a suboptimal strategy. The
             `entropy_coeff` controls the strength of this exploration incentive.

    Args:
        logits_new (torch.Tensor): Logits from current policy. Shape (batch, resp_len-1, vocab_size).
        response_mask (torch.Tensor): Mask for valid response tokens. Shape (batch, resp_len-1).

    Steps:
    1. Create a categorical distribution from `logits_new`.
    2. Calculate the entropy of the distribution for each token position.
    3. Compute the masked mean of the entropy using `response_mask`.
    4. Entropy loss = `- mean_entropy` (negative because we add the bonus, equivalent to subtracting loss).

    Returns:
        entropy_loss (torch.Tensor): Scalar entropy loss.
    """
    entropy_loss = torch.tensor(0.0, device=logits_new.device) # Placeholder

    # <<<< YOUR ENTROPY LOSS IMPLEMENTATION HERE >>>>
    print("Warning: Entropy loss calculation not implemented!") # Remove this line

    return entropy_loss

# --- EXERCISE END: Implement Core RL Functions ---


# --- 5. Rollout Phase ---
generation_config = GenerationConfig(
    max_new_tokens=config["max_gen_length"],
    min_new_tokens=5,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

def perform_rollouts(actor_model, ref_model, tokenizer, prompt_dataloader, gen_config):
    """
    Generates responses and computes necessary data for PPO update.
    WHY Rollouts?: This phase gathers experience by interacting with the 'environment'
                 (in this case, generating text based on prompts) using the current policy (actor).
                 We need to collect not just the generated text and rewards, but also critical
                 information about the policy's state during generation (log probabilities,
                 value estimates) to perform the PPO update calculations later.
    """
    rollout_buffer = {
        "prompt_input_ids": [], "prompt_attention_mask": [],
        "response_input_ids": [], "response_attention_mask": [],
        "logprobs": [], "ref_logprobs": [], "values": [],
        "rewards": [], "full_texts": [], "ground_truth_answers": []
    }
    actor_model.eval() # Set models to evaluation mode for generation
    ref_model.eval()

    progress_bar = tqdm(prompt_dataloader, desc="Rollout", leave=False)
    for batch in progress_bar:
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        ground_truths = batch["ground_truth_answers"]

        with torch.no_grad(): # Disable gradient calculations during rollout
            # Step 1: Generate responses using the current actor policy.
            # This simulates taking actions (generating tokens) in the environment.
            generated_output = actor_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            response_ids = generated_output.sequences[:, prompt_ids.shape[1]:]

            # Step 2: Prepare full sequences (prompt + response) for analysis.
            full_ids = torch.cat((prompt_ids, response_ids), dim=1)
            response_mask = (response_ids != tokenizer.pad_token_id).long()
            full_mask = torch.cat((prompt_mask, response_mask), dim=1)

            # Step 3: Perform forward passes to get needed values for PPO.
            # We need logprobs from the actor (for policy updates) and reference (for KL penalty),
            # and value estimates from the critic (for advantage calculation).
            outputs = actor_model(full_ids, attention_mask=full_mask)
            logits = outputs.logits
            values = outputs.value.squeeze(-1)

            ref_outputs = ref_model(full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits

            # Step 4: Calculate log probabilities of the *generated* tokens.
            # This requires careful slicing and indexing based on next-token prediction setup.
            prompt_len = prompt_ids.shape[1]
            response_len = response_ids.shape[1]
            if response_len > 1:
                 logprobs_logits = F.log_softmax(logits[:, prompt_len - 1 : prompt_len + response_len - 1, :], dim=-1)
                 ref_logprobs_logits = F.log_softmax(ref_logits[:, prompt_len - 1 : prompt_len + response_len - 1, :], dim=-1)
                 target_ids = response_ids[:, 1:]
                 logprobs = torch.gather(logprobs_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                 ref_logprobs = torch.gather(ref_logprobs_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                 # Get values corresponding to states *before* generating tokens
                 values_response = values[:, prompt_len -1 : prompt_len + response_len -1]
            else: # Handle short/empty responses
                 logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                 ref_logprobs = torch.zeros((prompt_ids.shape[0], 0), device=device)
                 values_response = torch.zeros((prompt_ids.shape[0], 0), device=device)

            # Step 5: Calculate the final reward for the generated sequence based on the task.
            full_decoded_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=True)
            rewards = torch.tensor(
                [compute_gsm8k_reward(txt, gt) for txt, gt in zip(full_decoded_texts, ground_truths)],
                dtype=torch.float32, device=device
            )

            # Step 6: Store all computed data in the rollout buffer.
            # Ensure we store tensors on CPU to free up GPU memory if needed.
            rollout_buffer["prompt_input_ids"].append(prompt_ids.cpu())
            rollout_buffer["prompt_attention_mask"].append(prompt_mask.cpu())
            rollout_buffer["response_input_ids"].append(response_ids.cpu())
            rollout_buffer["response_attention_mask"].append(response_mask.cpu())
            rollout_buffer["logprobs"].append(logprobs.cpu())
            rollout_buffer["ref_logprobs"].append(ref_logprobs.cpu())
            rollout_buffer["values"].append(values_response.cpu())
            rollout_buffer["rewards"].append(rewards.cpu())
            rollout_buffer["full_texts"].extend(full_decoded_texts)
            rollout_buffer["ground_truth_answers"].extend(ground_truths)

    # Collate the buffer after collecting data from all batches.
    # (Keep collation logic intact)
    collated_buffer = {}
    for key, data_list in rollout_buffer.items():
        if key in ["full_texts", "ground_truth_answers"]: collated_buffer[key] = data_list
        elif key == "rewards":
             if data_list: collated_buffer[key] = torch.cat(data_list, dim=0)
             else: collated_buffer[key] = torch.empty(0)
        else:
             if data_list:
                 try: collated_buffer[key] = torch.cat(data_list, dim=0)
                 except RuntimeError:
                     if data_list[0].dim() > 1:
                          max_len = max(t.shape[1] for t in data_list if t.dim()>1)
                          padded_list = []
                          for t in data_list:
                              pad_size = max_len - t.shape[1]
                              if pad_size > 0:
                                  if t.dim() == 3: padded_t = F.pad(t, (0, 0, 0, pad_size))
                                  elif t.dim() == 2: padded_t = F.pad(t, (0, pad_size))
                                  else: padded_t = t
                                  padded_list.append(padded_t)
                              else: padded_list.append(t)
                          collated_buffer[key] = torch.cat(padded_list, dim=0)
                     else:
                          print(f"Warning: Could not collate/pad key '{key}'"); collated_buffer[key] = torch.empty(0)
             else: collated_buffer[key] = torch.empty(0)
    return collated_buffer


# --- 6. Update Phase (Exercise: Implement PPO Update Logic) ---
def perform_ppo_update(actor_model, optimizer, rollout_buffer, config):
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
    actor_model.train() # Set model to training mode
    metrics = {}
    ppo_step_count = 0 # Counter for grad accumulation

    # <<<< YOUR PPO UPDATE LOOP IMPLEMENTATION HERE >>>>
    # Steps:
    # 1. Prepare data: Load tensors from buffer to device. Check buffer validity.
    # 2. Compute advantages & returns: Calculate KL, call your `compute_gae_advantages`.
    # 3. PPO Epoch Loop (config["ppo_epochs"]):
    #    - Shuffle indices for mini-batch sampling.
    #    - Mini-batch Loop:
    #       - Get mini-batch data slices.
    #       - Forward pass with *current* actor model -> logits_new, values_new.
    #       - Calculate logprobs_new from logits_new.
    #       - Calculate losses (policy, value, entropy) using your implemented functions.
    #       - Combine losses: loss = policy_loss + vf_coeff * value_loss + entropy_coeff * entropy_loss.
    #       - Backward pass: Scale loss for grad accumulation, loss.backward().
    #       - Store metrics.
    #       - Optimizer step: If enough gradients accumulated, clip grads (optional), optimizer.step(), optimizer.zero_grad().
    # 4. Aggregate and return metrics.

    print("Warning: PPO update loop not implemented!")

    # Placeholder metrics
    final_metrics = {"loss/total": 0.0, "loss/policy": 0.0, "loss/value": 0.0, "loss/entropy": 0.0}

    return final_metrics


# --- 7. Main Training Loop ---
# (Keep main loop intact)
if __name__ == "main":
    optimizer = AdamW(actor_model.parameters(), lr=config["learning_rate"])
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    print("--- Starting PPO Training (Exercise Mode) ---")
    for ppo_step in range(config["total_ppo_steps"]):
        print(f"\nPPO Step {ppo_step + 1}/{config['total_ppo_steps']}")

        # --- Rollout Phase ---
        print("Phase 1: Generating Rollouts...")
        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        rollout_buffer = perform_rollouts(actor_model, ref_model, tokenizer, dataloader, generation_config)

        # Basic Rollout Stats
        avg_reward = 0.0
        if rollout_buffer and "rewards" in rollout_buffer and rollout_buffer["rewards"].numel() > 0 :
            avg_reward = rollout_buffer["rewards"].mean().item()
            print(f"Rollout complete. Average reward: {avg_reward:.4f}")
            # (Sample generation printing kept for inspection)
            # ...
        else:
            print("Rollout buffer seems empty or invalid after generation.")

        # --- Update Phase ---
        print("Phase 2: Performing PPO Updates...")
        is_buffer_valid = (
            rollout_buffer and
            "prompt_input_ids" in rollout_buffer and rollout_buffer["prompt_input_ids"].numel() > 0 and
            "response_input_ids" in rollout_buffer and rollout_buffer["response_input_ids"].shape[1] > 1
        )
        if is_buffer_valid:
            metrics = perform_ppo_update(actor_model, optimizer, rollout_buffer, config)
            if metrics and (ppo_step + 1) % config["log_interval"] == 0:
                print(f"PPO Step {ppo_step+1} Metrics (from update):")
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(log_str)
                print(f"  Reward (mean from rollout): {avg_reward:.4f}")
            elif not metrics: print("Update function returned empty metrics.")
        else: print("Skipping update step because rollout buffer is invalid.")

        # --- Save Model Checkpoint ---
        if (ppo_step + 1) % config["save_interval"] == 0:
            print(f"Saving model checkpoint at step {ppo_step + 1}...")
            output_path = f"{config['output_dir']}/step_{ppo_step + 1}"
            try:
                actor_model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                print(f"Model saved to {output_path}")
            except Exception as e: print(f"Error saving model: {e}")

    print("--- PPO Training Finished ---")

    # --- Final Model Saving ---
    # (Keep final saving and inference logic intact)
    # ...
