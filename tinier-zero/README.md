* (WIP) Reinforcement Learning: TinyZero in A Single File

`tinzero.py` contains the complete implementation.
`tinyzero-exercise.py` has certain parts for user to fill in.

--- Recommended Implementation Order For the Exercises ---

1. `compute_policy_loss`:
   - Why first? This is the core PPO objective function, dictating how the actor learns.
     Understanding the clipped surrogate objective is fundamental.
   - Focus: Implement the ratio calculation, clipped/unclipped objectives, and masked mean.

2. `compute_value_loss`:
   - Why second? This trains the critic, whose value estimates V(s) are needed for
     stable and efficient policy updates (via advantage calculation). Learning how V(s)
     is trained against the target 'returns' is key.
   - Focus: Implement value clipping and the MSE calculation against returns.

3. `compute_entropy_loss`:
   - Why third? A simpler but important component for encouraging exploration.
   - Focus: Calculate policy distribution entropy from logits and take the masked mean.

4. `compute_gae_advantages`:
   - Why fourth? This calculates the 'advantages' (how good were the actions?) and
     'returns' (what's the target for the value function?) needed by the policy and
     value loss functions. Implementing it after the losses provides context for why these
     inputs are calculated this way (incorporating rewards, KL penalty, and value estimates).
   - Focus: Implement the backward GAE loop, combining rewards/KL/values. Whiten advantages.

5. `perform_ppo_update` (The Update Loop Logic):
   - Why last? This orchestrates the entire learning step, bringing together the rollout
     data, GAE calculation, and all the loss functions you've implemented.
   - Focus: Structure the PPO epoch and mini-batch loops, correctly call GAE and loss
     functions, combine losses, and perform the backward pass/optimizer step.

--- The Big Picture: How It Fits Together ---

The PPO algorithm iterates through a cycle of experience gathering and policy improvement:

A. Rollout Phase (`perform_rollouts`):
   - The current Actor model generates responses (trajectories) based on input prompts.
   - During generation, we store crucial data:
     - Input prompts and generated responses.
     - `logprobs` of the generated tokens under the Actor policy (needed for importance sampling ratio).
     - `values` estimated by the Critic for each state/token (needed for GAE).
     - `ref_logprobs` of generated tokens under the frozen Reference policy (needed for KL penalty).
     - `rewards` obtained from the environment (e.g., 1.0 for correct GSM8K answer, 0 otherwise).

B. Advantage Calculation Phase (within `perform_ppo_update`):
   - The collected rollout data (`rewards`, `values`, `logprobs`, `ref_logprobs`) is processed.
   - First, KL penalties are calculated (`logprobs - ref_logprobs`).
   - Then, `compute_gae_advantages` uses the rewards, KL penalties, and values to estimate:
     - `advantages`: How much better were the generated actions than expected by the critic? (Incorporates KL).
     - `returns`: What is the actual discounted reward-to-go observed? (Target for the critic).

C. Update Phase (`perform_ppo_update`):
   - This phase uses the rollout data and the calculated advantages/returns to update the model.
   - It loops for multiple `ppo_epochs` over the *same* batch of rollout data.
   - Within each epoch, it iterates over mini-batches:
     - It re-evaluates the generated sequences with the *current* Actor and Critic to get `logprobs_new` and `values_new`.
     - It calculates the three core losses using your implemented functions:
       - `compute_policy_loss` uses `logprobs_new`, `logprobs_old` (from rollout), and `advantages` to calculate how to adjust the policy.
       - `compute_value_loss` uses `values_new` and `returns` to improve the critic's predictions.
       - `compute_entropy_loss` uses the `logits_new` to encourage exploration.
     - These losses are combined into a single loss value.
     - Standard backpropagation and an optimizer step (`optimizer.step()`) update the Actor and Critic model weights based on this combined loss.

D. Repeat:
   - The entire cycle (Rollout -> GAE -> Update) repeats, using the newly updated Actor policy
     to generate the next batch of rollouts.

By iterating this loop, the PPO algorithm gradually improves the Actor's policy to generate
responses that maximize the expected reward (while staying reasonably close to the reference policy
and maintaining exploration), guided by the advantage estimates and the learned value function.
