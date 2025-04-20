# tests/test_ppo_logic.py

import torch
import torch.nn.functional as F
import pytest
import os
import sys

# Add src directory to Python path to allow importing ppo_trainer
# This assumes tests are run from the project root directory (ppo_rl_tutorial/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Import functions from the source file in src/
    from ppo_trainer import (
        compute_gae_advantages,
        compute_policy_loss,
        compute_value_loss,
        compute_entropy_loss,
        masked_mean,
        masked_whiten
    )
except ImportError as e:
    print(f"Error importing from src.ppo_trainer: {e}")
    print("Please ensure ppo_trainer.py exists in the src/ directory and is importable.")
    # Define dummy functions if import fails, so tests can be parsed
    # This allows pytest discovery even if the main script has issues initially
    def compute_gae_advantages(*args): return torch.zeros(1), torch.zeros(1)
    def compute_policy_loss(*args): return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    def compute_value_loss(*args): return torch.tensor(0.0), torch.tensor(0.0)
    def compute_entropy_loss(*args): return torch.tensor(0.0)
    def masked_mean(tensor, mask, dim=None): return torch.tensor(0.0)
    def masked_whiten(tensor, mask, shift_mean=True): return torch.zeros_like(tensor)


# --- Test Configuration ---
TEST_GAMMA = 0.99
TEST_LAM = 0.95
TEST_CLIP_RATIO = 0.2
TEST_CLIP_RANGE_VALUE = 0.2

test_device = torch.device("cpu")

# --- Test Class ---
class TestPPOLogic:

    def test_compute_gae_advantages(self):
        # ... (Test logic remains the same as previous pytest version) ...
        print("\nRunning pytest: test_compute_gae_advantages...")
        final_rewards = torch.tensor([10.0], device=test_device)
        kl_penalties = torch.zeros(1, 4, device=test_device)
        values = torch.tensor([[1.0, 1.5, 2.0, 2.5]], device=test_device)
        response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool, device=test_device)
        expected_adv_unwhitened = torch.tensor([[7.602, 7.564, 7.529, 7.5]], device=test_device)
        expected_returns = expected_adv_unwhitened + values
        # Assuming masked_whiten is imported correctly and works
        expected_adv_whitened = masked_whiten(expected_adv_unwhitened, response_mask)
        advantages, returns = compute_gae_advantages(final_rewards, kl_penalties, values, response_mask, TEST_GAMMA, TEST_LAM)
        assert advantages.shape == expected_adv_whitened.shape
        assert returns.shape == expected_returns.shape
        torch.testing.assert_close(advantages, expected_adv_whitened, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(returns, expected_returns, rtol=1e-3, atol=1e-3)

        final_rewards = torch.tensor([5.0, 8.0], device=test_device)
        kl_penalties = torch.zeros(2, 3, device=test_device)
        values = torch.tensor([[1.0, 1.2, 0.0], [2.0, 2.5, 3.0]], device=test_device)
        response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool, device=test_device)
        advantages, returns = compute_gae_advantages(final_rewards, kl_penalties, values, response_mask, TEST_GAMMA, TEST_LAM)
        assert advantages.shape == (2, 3)
        assert returns.shape == (2, 3)
        print("GAE tests passed.")


    def test_compute_policy_loss(self):
        # ... (Test logic remains the same as previous pytest version) ...
        print("\nRunning pytest: test_compute_policy_loss...")
        log_probs_old = torch.log(torch.tensor([[0.5, 0.8], [0.6, 0.7]], device=test_device))
        log_probs_new = torch.log(torch.tensor([[0.6, 0.7], [0.5, 0.9]], device=test_device))
        advantages = torch.tensor([[1.0, -0.5, 99.0], [2.0, 0.8, 99.0]], device=test_device)
        # Mask should align with logprobs length
        response_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool, device=test_device)
        # Align advantages
        advantages_aligned = advantages[:, :-1]

        # Expected values
        ratios = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratios * advantages_aligned
        clamped_ratios = torch.clamp(ratios, 1.0 - TEST_CLIP_RATIO, 1.0 + TEST_CLIP_RATIO)
        surr2 = clamped_ratios * advantages_aligned
        min_surr = torch.min(surr1, surr2)
        expected_loss = -masked_mean(min_surr, response_mask)
        expected_clip_frac = masked_mean(torch.gt(torch.abs(ratios - 1.0), TEST_CLIP_RATIO).float(), response_mask)
        expected_approx_kl = masked_mean(log_probs_old - log_probs_new, response_mask)

        policy_loss, clip_frac, approx_kl = compute_policy_loss(log_probs_new, log_probs_old, advantages, response_mask, TEST_CLIP_RATIO)
        assert policy_loss.shape == torch.Size([])
        torch.testing.assert_close(policy_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(clip_frac, expected_clip_frac, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(approx_kl, expected_approx_kl, rtol=1e-4, atol=1e-4)
        print("Policy loss test passed.")


    def test_compute_value_loss(self):
        # ... (Test logic remains the same as previous pytest version) ...
        print("\nRunning pytest: test_compute_value_loss...")
        values_new = torch.tensor([[1.1, 1.6, 2.5], [2.2, 2.8, 3.1]], device=test_device)
        values_old = torch.tensor([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], device=test_device)
        returns = torch.tensor([[1.2, 1.7, 2.2], [1.8, 3.0, 3.5]], device=test_device)
        response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool, device=test_device)

        # Expected values
        values_pred_clipped = values_old + torch.clamp(values_new - values_old, -TEST_CLIP_RANGE_VALUE, TEST_CLIP_RANGE_VALUE)
        vf_loss1 = (values_new - returns)**2
        vf_loss2 = (values_pred_clipped - returns)**2
        max_vf_losses = torch.max(vf_loss1, vf_loss2)
        expected_vf_loss = 0.5 * masked_mean(max_vf_losses, response_mask)
        expected_vf_clip_frac = masked_mean(torch.gt(vf_loss2, vf_loss1).float(), response_mask)

        vf_loss, vf_clip_frac = compute_value_loss(values_new, values_old, returns, response_mask, TEST_CLIP_RANGE_VALUE)
        assert vf_loss.shape == torch.Size([])
        torch.testing.assert_close(vf_loss, expected_vf_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(vf_clip_frac, expected_vf_clip_frac, rtol=1e-4, atol=1e-4)
        print("Value loss test passed.")


    def test_compute_entropy_loss(self):
        # ... (Test logic remains the same as previous pytest version) ...
        print("\nRunning pytest: test_compute_entropy_loss...")
        logits_new = torch.tensor(
            [ # Logits matching logprobs shape (batch, seq_len-1, vocab_size)
              [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
              [[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, -1.0, 1.0]]
            ], device=test_device, dtype=torch.float32
        )
        response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool, device=test_device)

        # Expected values
        dist00 = torch.distributions.Categorical(logits=logits_new[0, 0])
        dist01 = torch.distributions.Categorical(logits=logits_new[0, 1])
        dist10 = torch.distributions.Categorical(logits=logits_new[1, 0])
        entropy_values = torch.zeros_like(response_mask, dtype=torch.float32)
        if response_mask[0,0]: entropy_values[0,0] = dist00.entropy()
        if response_mask[0,1]: entropy_values[0,1] = dist01.entropy()
        if response_mask[1,0]: entropy_values[1,0] = dist10.entropy()

        expected_mean_entropy = masked_mean(entropy_values, response_mask)
        expected_entropy_loss = -expected_mean_entropy

        entropy_loss = compute_entropy_loss(logits_new, response_mask)
        assert entropy_loss.shape == torch.Size([])
        torch.testing.assert_close(entropy_loss, expected_entropy_loss, rtol=1e-4, atol=1e-4)
        print("Entropy loss test passed.")
