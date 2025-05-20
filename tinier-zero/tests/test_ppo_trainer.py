"""
Unit tests for the refactored PPO trainer script using pytest.
"""

# To test the provided solutions in ppo_trainer_solutions.py instead of your
# implementations in ppo_trainer.py, you can temporarily modify the import
# statements below. For example, change:
# from ppo_trainer import compute_policy_loss, ...
# to:
# from ppo_trainer_solutions import compute_policy_loss, ...
# Remember to revert these changes to test your own exercise implementations.

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from omegaconf import OmegaConf # For dummy config if needed by tested funcs

# --- Import functions from the refactored script ---
# Adjust the import path if your file structure is different
try:
    from ppo_trainer import (
        masked_mean,
        masked_whiten,
        extract_gsm8k_solution,
        compute_gsm8k_reward,
        pad_and_collate_tensors,
        compute_policy_loss,
        compute_value_loss,
        compute_entropy_loss,
        compute_gae_advantages
        # ActorModelWithValueHead is not unit tested here (requires model loading)
        # Other functions like generate_responses, calculate_rollout_stats, etc.,
        # are harder to unit test due to model dependencies.
    )
except ImportError as e:
    print(f"Error importing from ppo_trainer_refactored.py: {e}")
    print("Please ensure the refactored script is in the Python path.")
    # Define dummy functions to allow tests to be collected, though they will fail
    def masked_mean(*args, **kwargs): raise NotImplementedError("Import failed")
    def masked_whiten(*args, **kwargs): raise NotImplementedError("Import failed")
    def extract_gsm8k_solution(*args, **kwargs): raise NotImplementedError("Import failed")
    def compute_gsm8k_reward(*args, **kwargs): raise NotImplementedError("Import failed")
    def pad_and_collate_tensors(*args, **kwargs): raise NotImplementedError("Import failed")
    def compute_policy_loss(*args, **kwargs): raise NotImplementedError("Import failed")
    def compute_value_loss(*args, **kwargs): raise NotImplementedError("Import failed")
    def compute_entropy_loss(*args, **kwargs): raise NotImplementedError("Import failed")
    def compute_gae_advantages(*args, **kwargs): raise NotImplementedError("Import failed")


# ==============================================================================
# == Test Fixtures (Optional, can define common data here)
# ==============================================================================

# Example fixture if needed later
# @pytest.fixture
# def sample_tensor():
#     return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ==============================================================================
# == 1. Tests for Helper Functions
# ==============================================================================

class TestMaskedOps:
    def test_masked_mean_no_mask(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        result = masked_mean(tensor, None)
        torch.testing.assert_close(result, torch.tensor(2.5))

    def test_masked_mean_with_mask(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_mean(tensor, mask) # Mean of 1, 3, 4 = 8 / 3
        torch.testing.assert_close(result, torch.tensor(8.0 / 3.0))

    def test_masked_mean_dim(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        mask = torch.tensor([[True, False], [True, True]])
        result_dim0 = masked_mean(tensor, mask, dim=0) # Mean([1, 3]), Mean([nan, 4]) -> [2, 4]
        result_dim1 = masked_mean(tensor, mask, dim=1) # Mean([1, nan]), Mean([3, 4]) -> [1, 3.5]
        torch.testing.assert_close(result_dim0, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(result_dim1, torch.tensor([1.0, 3.5]))

    def test_masked_mean_all_false_mask(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        result = masked_mean(tensor, mask) # Should be 0 due to epsilon
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_masked_whiten_basic(self):
        tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        # Masked elements: 1, 2, 4, 5, 6. Mean = 3.6. Var = E[X^2] - (E[X])^2
        # E[X^2] = (1+4+16+25+36)/5 = 82/5 = 16.4. Var = 16.4 - 3.6^2 = 16.4 - 12.96 = 3.44. Std = sqrt(3.44)
        std_expected = np.sqrt(3.44)
        mean_expected = 3.6
        expected = (tensor - mean_expected) / std_expected
        expected[0, 2] = 0.0 # Zero out non-masked element
        result = masked_whiten(tensor, mask, shift_mean=True)
        torch.testing.assert_close(result, expected)

    def test_masked_whiten_no_shift(self):
        tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        std_expected = np.sqrt(3.44)
        expected = tensor / std_expected
        expected[0, 2] = 0.0 # Zero out non-masked element
        result = masked_whiten(tensor, mask, shift_mean=False)
        torch.testing.assert_close(result, expected)

    def test_masked_whiten_all_false_mask(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        result = masked_whiten(tensor, mask)
        torch.testing.assert_close(result, torch.zeros_like(tensor)) # Should be all zeros

class TestRewardUtils:
    @pytest.mark.parametrize("text, expected", [
        ("The answer is #### 123", "123"),
        ("Blah blah ####    45.67 end", "45.67"),
        ("Final result: #### -78,910.5", "-78910.5"),
        ("Mixed #### 1,234,567", "1234567"),
        ("No hash 123", "123"), # Fallback
        ("Multiple numbers 10 then 20", "20"), # Fallback last number
        ("No number here", None),
        ("Last thing is text 50 end", "50"),
        ("Invalid last number #### 12a", None), # Strict format fails
        ("Invalid last number fallback 12a", None), # Fallback fails
        ("Negative fallback -50", "-50"),
    ])
    def test_extract_gsm8k_solution(self, text, expected):
        assert extract_gsm8k_solution(text) == expected

    @pytest.mark.parametrize("gen_text, truth_str, expected_reward", [
        ("Some text #### 100", "100", 1.0),
        ("Some text #### 100.0", "100", 1.0),
        ("Some text #### 100", "100.0", 1.0),
        ("Result: #### 99.9", "100", 0.0),
        ("No answer here", "100", 0.0),
        ("Fallback answer 50", "50", 1.0),
        ("Fallback answer 50", "51", 0.0),
        ("Text #### 1,200", "1200", 1.0),
        ("Text #### -5.5", "-5.5", 1.0),
        ## ("Text #### 1e3", "1000", 1.0), Scientific notation not handled
        ("Text #### 1000", "1e3", 1.0),
    ])
    def test_compute_gsm8k_reward(self, gen_text, truth_str, expected_reward):
        assert compute_gsm8k_reward(gen_text, truth_str) == expected_reward

class TestCollation:
    def test_pad_and_collate_tensors(self):
        t1 = torch.ones((2, 3))
        t2 = torch.ones((1, 5)) * 2
        t3 = torch.ones((3, 4)) * 3
        tensor_list = [t1, t2, t3]
        padding_value = -1.0
        result = pad_and_collate_tensors(tensor_list, padding_value=padding_value)

        assert result.shape == (6, 5) # Total batch = 2+1+3=6, max len = 5
        # Check padding value in t1
        assert torch.allclose(result[0:2, 3:], torch.tensor(-1.0))
        # Check padding value in t3
        assert torch.allclose(result[3:6, 4:], torch.tensor(-1.0))
        # Check original values
        torch.testing.assert_close(result[0:2, :3], t1)
        torch.testing.assert_close(result[2:3, :5], t2)
        torch.testing.assert_close(result[3:6, :4], t3)

    def test_pad_and_collate_zero_len_seq(self):
        t1 = torch.ones((2, 0, 3))
        t2 = torch.ones((1, 0, 3))
        result = pad_and_collate_tensors([t1, t2])
        assert result.shape == (3, 0, 3)

# ==============================================================================
# == 2. Tests for Core PPO Algorithm Components
# ==============================================================================

class TestPPOAlgos:
    # --- Setup common data for PPO tests ---
    batch_size = 2
    resp_len = 3
    vocab_size = 10
    clip_ratio = 0.2
    clip_range_value = 0.2
    gamma = 0.99
    lam = 0.95

    # Create some dummy tensors (use float for logprobs/values)
    log_probs_new = torch.randn(batch_size, resp_len, requires_grad=True)
    log_probs_old = torch.randn(batch_size, resp_len)
    advantages = torch.randn(batch_size, resp_len)
    response_mask = torch.ones(batch_size, resp_len, dtype=torch.long) # Assume no padding for simplicity first
    response_mask_padded = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long) # Example padding
    values_new = torch.randn(batch_size, resp_len, requires_grad=True)
    values_old = torch.randn(batch_size, resp_len)
    returns = torch.randn(batch_size, resp_len)
    logits_new = torch.randn(batch_size, resp_len, vocab_size, requires_grad=True)
    final_rewards = torch.tensor([1.0, -0.5]) # Shape (batch_size,)
    kl_penalties = torch.rand(batch_size, resp_len) * 0.1 # Small KL penalties
    values_gae = torch.randn(batch_size, resp_len) # Values for GAE


    def test_compute_policy_loss_shapes(self):
        loss, clip_frac, approx_kl = compute_policy_loss(
            self.log_probs_new, self.log_probs_old, self.advantages,
            self.response_mask, self.clip_ratio
        )
        assert loss.shape == ()
        assert clip_frac.shape == ()
        assert approx_kl.shape == ()
        assert loss.requires_grad # Should have gradient w.r.t. log_probs_new

    def test_compute_policy_loss_positive_adv(self):
        # If advantages are positive, loss should be negative (encouraging increase in logprob)
        pos_advantages = torch.abs(self.advantages)
        loss, _, _ = compute_policy_loss(
            self.log_probs_new, self.log_probs_old, pos_advantages,
            self.response_mask, self.clip_ratio
        )
        assert loss.item() <= 0 # Policy loss is negative of mean surrogate

    def test_compute_policy_loss_negative_adv(self):
        # If advantages are negative, loss should be positive (discouraging increase in logprob)
        neg_advantages = -torch.abs(self.advantages)
        loss, _, _ = compute_policy_loss(
            self.log_probs_new, self.log_probs_old, neg_advantages,
            self.response_mask, self.clip_ratio
        )
        assert loss.item() >= 0

    def test_compute_policy_loss_masking(self):
         # Compare loss with full mask vs padded mask
        loss_full, _, _ = compute_policy_loss(
            self.log_probs_new, self.log_probs_old, self.advantages,
            self.response_mask, self.clip_ratio
        )
        loss_padded, _, _ = compute_policy_loss(
            self.log_probs_new, self.log_probs_old, self.advantages,
            self.response_mask_padded, self.clip_ratio
        )
        # Loss with padding mask should generally differ if advantages/logprobs are non-zero
        # (unless masked out elements happened to be zero)
        # This is a weak test, but checks if mask is used
        if not torch.allclose(self.advantages * (1-self.response_mask_padded.float()), torch.tensor(0.0)):
             assert not torch.allclose(loss_full, loss_padded)


    def test_compute_value_loss_shapes(self):
        loss, clip_frac = compute_value_loss(
            self.values_new, self.values_old, self.returns,
            self.response_mask, self.clip_range_value
        )
        assert loss.shape == ()
        assert clip_frac.shape == ()
        assert loss.requires_grad # Should have gradient w.r.t. values_new
        assert loss.item() >= 0 # Value loss based on MSE should be non-negative

    def test_compute_value_loss_clipping(self):
        # Case 1: values_new is close to values_old (no clipping)
        values_new_close = self.values_old + self.clip_range_value * 0.5
        loss_unclipped, clip_frac_unclipped = compute_value_loss(
            values_new_close, self.values_old, self.returns,
            self.response_mask, self.clip_range_value
        )
        # Case 2: values_new is far from values_old (clipping occurs)
        values_new_far = self.values_old + self.clip_range_value * 2.0
        loss_clipped, clip_frac_clipped = compute_value_loss(
            values_new_far, self.values_old, self.returns,
            self.response_mask, self.clip_range_value
        )
        # Check clip fractions
        assert clip_frac_unclipped.item() == 0.0
        assert clip_frac_clipped.item() > 0.0 # Some clipping should happen

        # Check loss calculation (vf_loss1 vs vf_loss2)
        # Manually calculate for clipped case
        values_pred_clipped = self.values_old + self.clip_range_value # Clipped value
        vf_loss1 = (values_new_far - self.returns)**2
        vf_loss2 = (values_pred_clipped - self.returns)**2
        expected_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), self.response_mask)
        torch.testing.assert_close(loss_clipped, expected_loss)

    def test_compute_value_loss_masking(self):
        loss_full, _ = compute_value_loss(
            self.values_new, self.values_old, self.returns,
            self.response_mask, self.clip_range_value
        )
        loss_padded, _ = compute_value_loss(
            self.values_new, self.values_old, self.returns,
            self.response_mask_padded, self.clip_range_value
        )
        # Differs if returns/values non-zero in masked region
        if not torch.allclose(self.returns * (1-self.response_mask_padded.float()), torch.tensor(0.0)):
            assert not torch.allclose(loss_full, loss_padded)


    def test_compute_entropy_loss_shape_sign(self):
        loss = compute_entropy_loss(self.logits_new, self.response_mask)
        assert loss.shape == ()
        assert loss.requires_grad
        assert loss.item() <= 0 # Entropy loss is negative mean entropy

    def test_compute_entropy_loss_masking(self):
        loss_full = compute_entropy_loss(self.logits_new, self.response_mask)
        loss_padded = compute_entropy_loss(self.logits_new, self.response_mask_padded)
        # Entropy depends only on logits, so masking affects the averaging
        # If logits in the masked region are non-uniform, losses should differ
        assert not torch.allclose(loss_full, loss_padded)

    def test_compute_entropy_loss_uniform_vs_peaked(self):
        # Uniform logits -> high entropy -> large negative loss
        logits_uniform = torch.zeros_like(self.logits_new)
        loss_uniform = compute_entropy_loss(logits_uniform, self.response_mask)

        # Peaked logits -> low entropy -> small negative loss
        logits_peaked = torch.zeros_like(self.logits_new)
        logits_peaked[:, :, 0] = 10.0 # Make one logit large
        loss_peaked = compute_entropy_loss(logits_peaked, self.response_mask)

        assert loss_uniform.item() < loss_peaked.item() # More negative for uniform


    def test_compute_gae_advantages_shapes(self):
        advantages, returns = compute_gae_advantages(
            self.final_rewards, self.kl_penalties, self.values_gae,
            self.response_mask, self.gamma, self.lam
        )
        assert advantages.shape == (self.batch_size, self.resp_len)
        assert returns.shape == (self.batch_size, self.resp_len)

    def test_compute_gae_advantages_simple_case(self):
        # Manual calculation for a simple case B=1, L=2
        final_rewards_s = torch.tensor([10.0]) # R_T (received after s_1, a_1 -> s_2)
        kl_penalties_s = torch.tensor([[0.1, 0.2]]) # KL(a_0), KL(a_1)
        values_s = torch.tensor([[1.0, 2.0]]) # V(s_0), V(s_1)
        response_mask_s = torch.tensor([[1, 1]], dtype=torch.long)
        gamma = 0.9
        lam = 0.9

        # Rewards r_t = -kl_t, except r_T-1 = R_T - kl_{T-1}
        # r_0 = -0.1
        # r_1 = 10.0 - 0.2 = 9.8
        token_rewards = torch.tensor([[-0.1, 9.8]])

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1} * mask_{t+1}
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        # t = 1 (last step):
        # V(s_2) = 0 (end of episode)
        # delta_1 = r_1 + gamma * V(s_2) - V(s_1) = 9.8 + 0.9 * 0 - 2.0 = 7.8
        # A_1 = delta_1 + gamma * lambda * A_2 * mask_2 = 7.8 + 0 (A_2=0) = 7.8
        adv_rev = [7.8]
        last_gae = 7.8

        # t = 0:
        # V(s_1) = 2.0
        # delta_0 = r_0 + gamma * V(s_1) - V(s_0) = -0.1 + 0.9 * 2.0 - 1.0 = -0.1 + 1.8 - 1.0 = 0.7
        # A_0 = delta_0 + gamma * lambda * A_1 * mask_1 = 0.7 + 0.9 * 0.9 * 7.8 * 1 = 0.7 + 0.81 * 7.8 = 0.7 + 6.318 = 7.018
        adv_rev.append(7.018)

        advantages_expected = torch.tensor([[7.018, 7.8]])
        # Returns = Advantages + Values
        returns_expected = advantages_expected + values_s

        # Whiten advantages (mean=7.409, std=sqrt(((7.018-7.409)^2 + (7.8-7.409)^2)/2)) )
        # std = sqrt(((-0.391)^2 + (0.391)^2)/2) = sqrt(0.152881) = 0.391
        mean_adv = 7.409
        std_adv = 0.391
        advantages_whitened_expected = (advantages_expected - mean_adv) / std_adv
        # adv_w = [(7.018-7.409)/0.391, (7.8-7.409)/0.391] = [-1, 1] approx
        advantages_whitened_expected = torch.tensor([[-1.0, 1.0]])


        advantages_calc, returns_calc = compute_gae_advantages(
            final_rewards_s, kl_penalties_s, values_s, response_mask_s, gamma, lam
        )

        torch.testing.assert_close(advantages_calc, advantages_whitened_expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(returns_calc, returns_expected, rtol=1e-3, atol=1e-3)

def test_compute_gae_advantages_masking():
    """
    Tests GAE calculation when the response mask indicates the sequence
    ended early (e.g., only the first step is valid).
    """
    # --- Test Setup ---
    # Mask indicates only step t=0 is valid in a sequence of length 2
    response_mask_s_padded = torch.tensor([[1, 0]], dtype=torch.long)
    # Final reward received after the valid sequence (after step t=0)
    final_rewards_s = torch.tensor([10.0])
    # Example KL penalties for t=0 and t=1
    kl_penalties_s = torch.tensor([[0.1, 0.2]])
    # Value estimates V(s0)=1.0, V(s1)=0.0 (V(s1) should be ignored in propagation)
    # Values for masked portion cannot be non-zero
    values_s = torch.tensor([[1.0, 2.0]])
    gamma = 0.9
    lam = 0.9

    # --- Expected Calculation (Manual Trace) ---
    # Token Rewards: r0 = (10.0 - 0.1) = 9.9; r1 = 0.0 
    # GAE Loop (t=1): delta1 = r1*m1 + g*V(s2)*m2 - V(s1)*m1 = 0.0
    #                 A1 = delta1 + g*l*A2*m2 = 0.0 + 0 = 0.0 
    # GAE Loop (t=0): delta0 = r0 * m0 + g*V(s1)*m1 - V(s0) = 9.9 + 0.9*2.0*0 - 1.0 = 8.9
    # A0 = delta0 + g*l*A1*mask_1 = 8.9 + g*l*(-2.2)*0 = 8.9
    # Advantages (unwhitened): [[8.9, 0.0]]
    # Returns = Advantages + Values: [[8.9+1.0, 0.0]] = [[9.9, 0.0]]
    # Whiten Advantages: Mean(masked=8.9)=8.9. Var=0. Std=eps. Whitened=[[(8.9-8.9)/eps, (-2.2-8.9)/eps]] -> [[0.0, large_neg]]. Masked=[[0.0, 0.0]]

    advantages_whitened_expected = torch.tensor([[0.0, 0.0]])
    ## r_1 will never actually be used and can be ignored
    returns_expected = torch.tensor([[9.9, 0.0]]) # Unmasked returns are the target

    # --- Run Function ---
    advantages_calc, returns_calc = compute_gae_advantages(
        final_rewards_s, kl_penalties_s, values_s, response_mask_s_padded, gamma, lam
    )

    # --- Assertions ---
    assert advantages_calc.shape == (1, 2), "Advantage shape mismatch"
    assert returns_calc.shape == (1, 2), "Return shape mismatch"
    torch.testing.assert_close(advantages_calc, advantages_whitened_expected, rtol=1e-3, atol=1e-3, msg="Whitened Advantages mismatch")
    torch.testing.assert_close(returns_calc, returns_expected, rtol=1e-3, atol=1e-3, msg="Returns mismatch")
