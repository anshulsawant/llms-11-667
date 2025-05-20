# PPO Reinforcement Learning Tutorial for LLMs

This project provides a hands-on tutorial for understanding and implementing the Proximal Policy Optimization (PPO) algorithm to fine-tune Large Language Models (LLMs) using Reinforcement Learning (RL). It is inspired by the logic found in the TinyZero repository but significantly simplified for pedagogical purposes, focusing on core concepts within a standard Python project structure.

The primary goal is to train an LLM (e.g., Qwen 1.8B or a smaller debug model) on the GSM8K (Grade School Math) dataset, improving its ability to solve math word problems accurately using RL.

This repository includes:
- A version of the trainer (`src/ppo_trainer.py`) with five core PPO logic sections left as exercises for the user to implement.
- A solution file (`src/ppo_trainer_solutions.py`) with the implementations filled in.
- Configuration files (`configs/`) for different setups (GPU vs CPU debug).
- Unit tests (`tests/`) designed to verify your PPO logic implementations in `src/ppo_trainer.py`.

## The Big Picture: How PPO RLHF Works Here

The PPO algorithm iterates through a cycle of experience gathering and policy improvement using several components:

**A. Rollout Phase (`perform_rollouts`):**
   - The current **Actor** model (the LLM being trained, e.g., Qwen 1.8B) generates responses (sequences of tokens) based on input prompts. Generating a token is the Actor's "action".
   - The probability distribution over the next possible token output by the LLM's language model head represents the **policy**.
   - During generation, we store crucial data:
     - Input prompts and generated response tokens.
     - `logprobs`: Log probabilities of the generated tokens under the current Actor policy.
     - `values`: The predicted "value" (expected future reward) for each token state, estimated by the **Critic** (the value head attached to the base LLM, taking hidden states as input).
     - `ref_logprobs`: Log probabilities of the generated tokens under the frozen **Reference** policy (an identical, frozen copy of the initial Actor model).
     - `rewards`: The final score (e.g., 1.0 for correct GSM8K answer) for the complete generated sequence.

**B. Advantage Calculation Phase (within `run_ppo_update_epoch`'s caller):**
   - The collected rollout data is processed.
   - KL penalties (`logprobs - ref_logprobs`) are calculated to measure how much the Actor's policy has diverged from the Reference policy.
   - `compute_gae_advantages` uses the task rewards, KL penalties, and the Critic's `values` (`V(s)`) to estimate:
     - `advantages` (`A(s,a)`): How much better or worse were the generated tokens (actions) than what the Critic expected for those states? This signal incorporates both the task reward and the KL penalty.
     - `returns`: What was the actual observed discounted reward-to-go? This serves as the learning target for the Critic (value head).

**C. Update Phase (`perform_ppo_updates` calls `run_ppo_update_epoch`):**
   - This phase uses the rollout data and the calculated advantages/returns to update the Actor and Critic models.
   - It loops for multiple `ppo_epochs` over the *same* batch of rollout data (improving sample efficiency).
   - `run_ppo_update_epoch` iterates over mini-batches:
     - It re-evaluates the generated sequences with the *current* Actor model to get `logprobs_new` (from LM head) and `values_new` (from value head).
     - It calculates the PPO losses by calling your implemented functions:
       - `compute_policy_loss`: Uses the ratio of new/old probabilities and advantages to update the parameters of the **Actor** (the base LLM and its LM head), encouraging actions with positive advantages while clipping updates to maintain stability.
       - `compute_value_loss`: Uses the difference between the Critic's new predictions (`values_new`) and the calculated target `returns` to update the **Critic** (the value head and potentially shared base LLM layers) to become a better predictor of future rewards.
       - `compute_entropy_loss`: Encourages exploration by slightly penalizing the **Actor** for being too certain about its next token prediction.
     - These losses are combined.
     - Gradient descent (`optimizer.step()`) updates the trainable parameters (Actor base + LM head, Value head) based on the combined loss.

**D. Repeat:**
   - The entire cycle (Rollout -> GAE -> Update) repeats, using the newly updated Actor model (LLM) to generate the next batch of rollouts, gradually improving its ability to generate high-reward sequences (correct GSM8K answers) while adhering to the KL constraint.

## Exercise Order

The file `src/ppo_trainer.py` contains the full script structure, but five key sections of the PPO algorithm logic are left blank for you to implement. These are marked with `<<<< YOUR ... IMPLEMENTATION HERE >>>>`. The file `src/ppo_trainer_solutions.py` contains the complete implementation for reference and testing.

We recommend tackling the exercises in `src/ppo_trainer.py` in the following order, as this builds understanding progressively:

1.  **`compute_policy_loss` (Exercise 1):** Implement the PPO clipped surrogate objective. This is central to how the policy (Actor) learns.
2.  **`compute_value_loss` (Exercise 2):** Implement the clipped value function loss. This trains the Critic (value head) to better estimate future rewards.
3.  **`compute_entropy_loss` (Exercise 3):** Implement the entropy calculation for the policy's action distribution. This loss encourages exploration.
4.  **`compute_gae_advantages` (Exercise 4):** Implement Generalized Advantage Estimation (GAE) to calculate advantages and returns. These are crucial inputs for the policy and value loss functions.
5.  **PPO Mini-batch Update Logic (within `run_ppo_update_epoch` - Exercise 5):** Implement the core mini-batch update logic. This involves:
    *   Performing a forward pass with the current model.
    *   Calculating new log probabilities and values for the response tokens.
    *   Calling your previously implemented loss functions (`compute_policy_loss`, `compute_value_loss`, `compute_entropy_loss`).
    *   Combining these losses.
    *   Performing the backward pass and optimizer step.

Unit tests are provided in `tests/test_ppo_trainer.py` to help you verify your implementations for each of these exercises.

## Project Structure
```text
tinier-zero/
├── configs/
│   ├── config.yaml         # Main config (e.g., Qwen 1.8B on GPU)
│   └── config_debug.yaml   # Debug config (e.g., tiny-lm-chat on CPU)
├── src/
│   ├── __init__.py
│   ├── ppo_trainer.py      # Main script logic WITH EXERCISE PLACEHOLDERS
│   └── ppo_trainer_solutions.py # Main script logic WITH SOLUTIONS
├── tests/
│   ├── __init__.py
│   └── test_ppo_trainer.py # Pytest tests for exercise functions
├── requirements.txt        # Python dependencies
├── setup.py                # Setup script for installation
├── README.md               # This file
└── faq.md                  # Frequently Asked Questions (from previous context)
```

## Setup

1.  **Prerequisites:**
    * Python >= 3.9
    * PyTorch >= 2.0
    * Optionally: NVIDIA GPU with CUDA support (required for larger models and 8-bit optimizations). Check CUDA compatibility for `bitsandbytes` if using 8-bit Adam.

2.  **Clone the Repository (If Applicable):**
    ```bash
    git clone https://github.com/anshulsawant/llms-11-667.git
    cd llms-11-667/tinier-zero
    ```

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv trz
    # Activate the environment
    # Linux/macOS:
    source trz/bin/activate
    # Windows (Git Bash/PowerShell):
    # trz\Scripts\activate
    ```

4.  **Install Dependencies:**
    You can install dependencies using either `pip` with `requirements.txt` or using the `setup.py` script in editable mode (which also uses `requirements.txt`).

    * **Using pip:**
        ```bash
        pip install -r requirements.txt
        ```
    * **Using setup.py (recommended for development):** Installs the package in editable mode (`-e`) and includes development dependencies like `pytest`.
        ```bash
        pip install -e .[dev]
        ```
        *(Note: If you don't need `bitsandbytes` for 8-bit Adam, you can remove it from `requirements.txt` before installing, though it's generally handled gracefully if CUDA is unavailable).*

5.  **Hugging Face Authentication (Optional but Recommended):**
    If you plan to use gated models or push models to the Hugging Face Hub:
    ```bash
    huggingface-cli login
    ```

6.  **Weights & Biases Login (Optional):**
    If you want to log metrics to Weights & Biases (wandb.ai):
    ```bash
    wandb login
    ```
    Training will proceed without wandb login if `wandb.report_to_wandb` is `false` in the config.

## Configuration

* Configuration files are located in the `configs/` directory and use YAML format. They are parsed using OmegaConf.
* `config.yaml`: Configured for training a larger model (like Qwen 1.8B) on a GPU, potentially using 8-bit Adam.
* `config_debug.yaml`: Configured for quick debugging runs using a tiny model (`EventsRLF/tiny-gpt-fast-tokenizer`) on the CPU. It inherits defaults from `config.yaml` and overrides key parameters.
* **Key Parameters:** You might want to adjust parameters in the YAML files, such as:
    * `model.name`, `model.tokenizer_name`
    * `model.torch_dtype` (`bfloat16`, `float16`, `float32`, `auto`)
    * `ppo.batch_size`, `ppo.mini_batch_size`, `ppo.gradient_accumulation_steps` (adjust based on VRAM)
    * `ppo.learning_rate`
    * `ppo.use_8bit_adam` (enable/disable 8-bit Adam, requires CUDA & `bitsandbytes`)
    * `training.device` (`cuda` or `cpu`)
    * `training.total_ppo_steps`
    * `training.output_dir`
* **Command-Line Overrides:** You can override any configuration parameter from the command line using the format `key=value`. Nested keys are accessed with dots (e.g., `ppo.learning_rate=5e-7`).

## Usage

1.  **Implement Exercises:**
    Open `src/ppo_trainer.py` and fill in the PPO logic in the five sections marked `<<<< YOUR ... IMPLEMENTATION HERE >>>>`. Follow the "Exercise Order" section above and use the in-code comments as a guide. If you get stuck, you can refer to `src/ppo_trainer_solutions.py` for the complete implementations.

2.  **Run Training:**
    Execute the trainer script from the **project root directory** (`tinier-zero/`), specifying the desired configuration file.
    *   **To run your implementations (after completing exercises in `src/ppo_trainer.py`):**
        *   Debug Run (CPU, Tiny Model):
            ```bash
            python src/ppo_trainer.py --config configs/config_debug.yaml
            ```
        *   GPU Run (Larger Model):
            ```bash
            python src/ppo_trainer.py --config configs/config.yaml training.device=cuda:0 # Specify GPU if needed
            ```

    *   **To run the provided solutions directly:**
        *   Debug Run (CPU, Tiny Model):
            ```bash
            python src/ppo_trainer_solutions.py --config configs/config_debug.yaml
            ```
        *   GPU Run (Larger Model):
            ```bash
            python src/ppo_trainer_solutions.py --config configs/config.yaml training.device=cuda:0
            ```

    *   **Run with Overrides (applies to either script):**
        ```bash
        python src/ppo_trainer.py --config configs/config.yaml ppo.learning_rate=5e-7 training.total_ppo_steps=200
        ```

3.  **Run Unit Tests:**
    Unit tests are crucial for verifying your implementations of the PPO exercises.

    *   **Testing Your Exercise Implementations:**
        After implementing the exercise functions in `src/ppo_trainer.py`, you can verify them using `pytest`. Run from the **project root directory** (`tinier-zero/`):
        ```bash
        pytest
        ```
        This will automatically discover and run the tests in `tests/test_ppo_trainer.py` against your code in `src/ppo_trainer.py`.
        **Note:** These tests are expected to fail until you correctly implement the corresponding exercise sections. Each test function in `tests/test_ppo_trainer.py` is named to clearly indicate which exercise function it targets.

    *   **Testing the Provided Solutions:**
        If you want to verify that the provided solutions in `src/ppo_trainer_solutions.py` pass the tests, or if you've completed your own solution and want to test it by placing it in `ppo_trainer_solutions.py`, you can temporarily modify the import statements in `tests/test_ppo_trainer.py`.
        For example, change lines like:
        ```python
        from src.ppo_trainer import compute_policy_loss, compute_value_loss # ... and other imports
        ```
        to:
        ```python
        from src.ppo_trainer_solutions import compute_policy_loss, compute_value_loss # ... and other imports
        ```
        After making this change, running `pytest` from the project root will execute the tests against the code in `src/ppo_trainer_solutions.py`.
        Remember to revert these import changes in `tests/test_ppo_trainer.py` if you want to go back to testing your own exercise implementations in `src/ppo_trainer.py`.

## Outputs

* Checkpoints (model weights and tokenizer files) are saved periodically during training in subdirectories within the `training.output_dir` specified in your configuration (e.g., `outputs/ppo_gsm8k_qwen1.8b/step_10/`).
* The final trained model is saved in a `final/` subdirectory within the `training.output_dir`.
* The effective configuration used for the run (including merges and overrides) is saved as `effective_config.yaml` in the `training.output_dir`.

We hope this tutorial helps you understand PPO better. Happy learning!
