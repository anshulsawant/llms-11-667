# PPO Reinforcement Learning Tutorial for LLMs

This project provides a hands-on tutorial for understanding and implementing the Proximal Policy Optimization (PPO) algorithm to fine-tune Large Language Models (LLMs) using Reinforcement Learning (RL). It is inspired by the logic found in the TinyZero repository but significantly simplified for pedagogical purposes, focusing on core concepts within a standard Python project structure.

The primary goal is to train an LLM (e.g., Qwen 1.8B or a smaller debug model) on the GSM8K (Grade School Math) dataset, improving its ability to solve math word problems accurately using RL.

This repository includes:
- A version of the trainer (`src/ppo_trainer.py`) with core PPO logic sections left as exercises for the user to implement.
- A solution file (`src/ppo_trainer_solutions.py`) with the implementations filled in.
- Configuration files (`configs/`) for different setups (GPU vs CPU debug).
- Unit tests (`tests/`) to verify the PPO logic implementations.

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

**B. Advantage Calculation Phase (within `perform_ppo_update`):**
   - The collected rollout data is processed.
   - KL penalties (`logprobs - ref_logprobs`) are calculated to measure how much the Actor's policy has diverged from the Reference policy.
   - `compute_gae_advantages` uses the task rewards, KL penalties, and the Critic's `values` (`V(s)`) to estimate:
     - `advantages` (`A(s,a)`): How much better or worse were the generated tokens (actions) than what the Critic expected for those states? This signal incorporates both the task reward and the KL penalty.
     - `returns`: What was the actual observed discounted reward-to-go? This serves as the learning target for the Critic (value head).

**C. Update Phase (`perform_ppo_update`):**
   - This phase uses the rollout data and the calculated advantages/returns to update the Actor and Critic models.
   - It loops for multiple `ppo_epochs` over the *same* batch of rollout data (improving sample efficiency).
   - Within each epoch, it iterates over mini-batches:
     - It re-evaluates the generated sequences with the *current* Actor model to get `logprobs_new` (from LM head) and `values_new` (from value head).
     - It calculates the PPO losses:
       - `compute_policy_loss`: Uses the ratio of new/old probabilities and advantages to update the parameters of the **Actor** (the base LLM and its LM head), encouraging actions with positive advantages while clipping updates to maintain stability.
       - `compute_value_loss`: Uses the difference between the Critic's new predictions (`values_new`) and the calculated target `returns` to update the **Critic** (the value head and potentially shared base LLM layers) to become a better predictor of future rewards.
       - `compute_entropy_loss`: Encourages exploration by slightly penalizing the **Actor** for being too certain about its next token prediction.
     - These losses are combined.
     - Gradient descent (`optimizer.step()`) updates the trainable parameters (Actor base + LM head, Value head) based on the combined loss.

**D. Repeat:**
   - The entire cycle (Rollout -> GAE -> Update) repeats, using the newly updated Actor model (LLM) to generate the next batch of rollouts, gradually improving its ability to generate high-reward sequences (correct GSM8K answers) while adhering to the KL constraint.
   
## Exercise Order

The file `src/ppo_trainer.py` contains the full script structure, but the core PPO algorithm logic is left blank for you to implement as an exercise. The file `src/ppo_trainer_solutions.py` contains the complete implementation for reference.

Here's a suggested order for tackling the exercises in `src/ppo_trainer.py`:

1.  **`compute_policy_loss`**: Implement the PPO clipped surrogate objective. This is central to how the policy learns.
2.  **`compute_value_loss`**: Implement the clipped value function loss. This trains the critic baseline.
3.  **`compute_entropy_loss`**: Implement the entropy calculation to encourage exploration.
4.  **`compute_gae_advantages`**: Implement Generalized Advantage Estimation to calculate advantages and returns, which are inputs to the loss functions.
5.  **`perform_ppo_update` (Update Loop Logic)**: Implement the main PPO update loop, bringing together GAE and the loss functions to perform gradient updates over multiple epochs and mini-batches.

Unit tests are provided in `tests/test_ppo_logic.py` to help verify your implementations.

## Project Structure
```text
tinier-zero/
├── configs/
│   ├── config.yaml         # Main config (e.g., Qwen 1.8B on GPU)
│   └── config_debug.yaml   # Debug config (e.g., tiny-lm-chat on CPU)
├── src/
│   ├── init.py
│   ├── ppo_trainer.py      # Main script logic WITH EXERCISE PLACEHOLDERS
│   └── ppo_trainer_solutions.py # Main script logic WITH SOLUTIONS
├── tests/
│   ├── init.py
│   └── test_ppo_logic.py   # Pytest tests for exercise functions
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
        *(Note: If you don't need `bitsandbytes` for 8-bit Adam, you can remove it from `requirements.txt` before installing).*
6. Hugging Face Authentication 
```bash
huggingface-cli login
```
5. Login to wandb
   ```bash
   wandb login
   ```
## Configuration

* Configuration files are located in the `configs/` directory and use YAML format. They are parsed using OmegaConf.
* `config.yaml`: Configured for training a larger model (like Qwen 1.8B) on a GPU, potentially using 8-bit Adam.
* `config_debug.yaml`: Configured for quick debugging runs using a tiny model (`sbintuitions/tiny-lm-chat`) on the CPU. It inherits defaults from `config.yaml` and overrides key parameters.
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

1.  **Implement Exercises (Optional):** Open `src/ppo_trainer.py` and fill in the PPO logic in the sections marked `<<<< YOUR ... IMPLEMENTATION HERE >>>>`. Use the comments and the recommended order as a guide. You can refer to `src/ppo_trainer_solutions.py` if you get stuck.

2.  **Run Training:** Execute the trainer script from the **project root directory** (`ppo_rl_tutorial/`), specifying the desired configuration file.

    * **Debug Run (CPU, Tiny Model):**
        ```bash
        python src/ppo_trainer_solutions.py --config-name config_debug.yaml
        # Or use src/ppo_trainer.py if you've filled in the exercises
        ```

    * **GPU Run (Larger Model):**
        ```bash
        python src/ppo_trainer_solutions.py --config-name config.yaml training.device=cuda:0 # Specify GPU if needed
        ```

    * **Run with Overrides:**
        ```bash
        python src/ppo_trainer_solutions.py --config-name config.yaml ppo.learning_rate=5e-7 training.total_ppo_steps=200
        ```

3.  **Run Unit Tests:** After implementing the exercise functions in `src/ppo_trainer.py`, you can verify them using `pytest`. Run from the **project root directory**:
    ```bash
    pytest
    ```
    This will automatically discover and run the tests in `tests/test_ppo_logic.py`.

## Outputs

* Checkpoints (model weights and tokenizer files) are saved periodically during training in subdirectories within the `training.output_dir` specified in your configuration (e.g., `outputs/ppo_gsm8k_qwen1.8b/step_10/`).
* The final trained model is saved in a `final/` subdirectory within the `training.output_dir`.
* The effective configuration used for the run (including merges and overrides) is saved as `effective_config.yaml` in the `training.output_dir`.
