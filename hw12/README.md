# Supervised Fine-Tuning (SFT) Project: Full vs. LoRA

This project performs supervised fine-tuning (SFT) of a large language model using Hugging Face Transformers, Datasets, and Accelerate. It supports both **Full Parameter Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**. The initial configuration targets fine-tuning the `google/gemma-2b-it` model on the `gsm8k` dataset.

## Model Choice
We initially started with `googel/gemma-2-9b-it`. However, the base model is too good for the task and it gave 86% performance with one shot prompt with explicit instructions. Therefore, we will try `google/gemma-2b-it`.

## Project Structure

```text
llms-11-667/hw12/
├── src/
│   └── sft_project/       # Python package source code
│       ├── __init__.py    # Marks directory as Python package
│       ├── utils.py       # Shared utility functions
│       ├── train.py       # Main training script (Full & LoRA)
│       ├── evaluate.py    # Evaluation script
│       └── inference.py   # Inference script
├── config.yaml            # Base configuration file
├── config_full_main.yaml  # Example override config
├── config_lora_main.yaml  # Example override config
├── config_*.yaml          # Other override configs...
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup script
└── README.md              # This file
```

## 1. Setup Instructions

(Setup instructions remain the same - clone, venv, install requirements, install package, login to HF/WandB)

**a. Clone Repository**
```bash
git clone https://github.com/anshulsawant/llms-11-667.git
cd llms-11-667/hw12 
```
**b. Create Virtual Environment (Python 3.11 Recommended)**
It's recommended to use Python 3.11 for this project.
```bash
# Ensure you have Python 3.11 installed. 
# You might need to use python3.11 instead of python depending on your system.
python3.11 -m venv cmu-11967-hw12 
source cmu-11967-hw12/bin/activate  # On Windows use `cmu-11967-hw12\Scripts\activate`
```
*Alternatively, if you prefer Conda:*
```bash
# bash setup-conda.sh && source ~/.bashrc # If you haven't installed conda
# conda create -n cmu-11967-hw12 python=3.11
# conda activate cmu-11967-hw12
```

**c. Install Requirements**
```bash
pip install -r requirements.txt
```
**d. Install Project Package (Editable Mode)**
```bash
pip install -e .
```
**e. Authenticate Services**
```bash
huggingface-cli login
wandb login
```
*(You will need accounts for Hugging Face and Weights & Biases.)*

**g. Hardware Requirements** 
Full SFT typically requires significant VRAM (e.g., A100 or H100 GPUs). LoRA is more memory-efficient and can often run on consumer-grade GPUs with sufficient VRAM (e.g., RTX 3090/4090). Check model and batch size configurations.

## 2. Configuration

Project behavior is controlled by YAML configuration files.

* **`config.yaml`**: The base configuration file containing default settings for the model, dataset, training, evaluation, inference, LoRA, and WandB.
* **Override Configs (`configs/*.yaml`):** Place experiment-specific configurations in the `configs/` directory (e.g., `configs/lora_main_config.yaml`). These files only need to contain parameters that differ from `config.yaml`. The scripts merge the override config onto the base config.

**Key Configuration Sections:**

* `model`: Base model name, access token, etc.
* `dataset`: Dataset name, config name (e.g., 'main', 'socratic'), splits, sample counts, sequence length, prompt format.
* `training`: Hyperparameters for the `Trainer` (epochs, batch size, learning rate, optimizer, output directory, logging steps, gradient checkpointing, etc.), `tuning_method` ('full' or 'lora'), `report_to`.
* `lora_config`: Parameters for LoRA (r, alpha, target_modules, etc.), used only if `tuning_method: lora`.
* `evaluation`: Parameters for evaluation runs (batch size, generation settings like `max_new_tokens`, `temperature`, base model prompting strategy/example).
* `inference`: Parameters for inference runs (input file path, batch size, generation settings).
* `wandb`: Project name, run name prefix/details for logging.

**Before running any script, ensure you have an appropriate override config file created in the `configs/` directory.**

## 3. Running Training

The `train.py` script handles fine-tuning (Full or LoRA).

**a. Configure Accelerate (One-time Setup)**
```bash
accelerate config
```

**b. Launch the Training Script**
Provide the path to your specific **override** config file.
```bash
# Example: Run Full SFT using settings in config_full_main.yaml
accelerate launch src/sft_project/train.py --config_path config_full_main.yaml

# Example: Run LoRA SFT using settings in config_lora_main.yaml
accelerate launch src/sft_project/train.py --config_path config_lora_main.yaml

# Specify a different base config if needed:
# accelerate launch src/sft_project/train.py --config_path <override.yaml> --base_config <base.yaml>
```
* Selects Full SFT or LoRA based on `tuning_method` in the *merged* config.
* Evaluates base model (pre-training), trains, evaluates base model again (post-training). **Note: Evaluation of the fine-tuned model itself is now done separately using `evaluate.py`.**

## 4. Running Evaluation

The `evaluate.py` script handles evaluation of base or fine-tuned models.

**a. Launch the Evaluation Script**
Provide the path to the **override config file** that corresponds to the model you want to evaluate. Use flags to specify details.
```bash
# Example: Evaluate the BASE model defined in config_lora_main.yaml
# (Uses eval settings from merged config, e.g., one-shot)
accelerate launch src/sft_project/evaluate.py --config_path config_lora_main.yaml --use_base_model

# Example: Evaluate the LoRA model trained using config_lora_main.yaml
# (Loads base model + adapter from checkpoint dir defined in config_lora_main.yaml)
accelerate launch src/sft_project/evaluate.py --config_path config_lora_main.yaml

# Example: Evaluate the Full SFT model trained using config_full_main.yaml
accelerate launch src/sft_project/evaluate.py --config_path config_full_main.yaml
```
* Dataset parameters (`eval_split`, `num_eval_samples`, etc.) are taken directly from the `dataset:` section of the merged configuration.

* Evaluation metrics (JSON file) are saved automatically to the `./eval_results/` directory. The filename includes details from the config, model type, and dataset.
* When evaluating the base model, zero or one-shot prompt with explicit instructions is used.

**Key Command-Line Arguments:**

* `--config_path`: (Required) Path to the override config file defining the model and evaluation settings.

* `--base_config_path`: (Optional) Path to the base config file (default: `config.yaml`).

* `--use_base_model`: (Optional) Add this flag to evaluate the base model specified in the config instead of loading a fine-tuned checkpoint/adapter.

## 5. Finding Training Results

* **Training Logs:** Console output and WandB dashboard (if enabled).
* **Checkpoints & Final Model/Adapter:** Saved in the `training.output_dir` specified in the *merged* config used for training (e.g., `./sft_results_full_main/final_checkpoint/`). LoRA saves only adapter weights.

## 6. Running Inference with Models

Use the `inference.py` script (config-driven).

**a. Prepare Input File** (`.jsonl` format)
Ensure the path matches `inference.input_file` in your config.
```jsonl
{"id": 1, "question": "What is the capital of France?"}
```

**b. Run the Inference Script**
Provide the path to the **override config file** for the desired model.
```bash
# Example: Inference with model trained using config_full_main.yaml
python src/sft_project/inference.py --config_path config_full_main.yaml

# Example: Inference with model trained using config_lora_main.yaml
python src/sft_project/inference.py --config_path config_lora_main.yaml

# Example: Inference with BASE model (using config_full_main for base model info)
python src/sft_project/inference.py --config_path config_full_main.yaml --use_base_model
```
**Command-Line Arguments for `inference.py`:**
* `--config_path`: (Required) Path to the override config file.
* `--base_config_path`: (Optional) Path to base config (default: `config.yaml`).
* `--use_base_model`: (Optional) Flag to force using the base model.

If using the base model, zero or one shot prompt with explicit instruction is used.

**c. Find Inference Results**
Saved to a **new file** named `[input_file_stem]__[config_file_stem_OR_base_id]__generations.jsonl` in the directory specified by `inference.output_dir` in the config.
