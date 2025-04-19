# Supervised Fine-Tuning (SFT) Project: Full vs. LoRA

This project performs supervised fine-tuning (SFT) of a large language model using Hugging Face Transformers, Datasets, and Accelerate. It supports both **Full Parameter Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**. The initial configuration targets fine-tuning the `google/gemma-2-9b-it` model on the `gsm8k` dataset.

## Project Structure

```text
your_project_root/
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
git clone <your-repository-url>
cd your_project_root
```
**b. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
**c. Install Requirements**
```bash
pip install -r requirements.txt
```
**d. Install Project Package**
```bash
pip install -e .
```
**e. Hugging Face Authentication** (`huggingface-cli login`)

**f. Weights & Biases Authentication** (`wandb login`)

**g. Hardware Requirements** (Full SFT needs high VRAM, LoRA is more efficient)

## 2. Configuration (`config.yaml` + Overrides)

Configuration is managed via YAML files using `OmegaConf` merging.

* **`config.yaml`**: Contains the base configuration (default model, dataset, training parameters, LoRA settings, evaluation settings, **default inference settings**).
* **Override Configs (e.g., `config_full_main.yaml`):** Smaller files specifying only the parameters that differ for a specific experiment run (like `tuning_method`, `dataset.config_name`, `training.output_dir`, `wandb.run_name`). These merge with `config.yaml`.
* **`inference` Section:** The base `config.yaml` contains an `inference` section defining default parameters used by `inference.py`:
    * `input_file`: Default path to the input data.
    * `output_dir`: Default directory for saving inference results.
    * `prompt_field`: Default field name for questions in the input file.
    * `output_field`: Default field name for generations in the output file.
    * `precision`, `batch_size`, `max_new_tokens`, `temperature`, `do_sample`: Default generation parameters.
    * **Note:** These values in the `inference` section can also be overridden in your specific run config files (e.g., `config_full_main.yaml`) if needed for a particular model.

Review `config.yaml` for base settings and create/modify override files for specific runs.

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
accelerate launch src/sft_project/train.py --config config_full_main.yaml

# Example: Run LoRA SFT using settings in config_lora_main.yaml
accelerate launch src/sft_project/train.py --config config_lora_main.yaml

# Specify a different base config if needed:
# accelerate launch src/sft_project/train.py --config <override.yaml> --base_config <base.yaml>
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

# Example: Evaluate LoRA model on the 'socratic' split using 50 random samples
accelerate launch src/sft_project/evaluate.py \
    --config_path config_lora_main.yaml \
    --dataset_config_name socratic \
    --num_eval_samples 50 \
    --eval_random_subset

# Example: Save metrics to a file
accelerate launch src/sft_project/evaluate.py \
    --config_path config_lora_main.yaml \
    --output_file ./results/lora_main_eval_metrics.json

```
**Command-Line Arguments for `evaluate.py`:**
* `--config_path`: (Required) Path to the override config file for the model/run.
* `--base_config_path`: (Optional) Path to base config (default: `config.yaml`).
* `--use_base_model`: (Optional) Flag to force evaluation of the base model.
* `--eval_split`: (Optional) Override evaluation split name (default from config).
* `--dataset_config_name`: (Optional) Override dataset config (e.g., "socratic", default from config).
* `--num_eval_samples`: (Optional) Override number of samples (default from config).
* `--eval_random_subset`: (Optional) Override random sampling flag (default from config).
* `--output_file`: (Optional) Path to save metrics JSON.

## 5. Finding Training & Evaluation Results

* **Training Logs:** Console output and WandB dashboard (if enabled).
* **Checkpoints & Final Model/Adapter:** Saved in the `training.output_dir` specified in the *merged* config used for training (e.g., `./sft_results_full_main/final_checkpoint/`). LoRA saves only adapter weights.
* **Evaluation Metrics:** Printed to console by `evaluate.py`. Can optionally be saved to a JSON file using `--output_file`. Evaluation results from the pre/post base model checks in `train.py` are logged to console/WandB.

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

**c. Find Inference Results**
Saved to a **new file** named `[input_file_stem]__[config_file_stem_OR_base_id]__generations.jsonl` in the directory specified by `inference.output_dir` in the config.
