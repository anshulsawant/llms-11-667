# Supervised Fine-Tuning (SFT) Project: Full vs. LoRA

This project performs supervised fine-tuning (SFT) of a large language model using Hugging Face Transformers, Datasets, and Accelerate. It supports both **Full Parameter Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**. The initial configuration targets fine-tuning the `google/gemma-2-9b-it` model on the `gsm8k` dataset.

## Project Structure

```text
your_project_root/
├── src/
│   └── sft_project/       # Python package source code
│       ├── __init__.py    # Marks directory as Python package
│       ├── sft_script.py  # Main training and evaluation script (supports Full & LoRA)
│       └── inference.py   # Script for running inference (supports Full & LoRA)
├── config.yaml            # Base configuration file
├── config_full_main.yaml  # Example override config
├── config_lora_main.yaml  # Example override config
├── config_*.yaml          # Other override configs...
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup script
├── data/
│   └── inference_input_sft.jsonl # Inference input file for the fine tuned mode.
│       ├──inference_input_base.jsonl # Inference input file for the pre-trained model (one-shot).
└── README.md              # This file
```

## 1. Setup Instructions

Follow these steps to set up your environment:

a. Clone the Repository
```bash
git clone https://github.com/anshulsawant/llms-11-667.git
cd llms-11-667/hw12
```

b. Create a Virtual Environment
  It's highly recommended to use a virtual environment:
```bash
python -m venv hw12
source hw12/bin/activate
```
c. Install Requirements
  Install the necessary Python libraries, including peft for LoRA:
```bash
pip install -r requirements.txt
```
d. Install Project Package
  Install the project code itself in editable mode (allows changes to take effect without reinstalling):

```bash
pip install -e .
```
e. Hugging Face Authentication (Required for Gemma)
  Log in to access gated models:

```bash
huggingface-cli login
```
  Alternatively, set HF_TOKEN environment variable or add token to config.yaml.

f. Weights & Biases Authentication (Optional)
  Log in to enable experiment tracking:

```bash
wandb login
```
g. Hardware Requirements

  Full Fine-tuning: Requires substantial GPU VRAM (e.g., >= 80GB for Gemma 9B).

  LoRA Fine-tuning: Significantly reduces VRAM requirements, making it feasible on GPUs with less memory (e.g., 24GB or 40GB might be sufficient, depending on config). Adjust batch sizes accordingly.

## 2. Configuration (config.yaml + Overrides)
Configuration is managed via YAML files using OmegaConf merging.

`config.yaml`: Contains the base configuration (default model, dataset, training parameters, LoRA settings, evaluation settings, default inference settings).

Override Configs (e.g., `config_full_main.yaml`): Smaller files specifying only the parameters that differ for a specific experiment run (like `tuning_method`, `dataset.config_name`, `training.output_dir`, `wandb.run_name`). These merge with `config.yaml`.

inference Section: The base `config.yaml` contains an inference section defining default parameters used by `inference.py`:

`input_file`: Default path to the input data.

`output_dir`: Default directory for saving inference results.

`prompt_field`: Default field name for questions in the input file.

`output_field`: Default field name for generations in the output file.

`precision`, `batch_size`, `max_new_tokens`, `temperature`, `do_sample`: Default generation parameters.

Note: These values in the inference section can also be overridden in your specific run config files (e.g., `config_full_main.yaml`) if needed for a particular model.

Review `config.yaml` for base settings and create/modify override files for specific runs.

## 3. Running Training and Evaluation
  The `sft_script.py` handles training and evaluation for both Full SFT and LoRA SFT, based on the tuning_method set in `config.yaml`.

a. Configure Accelerate (One-time Setup)
  If you haven't configured accelerate for your machine before, run the following command and answer the questions about your setup (e.g., number of GPUs, mixed precision):

```bash
accelerate config
```
This saves your configuration for future use.

b. Launch the Training Script
  Use accelerate launch to run the script:

  Ensure you are in the project root directory
```bash
accelerate launch src/sft_project/sft_script.py --config config.yaml
```
The script will automatically apply LoRA if tuning_method: lora is set in the config.

It evaluates the base model first, then trains (either fully or just adapters), then evaluates the fine-tuned result.

## 4. Finding Training Results
  Logs: Console output and WandB dashboard provide detailed metrics.

Checkpoints & Final Model/Adapter: Saved in `training.output_dir` (inside subdirectories like `final_checkpoint_full` or `final_checkpoint_lora`).

Full SFT: Saves the entire fine-tuned model checkpoint.

LoRA SFT: Saves only the trained adapter weights and configuration (`adapter_model.safetensors`, `adapter_config.json`) inside the checkpoint directory. The base model is not saved again.

Metrics: Saved in `training.output_dir` and logged to WandB.

## 5. Running Inference with Models
Use the inference.py script, now primarily driven by configuration files.

a. Prepare Input File
Ensure the input .jsonl file path matches the inference.input_file setting in your merged configuration.

### Example content for data/inference_input.jsonl
{"id": 1, "question": "What is the capital of France?"}
{"id": 2, "question": "Solve for x: 2x + 5 = 15"}

b. Run the Inference Script
Provide the path to the override config file that corresponds to the trained model you want to use. The script infers the model path, input/output files, and generation parameters from the merged config.

### Example: Inference with model trained using config_full_main.yaml
(Script reads input/output paths and params from merged config)
```bash
python src/sft_project/inference.py --config_path config_full_main.yaml
```
### Example: Inference with model trained using config_lora_main.yaml
```bash
python src/sft_project/inference.py --config_path config_lora_main.yaml
```
### Specify a different base config if needed:
```bash
python src/sft_project/inference.py --config_path <override.yaml> --base_config_path <base.yaml>
```

Command-Line Arguments:

`--config_path`: (Required) Path to the override config file for the desired model/run.

`--base_config_path`: (Optional) Path to base config (default: `config.yaml`).

Note: Most other parameters (input/output files, generation settings) are now controlled via the inference section in the configuration files.

c. Find Inference Results
The script saves results to a new file named like `[input_file_stem]__[config_file_stem]__generations.jsonl`.
