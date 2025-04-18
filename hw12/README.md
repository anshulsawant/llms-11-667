# Supervised Fine-Tuning (SFT) Project: Full vs. LoRA

This project performs supervised fine-tuning (SFT) of a large language model using Hugging Face Transformers, Datasets, and Accelerate. It supports both **Full Parameter Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**. The initial configuration targets fine-tuning the `google/gemma-2-9b-it` model on the `gsm8k` dataset.

## Project Structure

```text
your_project_root/
├── src/
│   └── sft_project/       # Python package source code
│       ├── init.py
│       ├── sft_script.py  # Main training and evaluation script (supports Full & LoRA)
│       └── inference.py   # Script for running inference (supports Full & LoRA)
├── config.yaml            # Configuration file for model, dataset, training, LoRA
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup script
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

## 2. Configuration (config.yaml)
  The config.yaml file controls the fine-tuning process:

  tuning_method: Set to "full" or "lora" to select the fine-tuning approach.

  model: Base model configuration.

  dataset: Dataset configuration.

  training: General training hyperparameters (batch size, learning rate, epochs, etc.). Note that optimal learning_rate and per_device_train_batch_size might differ between full and lora.

  lora_config: Parameters for LoRA (rank r, lora_alpha, target_modules, etc.). Only used if tuning_method is "lora".

  wandb: Weights & Biases configuration.

  evaluation: Evaluation settings, including base model prompting strategy.

  Review and modify this file to select your tuning method and set appropriate parameters.

## 3. Running Training and Evaluation
  The sft_script.py handles training and evaluation for both Full SFT and LoRA SFT, based on the tuning_method set in config.yaml.

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

Checkpoints & Final Model/Adapter: Saved in training.output_dir (inside subdirectories like final_checkpoint_full or final_checkpoint_lora).

Full SFT: Saves the entire fine-tuned model checkpoint.

LoRA SFT: Saves only the trained adapter weights and configuration (adapter_model.safetensors, adapter_config.json) inside the checkpoint directory. The base model is not saved again.

Metrics: Saved in training.output_dir and logged to WandB.

## 5. Running Inference with Models
  Use the inference.py script, which now supports loading both full models and LoRA adapters.

a. Prepare Input File
  Create a .jsonl file (e.g., input_prompts.jsonl) with prompts:

{"id": 1, "question": "What is the capital of France?"}
{"id": 2, "question": "Solve for x: 2x + 5 = 15"}

b. Run the Inference Script

  Example: Inference with a fully fine-tuned model:

```bash
python src/sft_project/inference.py \
    --model_name_or_path ./sft_results/final_checkpoint_full/ \
    --input_file input_prompts.jsonl \
    --precision bf16 \
    --batch_size 4 \
    --max_new_tokens 256 \
    --temperature 0.1 \
    # Other args...
```
Example: Inference with a LoRA fine-tuned model:
```bash
python src/sft_project/inference.py \
    --model_name_or_path google/gemma-2-9b-it \ # Provide BASE model name/path
    --adapter_path ./sft_results/final_checkpoint_lora/ \ # Provide path to LoRA adapters
    --input_file input_prompts.jsonl \
    --precision bf16 \
    --batch_size 4 \
    --max_new_tokens 256 \
    --temperature 0.1 \
    # Other args...
```
Key Inference Arguments:

--model_name_or_path: (Required) Path/name of the base model (for LoRA) or the full SFT checkpoint path (for full SFT).

--adapter_path: (Optional) Path to trained LoRA adapters. Only use when loading a LoRA model.

--input_file: (Required) Path to your .jsonl input file.

--prompt_field: Field in JSONL containing the question (default: question).

--output_field: Field name for the generated output (default: generation).

--precision: Model precision (bf16, fp16, fp32). Default: bf16.

--batch_size: Inference batch size. Adjust based on GPU memory.

--max_new_tokens: Max tokens to generate.

--temperature: Generation temperature.

--do_sample: Use sampling.

--hf_token: Optional HF token.

--force_cpu: Run on CPU if needed.

c. Find Inference Results
The script overwrites the --input_file with results, adding the generation field. Make backups if needed.

Example input_prompts.jsonl after running inference:
```json
{"id": 1, "question": "What is the capital of France?", "generation": " The capital of France is Paris."}
{"id": 2, "question": "Solve for x: 2x + 5 = 15", "generation": " 2x = 15 - 5\n2x = 10\nx = 10 / 2\nx = 5\n#### 5"}
```
