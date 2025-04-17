import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)
# Note: Removed peft and bitsandbytes imports
from trl import SFTTrainer

# --- Configuration ---
model_id = "google/gemma-2-3b"  # Base Gemma 2 3B model
# For instruction-tuned version, use "google/gemma-2-3b-it" and potentially adjust formatting_func

dataset_name = "gsm8k"
dataset_config = "main" # Or "socratic"
new_model_name = "gemma-2-3b-gsm8k-full-sft" # Name for saving the fine-tuned model
output_dir = "./results_full_ft" # Directory to save training results/checkpoints

# --- !!! Resource Warning !!! ---
# Full fine-tuning requires significantly more GPU VRAM than PEFT (LoRA).
# Training a 3B parameter model fully might require high-end GPUs (e.g., A100, H100)
# or multiple GPUs with sufficient memory. Adjust batch size and gradient accumulation
# carefully based on your hardware. Expect memory errors (OOM) if VRAM is insufficient.

# --- Load Model ---
print(f"Loading base model: {model_id}")
# Load the model without quantization. Use mixed precision (bf16 or fp16) for efficiency.
compute_dtype = torch.bfloat16 # Or torch.float16 if bf16 is not supported
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", # Automatically distribute model across available GPUs
    torch_dtype=compute_dtype, # Load in mixed precision
    # attn_implementation="flash_attention_2", # Optional: Use Flash Attention 2 if available/installed
    trust_remote_code=True, # Gemma models might require this
)
model.config.use_cache = False # Disable caching for training
model.config.pretraining_tp = 1 # Set tensor parallelism degree (1 for no parallelism)

# --- Load Tokenizer ---
print(f"Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Use EOS token as padding token
    model.config.pad_token_id = model.config.eos_token_id # Update model config

# --- Load and Prepare Dataset ---
print(f"Loading dataset: {dataset_name} ({dataset_config})")
dataset = load_dataset(dataset_name, name=dataset_config)

# --- Define Formatting Function ---
# This function formats each example into a single string for the model.
# Adjust the format based on how you want the model to learn.
def formatting_func(example):
    # Simple question-answer format
    text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return text

# --- Training Arguments ---
# These arguments are adjusted for full fine-tuning.
# *** CRITICAL: Adjust batch_size and gradient_accumulation based on your VRAM ***
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1, # Start with 1 epoch, increase if needed
    per_device_train_batch_size=1, # VERY SMALL batch size due to high memory usage of full FT
    gradient_accumulation_steps=8, # Increase accumulation to compensate for small batch size (1*8=8 effective batch size)
    optim="adamw_torch", # Standard AdamW optimizer
    save_steps=500, # Save checkpoints every 500 steps
    logging_steps=25, # Log training progress every 25 steps
    learning_rate=1e-5, # Often use a lower LR for full fine-tuning compared to LoRA (e.g., 1e-5 or 5e-6)
    weight_decay=0.01, # Weight decay for regularization
    fp16=False, # Set to True if using torch.float16 and bf16 is False/unsupported
    bf16=True, # Use bf16 mixed precision (requires Ampere+ GPU)
    max_grad_norm=0.3, # Gradient clipping norm
    max_steps=-1, # Set to a positive number for debugging (e.g., 100) to limit training steps
    warmup_ratio=0.03, # Ratio of steps for learning rate warmup
    group_by_length=True, # Group sequences of similar length for efficiency
    lr_scheduler_type="cosine", # Cosine learning rate scheduler is common
    report_to="wandb", # Log to tensorboard (can change to "wandb" etc.)
    # evaluation_strategy="steps", # Uncomment and add eval_dataset to SFTTrainer if you want evaluation
    # eval_steps=200,             # Evaluate every 200 steps
)

# --- Initialize SFTTrainer ---
# SFTTrainer is used here without a peft_config for full fine-tuning.
print("Initializing SFTTrainer for full fine-tuning...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"], # Use the training split
    # eval_dataset=dataset["test"], # Uncomment if you want evaluation during training
    # peft_config is NOT provided for full fine-tuning
    dataset_text_field="text", # We need to tell SFTTrainer to use the formatted text
    formatting_func=formatting_func, # Pass the formatting function
    max_seq_length=512, # Maximum sequence length (adjust based on model and data)
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Set to True if you want to pack multiple short sequences together
)

# --- Start Training ---
print("Starting full fine-tuning...")
# This command begins the fine-tuning process. It will take significant time and GPU resources.
# trainer.train()

# --- Save the Fine-tuned Model ---
# After training, save the full model. This will save the entire model weights.
# print(f"Saving the full fine-tuned model to {new_model_name}...")
# trainer.save_model(new_model_name) # Saves the full model
# tokenizer.save_pretrained(new_model_name) # Save tokenizer alongside model

print("\nScript finished. Uncomment 'trainer.train()' and 'trainer.save_model()' lines to run.")
print("--- WARNING ---")
print("Full fine-tuning is extremely resource-intensive (GPU VRAM).")
print("Monitor memory usage closely and adjust batch size/gradient accumulation as needed.")
print("Training may fail with Out-of-Memory (OOM) errors if resources are insufficient.")
