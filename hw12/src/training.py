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
from trl import SFTTrainer
from huggingface_hub import login


model_id = "google/gemma-2-9b"

dataset_name = "gsm8k"
dataset_config = "main" # Or "socratic"
new_model_name = "gemma-2-9b-gsm8k-full-sft" # Name for saving the fine-tuned model
output_dir = "./results_full_ft" # Directory to save training results/checkpoints

print(f"Loading base model: {model_id}")
compute_dtype = torch.bfloat16
login()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=compute_dtype,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print(f"Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"Loading dataset: {dataset_name} ({dataset_config})")
dataset = load_dataset(dataset_name, name=dataset_config)

def formatting_func(example):
    # Simple question-answer format
    text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return text

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    save_steps=500,
    logging_steps=25,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

print("Initializing SFTTrainer for full fine-tuning...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    formatting_func=formatting_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Set to True if you want to pack multiple short sequences together
)

# --- Start Training ---
print("Starting full fine-tuning...")
trainer.train()

print(f"Saving the full fine-tuned model to {new_model_name}...")
trainer.save_model(new_model_name) # Saves the full model
tokenizer.save_pretrained(new_model_name) # Save tokenizer alongside model
