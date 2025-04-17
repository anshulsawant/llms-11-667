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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login


model_id = "google/gemma-2-9b"

dataset_name = "gsm8k"
dataset_config = "main" # Or "socratic"
new_model_name = "gemma-2-9b-gsm8k-full-sft"
output_dir = "./results_full_ft"

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
    optim="paged_adamw_8bit",
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
collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer)
config = SFTConfig(max_len=512)
trainer = SFTTrainer(
    args=config,
    model=model,
    data_collator=collator,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# --- Start Training ---
print("Starting full fine-tuning...")
trainer.train()

print(f"Saving the full fine-tuned model to {new_model_name}...")
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)
