# src/sft_project/train.py
"""Main script for Supervised Fine-Tuning (SFT)."""

import os
import logging
import argparse
import sys
from pathlib import Path

# --- Setup logger FIRST ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End logger setup ---

try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq, # Suitable for Causal LM too with proper label masking
        set_seed
    )
    from datasets import load_dataset, Dataset
    from omegaconf import OmegaConf, DictConfig
    from accelerate import Accelerator # Although not explicitly used for distribution logic here, keep for consistency
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    # --- Import from utils ---
    from .utils import (
        load_config,
        load_model_and_tokenizer,
        format_prompt,
        init_wandb, # For WandB logging
        slugify # For potential use in run names etc.
    )
    # --- End imports ---

except ImportError as e:
    logger.error(f"Failed to import required libraries or functions from .utils: {e}. Make sure all dependencies are installed and utils.py is correct and accessible.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Run SFT training using configuration files.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config_path", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    # Add other potential CLI overrides if needed, but prefer config
    return parser.parse_args()

def load_and_prepare_train_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Loads, formats, and tokenizes the training dataset."""
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset.config_name
    train_split = cfg.dataset.train_split
    num_samples = cfg.dataset.get("num_train_samples", None)
    seed = cfg.training.seed
    max_seq_length = cfg.dataset.max_seq_length

    logger.info(f"Loading training dataset: {dataset_name} ({dataset_config}), split: {train_split}")
    try:
        raw_train_dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=train_split,
            token=cfg.model.get("access_token", None)
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}/{dataset_config} split {train_split}: {e}", exc_info=True)
        raise

    # --- Select subset if specified ---
    selected_train_dataset = raw_train_dataset
    if num_samples is not None and num_samples > 0:
        actual_num_samples = min(num_samples, len(raw_train_dataset))
        if actual_num_samples < len(raw_train_dataset):
            logger.info(f"Selecting {actual_num_samples} random samples from train split '{train_split}' using seed {seed}.")
            selected_train_dataset = raw_train_dataset.shuffle(seed=seed).select(range(actual_num_samples))
        else:
             logger.info(f"num_train_samples >= dataset size. Using full train split.")

    # --- Format prompts ---
    logger.info("Formatting training prompts...")
    # Keep track of original columns to remove them *after* formatting if needed
    original_columns = selected_train_dataset.column_names
    try:
        formatted_train_dataset = selected_train_dataset.map(
            lambda x: format_prompt(x, cfg),
            # Let map handle columns based on function output, but we'll clean up after
        )
        # Ensure the expected output columns from format_prompt are present
        if "prompt" not in formatted_train_dataset.column_names or \
           "ground_truth_answer" not in formatted_train_dataset.column_names:
            logger.error("Formatted dataset missing 'prompt' or 'ground_truth_answer' column.")
            raise ValueError("Formatted dataset missing required columns for tokenization.")
        logger.info(f"Formatting complete. Columns: {formatted_train_dataset.column_names}")

    except Exception as e:
        logger.error(f"Error during prompt formatting for training: {e}", exc_info=True)
        raise

    # --- Define Tokenization Function with Label Masking ---
    def tokenize_function(examples):
        combined_texts = []
        prompt_lengths = []
        for prompt, answer in zip(examples['prompt'], examples['ground_truth_answer']):
            # Ensure prompt and answer are strings
            prompt_str = str(prompt if prompt is not None else "")
            answer_str = str(answer if answer is not None else "")
            combined_text = prompt_str + answer_str + tokenizer.eos_token
            combined_texts.append(combined_text)
            # Tokenize prompt separately to find its length for masking
            prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokens['input_ids']))

        model_inputs = tokenizer(
            combined_texts,
            max_length=max_seq_length,
            padding=False, # Keep False: Padding handled by DataCollator
            truncation=True
        )
        labels = model_inputs["input_ids"].copy()
        masked_labels = []
        for i, label_seq in enumerate(labels):
            prompt_len = prompt_lengths[i]
            # Ensure prompt_len doesn't exceed sequence length (can happen with long prompts + short max_length)
            actual_prompt_len_in_seq = min(prompt_len, len(label_seq))
            masked_seq = [-100] * actual_prompt_len_in_seq + label_seq[actual_prompt_len_in_seq:]
            # Pad the masked sequence manually if needed? No, collator handles label padding.
            # Just ensure it aligns with input_ids length before collator.
            masked_labels.append(masked_seq) # Don't truncate here, collator needs original length info

        model_inputs["labels"] = masked_labels
        return model_inputs

    # --- Tokenize the dataset ---
    logger.info("Tokenizing dataset and creating labels...")
    try:
        # Determine columns to remove *after* formatting
        columns_before_tokenization = formatted_train_dataset.column_names
        tokenized_train_dataset = formatted_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_before_tokenization # Remove all columns that existed before this step
        )

        logger.info(f"Tokenization complete. Final columns: {tokenized_train_dataset.column_names}")
        # Verify necessary columns for Trainer exist
        if not all(col in tokenized_train_dataset.column_names for col in ['input_ids', 'attention_mask', 'labels']):
             logger.error(f"Tokenized dataset missing required columns for training. Found: {tokenized_train_dataset.column_names}")
             raise ValueError("Tokenized dataset missing required columns.")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        raise

    return tokenized_train_dataset


def train(cfg: DictConfig):
    """Main training function."""
    logger.info("Starting training process...")
    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")
    set_seed(cfg.training.seed)
    logger.info(f"Seed set to {cfg.training.seed}")
    wandb_run = init_wandb(cfg, accelerator)
    model, tokenizer = load_model_and_tokenizer(cfg)

    # --- Load and Prepare Data ---
    # Result is stored in 'train_dataset'
    train_dataset = load_and_prepare_train_data(cfg, tokenizer)
    logger.info(f"Training dataset prepared with {len(train_dataset)} samples.")

    # --- Configure LoRA (if applicable) ---
    if cfg.tuning_method == "lora":
        logger.info("Configuring LoRA...")
        lora_cfg_dict = cfg.get("lora_config")
        if not lora_cfg_dict: logger.error("LoRA specified but `lora_config:` missing."); sys.exit(1)
        lora_config_args = OmegaConf.to_container(lora_cfg_dict, resolve=True)
        if isinstance(lora_config_args.get("task_type"), str):
            try: lora_config_args["task_type"] = TaskType[lora_config_args["task_type"]]
            except KeyError: logger.error(f"Invalid LoRA task_type: {lora_config_args['task_type']}."); sys.exit(1)
        peft_config = LoraConfig(**lora_config_args)
        model = get_peft_model(model, peft_config)
        logger.info("LoRA configured successfully.")
        model.print_trainable_parameters()
    elif cfg.tuning_method == "full": logger.info("Using Full SFT method.")
    else: logger.error(f"Unknown tuning_method: {cfg.tuning_method}"); sys.exit(1)

    # --- Define Training Arguments ---
    logger.info("Defining Training Arguments...")
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    report_to_val = training_args_dict.get("report_to", [])
    if not isinstance(report_to_val, list): report_to_val = [report_to_val] if report_to_val else []
    if "wandb" not in report_to_val and wandb_run is not None: logger.warning("WandB init but not in report_to.")
    if "wandb" in report_to_val and wandb_run is None: logger.warning("WandB requested but failed init."); report_to_val = [r for r in report_to_val if r != "wandb"]
    if not report_to_val: report_to_val = "none"
    training_args_dict["report_to"] = report_to_val

    training_args = TrainingArguments(**training_args_dict, remove_unused_columns=False) # Keep remove_unused_columns=False
    logger.info(f"TrainingArguments defined. Output dir: {training_args.output_dir}")

    # --- Define Data Collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        padding=True # Explicitly set padding
    )
    logger.info("Data collator defined with explicit padding=True.")

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        # --- FIX: Use the correct variable name ---
        train_dataset=train_dataset, # Pass the actual prepared dataset variable
        # --- End FIX ---
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    logger.info("Hugging Face Trainer initialized.")

    # --- Start Training ---
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")
        # Ensure saving happens only on main process
        if accelerator.is_main_process:
            logger.info("Saving final model/adapter...")
            final_save_path = Path(training_args.output_dir) / "final_checkpoint"
            final_save_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(final_save_path)) # Saves full model or adapter correctly
            logger.info(f"Final model/adapter saved to {final_save_path}")

            # Log and save metrics
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset) # Use length of dataset passed to trainer
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state() # Save optimizer, scheduler, RNG states
            logger.info(f"Training metrics and state saved.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if wandb_run: wandb_run.finish(exit_code=1) # Mark WandB run as failed
        sys.exit(1) # Exit with error code

    # --- Clean up WandB ---
    if wandb_run:
        wandb_run.finish() # Mark WandB run as successful
        logger.info("WandB run finished.")


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    try:
        config = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        train(config)
    except Exception as e:
        logger.error(f"An error occurred in the main execution flow: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Executing train.py script...")
    main()
