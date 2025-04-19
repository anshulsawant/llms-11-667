# src/sft_project/train.py
"""Script for training (Full SFT or LoRA SFT) based on config."""

import os
import logging
import math
import argparse

import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf, DictConfig, ListConfig
from transformers import (
    AutoModelForCausalLM, # Keep for type hints maybe
    AutoTokenizer, # Keep for type hints maybe
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# --- Import from utils ---
from .utils import (
    load_config,
    init_wandb,
    load_model_and_tokenizer,
    format_prompt
)
# --- End imports ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_train_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Loads and prepares the training dataset subset."""
    logger.info(f"Loading dataset for training: {cfg.dataset.name} ({cfg.dataset.config_name})")
    # Load only the training split
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config_name,
        split=cfg.dataset.train_split, # Load only the train split specified
        token=cfg.model.get("access_token", None)
    )

    logger.info("Formatting training prompts...")
    # Use utils.format_prompt
    formatted_dataset = dataset.map(
        lambda x: format_prompt(x, cfg),
        remove_columns=list(dataset.column_names)
    )

    # Select subset if specified
    num_train_samples = cfg.dataset.get("num_train_samples", None)
    if num_train_samples is not None and num_train_samples > 0:
        logger.info(f"Selecting first {num_train_samples} samples from training split.")
        select_range = range(min(num_train_samples, len(formatted_dataset)))
        selected_train_dataset = formatted_dataset.select(select_range)
    else:
        selected_train_dataset = formatted_dataset # Use full loaded split

    logger.info("Tokenizing training dataset...")
    max_seq_length = cfg.dataset.max_seq_length
    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_seq_length
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_train_dataset = selected_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "prompt", "ground_truth_answer"]
    )

    logger.info(f"Training dataset prepared. Final size: {len(tokenized_train_dataset)}")
    return tokenized_train_dataset


def train(cfg: DictConfig):
    """Main training function."""
    set_seed(cfg.training.seed)
    is_wandb_initialized = init_wandb(cfg, job_type="train") # Pass job type

    # Accelerator setup (optional here, Trainer uses it internally, but good practice)
    # fsdp_plugin = None
    # if Accelerator().distributed_type == DistributedType.FSDP: fsdp_plugin = FullyShardedDataParallelPlugin(...)
    # accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    accelerator = Accelerator() # Basic init for is_main_process checks etc.
    logger.info(f"Accelerator initialized: {accelerator.device}, Num Processes: {accelerator.num_processes}, Type: {accelerator.distributed_type}")


    # Load base model and tokenizer using utils function
    # Ensure model loading happens once per node or globally
    # with accelerator.main_process_first(): # May not be needed if Trainer handles downloads
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load Data using specific train data function
    # with accelerator.main_process_first():
    train_dataset = load_and_prepare_train_data(cfg, tokenizer)

    # Apply PEFT if configured
    tuning_method = cfg.get("tuning_method", "full")
    logger.info(f"Selected tuning method: {tuning_method}")
    if tuning_method == "lora":
         logger.info("Applying LoRA PEFT adapter...")
         target_modules_list = list(cfg.lora_config.target_modules) if isinstance(cfg.lora_config.target_modules, ListConfig) else cfg.lora_config.target_modules
         peft_config = LoraConfig(
             r=cfg.lora_config.r, lora_alpha=cfg.lora_config.lora_alpha,
             target_modules=target_modules_list,
             lora_dropout=cfg.lora_config.lora_dropout, bias=cfg.lora_config.bias,
             task_type=getattr(TaskType, cfg.lora_config.task_type, TaskType.CAUSAL_LM)
         )
         model = get_peft_model(model, peft_config)
         logger.info("LoRA adapter applied for training.")
         model.print_trainable_parameters()
    elif tuning_method != "full":
        logger.error(f"Unknown tuning_method '{tuning_method}'. Exiting.")
        return

    # Set up Training Arguments
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    # Handle potential key mismatches or necessary adjustments
    if "evaluation_strategy" in training_args_dict and "eval_strategy" not in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")
        logger.info("Renamed config key 'evaluation_strategy' to 'eval_strategy'.")
    if training_args_dict.get("eval_strategy", "no") != "no":
         logger.warning("Evaluation during training is configured but train.py does not perform it. Set eval_strategy='no' in config.")
         training_args_dict["eval_strategy"] = "no" # Force no eval during training
    if "eval_steps" in training_args_dict: del training_args_dict["eval_steps"] # Remove eval_steps
    # Handle lr_scheduler_kwargs
    if "lr_scheduler_kwargs" in training_args_dict and training_args_dict["lr_scheduler_kwargs"] is not None:
         if not isinstance(training_args_dict["lr_scheduler_kwargs"], dict):
             try: training_args_dict["lr_scheduler_kwargs"] = dict(training_args_dict["lr_scheduler_kwargs"])
             except (TypeError, ValueError): logger.error("Could not convert lr_kwargs."); del training_args_dict["lr_scheduler_kwargs"]
         if training_args_dict.get("lr_scheduler_type") == "cosine_with_min_lr":
              if "min_lr" not in training_args_dict["lr_scheduler_kwargs"]: logger.warning("min_lr not in lr_kwargs.")
    if "min_lr" in training_args_dict: del training_args_dict["min_lr"]

    logger.info("Initializing Training Arguments...")
    training_args = TrainingArguments(**training_args_dict)

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None, # No evaluation during training
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    logger.info(f"--- Starting Fine-Tuning ({tuning_method}) ---")
    try:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")
        metrics = train_result.metrics
        if accelerator.is_main_process:
             trainer.log_metrics("train", metrics)
             trainer.save_metrics("train", metrics)
             trainer.save_state()
             if is_wandb_initialized and wandb.run:
                  final_metrics = {"train/final_" + k: v for k,v in metrics.items()}
                  wandb.log(final_metrics)

            # Save Model/Adapter
             logger.info("Saving final model/adapter...")
             # Use output_dir from TrainingArguments
             final_save_path = os.path.join(training_args.output_dir, f"final_checkpoint")
             os.makedirs(final_save_path, exist_ok=True)
             trainer.save_model(final_save_path) # Handles PEFT adapters automatically
             tokenizer.save_pretrained(final_save_path) # Save tokenizer explicitly
             try: OmegaConf.save(cfg, os.path.join(final_save_path, "training_config_merged.yaml"))
             except Exception as e_save: logger.error(f"Failed to save merged training config: {e_save}")
             logger.info(f"Final model/adapter saved to {final_save_path}")
             # --- End Saving ---

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        if accelerator.is_main_process:
            logger.info("Attempting to save state due to error...")
            try: trainer.save_state()
            except Exception as save_e: logger.error(f"Failed to save state after training error: {save_e}", exc_info=True)
        raise # Re-raise the original training error

    # Clean Up WandB
    if is_wandb_initialized and wandb.run and accelerator.is_main_process:
        logger.info("Finishing WandB run..."); wandb.finish()

    logger.info("Training script finished successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    args = parser.parse_args()
    # Load merged configuration using the utility function
    config = load_config(override_config_path=args.config, base_config_path=args.base_config)
    # Start training
    train(config)
