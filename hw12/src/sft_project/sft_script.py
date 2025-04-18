import os
import logging
import re
from typing import Dict, Any, Optional

import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling, # Explicit import
)
# --- Added PEFT imports ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# --- End PEFT imports ---

import wandb
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions (load_config, init_wandb, extract_gsm8k_answer) remain the same ---
# (Assuming they are defined as in the previous version)
def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Loads configuration from a YAML file using OmegaConf."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        conf = OmegaConf.load(config_path)
        logger.info("Configuration loaded successfully.")
        # Validation for one-shot example (from previous step)
        if conf.evaluation.get("base_model_prompt_strategy") == "one_shot":
            if not conf.evaluation.get("one_shot_example") or \
               not conf.evaluation.one_shot_example.get("question") or \
               not conf.evaluation.one_shot_example.get("answer"):
                logger.warning("Evaluation strategy is 'one_shot', but 'evaluation.one_shot_example' "
                               "with 'question' and 'answer' fields is missing or incomplete in config.yaml. "
                               "Defaulting to zero-shot for base model evaluation.")
        # --- Add validation for LoRA config ---
        if conf.get("tuning_method") == "lora":
             if not conf.get("lora_config"):
                  logger.error("Tuning method is 'lora' but no 'lora_config' section found in config.yaml.")
                  raise ValueError("Missing lora_config for LoRA tuning method.")
             # Basic check for required LoRA params
             required_lora_params = ['r', 'lora_alpha', 'target_modules', 'task_type']
             missing_params = [p for p in required_lora_params if not conf.lora_config.get(p)]
             if missing_params:
                  logger.error(f"Missing required parameters in lora_config: {missing_params}")
                  raise ValueError(f"Missing required LoRA parameters: {missing_params}")
             if conf.lora_config.task_type != "CAUSAL_LM":
                  logger.warning(f"lora_config.task_type is '{conf.lora_config.task_type}', expected 'CAUSAL_LM' for this script.")

        # --- End validation ---
        return conf
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def init_wandb(cfg: DictConfig):
    """Initializes Weights & Biases if enabled."""
    if "wandb" in cfg.training.report_to:
        logger.info("Initializing WandB...")
        try:
            # Append tuning method to run name if desired
            run_name = cfg.wandb.get("run_name", "sft-run")
            if cfg.get("tuning_method"):
                 run_name += f"-{cfg.tuning_method}"

            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )
            # Watching gradients might be less informative or problematic with LoRA
            watch_log = cfg.wandb.get("watch", "gradients")
            if cfg.get("tuning_method") == "lora" and watch_log == "gradients":
                 logger.warning("WandB watching gradients with LoRA might not be optimal. Consider 'all' or 'false'.")
            wandb.watch(models=None, log=watch_log, log_freq=100)

            logger.info("WandB initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            cfg.training.report_to = [r for r in cfg.training.report_to if r != "wandb"]
            return False
    return False

def extract_gsm8k_answer(completion: str) -> str | None:
    """Extracts the final numerical answer from GSM8K generated text."""
    match = re.search(r"####\s*([\d.,]+)\s*$", completion)
    if match:
        return match.group(1).replace(",", "")
    return None

# --- Data Loading and Preprocessing (format_prompt, load_and_prepare_data) remain the same ---
# (Assuming they are defined as in the previous version)
def format_prompt(example: Dict[str, Any], cfg: DictConfig) -> Dict[str, str]:
    """Formats a single example using the templates from the config."""
    prompt = cfg.dataset.prompt_format.format(question=example['question'])
    response = cfg.dataset.response_template.format(answer=example['answer'])
    text = prompt + response
    return {"text": text, "prompt": prompt, "ground_truth_answer": example['answer']}

def load_and_prepare_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Loads the dataset, formats prompts, and tokenizes."""
    logger.info(f"Loading dataset: {cfg.dataset.name} ({cfg.dataset.config_name})")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.config_name, token=cfg.model.access_token)

    logger.info("Formatting prompts...")
    formatted_dataset = dataset.map(
        lambda x: format_prompt(x, cfg),
        remove_columns=list(dataset[cfg.dataset.train_split].column_names)
    )

    logger.info("Tokenizing dataset...")
    max_seq_length = cfg.dataset.max_seq_length

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_datasets = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    train_dataset = tokenized_datasets[cfg.dataset.train_split]
    trainer_eval_dataset = tokenized_datasets[cfg.dataset.eval_split].remove_columns(["prompt", "ground_truth_answer"])
    raw_eval_dataset_for_generation = formatted_dataset[cfg.dataset.eval_split]

    logger.info(f"Dataset loaded and prepared. Train size: {len(train_dataset)}, Eval size: {len(raw_eval_dataset_for_generation)}")
    return {"train": train_dataset, "eval_for_trainer": trainer_eval_dataset, "raw_eval_for_generation": raw_eval_dataset_for_generation}


# --- Model Loading (load_model_and_tokenizer) remains mostly the same ---
# PEFT wrapping happens *after* loading the base model
def load_model_and_tokenizer(cfg: DictConfig) -> (AutoModelForCausalLM, AutoTokenizer):
    """Loads the base model and tokenizer based on the configuration."""
    logger.info(f"Loading base model: {cfg.model.name}")
    model_name = cfg.model.name
    access_token = cfg.model.access_token
    trust_remote_code = cfg.model.trust_remote_code

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": access_token,
    }
    # Determine precision
    if cfg.training.bf16:
        logger.info("Using bfloat16 precision for model loading.")
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif cfg.training.fp16:
        logger.info("Using float16 precision for model loading.")
        model_kwargs["torch_dtype"] = torch.float16
    else:
         logger.info("Using default float32 precision for model loading.")

    # Note: device_map is handled by Accelerator/Trainer later
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=access_token,
        trust_remote_code=trust_remote_code,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token=eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Base model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Evaluation (evaluate_gsm8k) remains the same ---
# It evaluates the model passed to it, whether it's base, full SFT, or LoRA adapted
@torch.no_grad()
def evaluate_gsm8k(
    model: AutoModelForCausalLM, # Can be base, full SFT, or PEFT model
    tokenizer: AutoTokenizer,
    dataset: Any,
    cfg: DictConfig,
    accelerator: Accelerator,
    is_base_model_eval: bool = False
) -> Dict[str, float]:
    """
    Evaluates the model on the GSM8K dataset using generation and exact match.
    Uses Accelerator for distributed evaluation.
    Applies zero-shot or one-shot prompting for base model evaluation based on config.
    """
    eval_type = "Base Model" if is_base_model_eval else "Fine-Tuned Model"
    logger.info(f"--- Starting GSM8K evaluation for {eval_type} ---")
    model.eval()

    prompts_base = dataset["prompt"]
    ground_truths = dataset["ground_truth_answer"]
    processed_samples = 0

    one_shot_text = ""
    if is_base_model_eval:
        strategy = cfg.evaluation.get("base_model_prompt_strategy", "zero_shot")
        logger.info(f"Base model evaluation strategy: {strategy}")
        if strategy == "one_shot":
            one_shot_example = cfg.evaluation.get("one_shot_example")
            if one_shot_example and one_shot_example.get("question") and one_shot_example.get("answer"):
                one_shot_q = one_shot_example.question
                one_shot_a = one_shot_example.answer
                one_shot_text = f"Question: {one_shot_q}\nAnswer: {one_shot_a}\n\n"
                logger.info("Using one-shot example for base model evaluation.")
            else:
                logger.warning("One-shot strategy selected, but valid 'one_shot_example' not found. Falling back to zero-shot.")
                strategy = "zero_shot"
        elif strategy != "zero_shot":
             logger.warning(f"Unknown base_model_prompt_strategy '{strategy}'. Defaulting to zero-shot.")

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    samples_per_process = (len(prompts_base) + num_processes - 1) // num_processes
    start_index = process_index * samples_per_process
    end_index = min(start_index + samples_per_process, len(prompts_base))

    logger.info(f"Process {process_index}/{num_processes}: Evaluating samples {start_index} to {end_index-1}")

    generation_kwargs = {
        "max_new_tokens": cfg.evaluation.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": cfg.evaluation.temperature,
        "do_sample": cfg.evaluation.do_sample,
    }
    # Handle potential empty token ids
    if generation_kwargs["pad_token_id"] is None:
         generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
         logger.warning(f"pad_token_id is None, setting to eos_token_id ({tokenizer.eos_token_id}) for generation.")
    if generation_kwargs["eos_token_id"] is None:
         logger.error("eos_token_id is None. Generation might not stop correctly.")
         # Potentially remove it from kwargs if it causes issues and model handles stopping internally
         # del generation_kwargs["eos_token_id"]


    local_predictions = []
    if start_index < end_index:
        for i in range(start_index, end_index):
            base_prompt = prompts_base[i]
            final_prompt = base_prompt
            if is_base_model_eval and one_shot_text:
                final_prompt = one_shot_text + base_prompt

            max_input_length = cfg.dataset.max_seq_length - cfg.evaluation.max_new_tokens
            # Ensure max_input_length is positive
            if max_input_length <= 0:
                 logger.error(f"max_seq_length ({cfg.dataset.max_seq_length}) is too small for max_new_tokens ({cfg.evaluation.max_new_tokens}). Cannot generate.")
                 # Handle error appropriately, e.g., skip sample or raise error
                 local_predictions.append("[ERROR: Input too long]")
                 continue # Skip this sample

            inputs = tokenizer(final_prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_input_length)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

            try:
                 outputs = model.generate(**inputs, **generation_kwargs)
                 completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                 completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                 local_predictions.append(completion)
            except Exception as gen_e:
                 logger.error(f"Error during generation for sample {i}: {gen_e}", exc_info=True)
                 local_predictions.append("[ERROR: Generation Failed]")


            processed_samples += 1
            if processed_samples % 50 == 0:
                logger.info(f"Process {process_index}: Generated {processed_samples} samples for {eval_type}...")

    logger.info(f"Process {process_index}: Finished generation for {eval_type}. Gathering results...")
    all_predictions_list = accelerator.gather_object(local_predictions)

    exact_match_count = 0
    total_count = 0
    results = {}

    if accelerator.is_main_process:
        logger.info(f"Main process calculating Exact Match accuracy for {eval_type}...")
        all_predictions = [item for sublist in all_predictions_list for item in sublist]

        if len(all_predictions) != len(ground_truths):
             logger.warning(f"Mismatch in gathered predictions ({len(all_predictions)}) and ground truths ({len(ground_truths)}) for {eval_type}. Accuracy calculation might be incorrect.")
             total_count = len(all_predictions)
        else:
             total_count = len(ground_truths)

        for i in range(len(all_predictions)):
            completion = all_predictions[i]
            pred_answer = extract_gsm8k_answer(completion)
            if i < len(ground_truths):
                true_answer = extract_gsm8k_answer(ground_truths[i])
                if pred_answer is not None and true_answer is not None:
                    try:
                        if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6:
                            exact_match_count += 1
                    except ValueError:
                        if pred_answer == true_answer:
                           exact_match_count += 1
                elif pred_answer == true_answer:
                    exact_match_count += 1
            else:
                 logger.warning(f"Skipping comparison for prediction index {i} due to ground truth length mismatch.")

        accuracy = (exact_match_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"GSM8K Evaluation Results ({eval_type}): Exact Match = {accuracy:.2f}% ({exact_match_count}/{total_count})")
        results = {"gsm8k_exact_match": accuracy}

        if wandb.run:
            log_key_prefix = "eval/base_model_" if is_base_model_eval else "eval/sft_model_"
            wandb.log({f"{log_key_prefix}gsm8k_exact_match": accuracy})

    accelerator.wait_for_everyone()
    model.train()
    return results


# --- Training ---

def train_model(cfg: DictConfig):
    """Main function to orchestrate the SFT process."""

    set_seed(cfg.training.seed)
    is_wandb_initialized = init_wandb(cfg)

    fsdp_plugin = None
    if Accelerator().distributed_type == DistributedType.FSDP:
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_type="FULL_STATE_DICT")
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num Processes: {accelerator.num_processes}, Distributed Type: {accelerator.distributed_type}")

    # Load base model and tokenizer
    with accelerator.main_process_first():
        model, tokenizer = load_model_and_tokenizer(cfg)

    # --- PEFT Setup (if applicable) ---
    tuning_method = cfg.get("tuning_method", "full")
    logger.info(f"Selected tuning method: {tuning_method}")

    if tuning_method == "lora":
        logger.info("Applying LoRA PEFT adapter...")
        peft_config = LoraConfig(
            r=cfg.lora_config.r,
            lora_alpha=cfg.lora_config.lora_alpha,
            target_modules=list(cfg.lora_config.target_modules), # Convert OmegaConf list
            lora_dropout=cfg.lora_config.lora_dropout,
            bias=cfg.lora_config.bias,
            task_type=getattr(TaskType, cfg.lora_config.task_type, TaskType.CAUSAL_LM) # Get enum from string
        )
        
        # Prepare model for PEFT (e.g., gradient checkpointing compatibility)
        # Note: prepare_model_for_kbit_training also handles gradient checkpointing prep
        # if cfg.training.gradient_checkpointing:
        #      model.enable_input_require_grads() # May be needed depending on model/transformers version

        model = get_peft_model(model, peft_config)
        logger.info("LoRA adapter applied.")
        model.print_trainable_parameters() # Log trainable parameters

    elif tuning_method != "full":
        logger.error(f"Unknown tuning_method '{tuning_method}'. Exiting.")
        return # Or raise error

    # --- Load Data ---
    with accelerator.main_process_first():
        datasets = load_and_prepare_data(cfg, tokenizer)
    train_dataset = datasets["train"]
    trainer_eval_dataset = datasets["eval_for_trainer"]
    raw_eval_dataset_for_generation = datasets["raw_eval_for_generation"]

    # --- Evaluate Base Model (Before Training / PEFT application) ---
    # Note: If using LoRA, this evaluates the *unadapted* base model
    # To evaluate the base model *before* PEFT wrapping, we need to load it again or do eval first.
    # Let's evaluate *before* PEFT wrapping for a true base model eval.
    logger.info("--- Evaluating Base Model (Before Any Training/Adapters) ---")
    # Need to prepare the original model for evaluation
    # Create a temporary model instance for base eval if PEFT is used?
    # Or evaluate first, then apply PEFT. Let's do that.

    # Load base model and tokenizer again just for clean evaluation (if PEFT)
    # This is slightly inefficient but ensures clean base model state.
    if tuning_method == "lora":
         # Reload model cleanly before applying PEFT
         logger.info("Reloading base model for clean pre-evaluation...")
         with accelerator.main_process_first():
              model_for_eval, _ = load_model_and_tokenizer(cfg) # Use same tokenizer
    else:
         model_for_eval = model # Use the already loaded model if full tuning

    eval_model_base = accelerator.prepare(model_for_eval)
    base_model_metrics = evaluate_gsm8k(
        eval_model_base, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=True
    )
    del eval_model_base, model_for_eval # Clean up temporary model
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    # --- End Base Model Evaluation ---

    # --- Apply PEFT now if configured ---
    if tuning_method == "lora":
         # We need the original 'model' variable which was loaded earlier
         logger.info("Applying LoRA PEFT adapter (post-base evaluation)...")
         # Re-create config and apply adapter
         peft_config = LoraConfig(
             r=cfg.lora_config.r, lora_alpha=cfg.lora_config.lora_alpha,
             target_modules=list(cfg.lora_config.target_modules),
             lora_dropout=cfg.lora_config.lora_dropout, bias=cfg.lora_config.bias,
             task_type=getattr(TaskType, cfg.lora_config.task_type, TaskType.CAUSAL_LM)
         )
         model = get_peft_model(model, peft_config)
         logger.info("LoRA adapter applied for training.")
         model.print_trainable_parameters()


    # --- Set up Training Arguments ---
    logger.info("Setting up Training Arguments...")
    # Adjust learning rate potentially for LoRA
    lr = cfg.training.learning_rate
    if tuning_method == "lora" and lr == 1e-5: # Example check
         logger.warning("Default learning rate 1e-5 might be low for LoRA. Consider 1e-4 or 2e-4.")

    # Check optimizer compatibility with PEFT if needed
    if cfg.training.optim == "adamw_bnb_8bit":
         logger.info("Using Paged 8-bit AdamW optimizer.")
         try:
            import bitsandbytes as bnb
         except ImportError:
            logger.error("bitsandbytes is not installed, required for adamw_bnb_8bit.")
            raise

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir, # Consider adding tuning method to path
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        learning_rate=lr, # Use potentially adjusted LR
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_total_limit=cfg.training.save_total_limit,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        optim=cfg.training.optim,
        seed=cfg.training.seed,
        report_to=cfg.training.report_to if is_wandb_initialized else "none",
        dataloader_num_workers = 2,
        # gradient_checkpointing_kwargs={'use_reentrant': False} # May be needed for newer torch/transformers with FSDP/PEFT
        # max_grad_norm=cfg.training.get("max_grad_norm", None), # Add if enabling grad clipping
    )

    # --- Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model, # Pass the potentially PEFT-adapted model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    # If using PEFT, ensure internal model reference is correct
    # model.config.use_cache = False # Often recommended with gradient checkpointing/PEFT

    # --- Train ---
    logger.info(f"--- Starting Fine-Tuning ({tuning_method}) ---")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")
        metrics = train_result.metrics

        if accelerator.is_main_process:
            # Saving handles PEFT adapters automatically if model is PeftModel
            logger.info("Saving final model/adapter...")
            final_save_path = os.path.join(cfg.training.output_dir, f"final_checkpoint_{tuning_method}")
            os.makedirs(final_save_path, exist_ok=True) # Ensure dir exists

            trainer.save_model(final_save_path) # Saves adapter/model, tokenizer, config
            # Tokenizer might not be saved automatically by save_model with PEFT, save explicitly
            tokenizer.save_pretrained(final_save_path)
            # Save the run config
            try:
                 OmegaConf.save(cfg, os.path.join(final_save_path, "training_config.yaml"))
            except Exception as e_save:
                 logger.error(f"Failed to save training config: {e_save}")

            logger.info(f"Final model/adapter saved to {final_save_path}")
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            if is_wandb_initialized and wandb.run:
                 final_metrics = {"train/final_" + k: v for k,v in metrics.items()}
                 wandb.log(final_metrics)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if accelerator.is_main_process:
             logger.info("Attempting to save state due to error...")
             trainer.save_state()
        raise

    # --- Evaluate Fine-Tuned Model ---
    accelerator.wait_for_everyone()
    logger.info(f"--- Evaluating Fine-Tuned Model ({tuning_method}) ---")
    eval_model_sft = accelerator.prepare(trainer.model)
    sft_model_metrics = evaluate_gsm8k(
        eval_model_sft, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=False
    )
    # Logging happens inside evaluate_gsm8k

    # --- Clean Up ---
    if is_wandb_initialized and wandb.run and accelerator.is_main_process:
        logger.info("Finishing WandB run...")
        wandb.finish()

    logger.info("Script finished successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)
