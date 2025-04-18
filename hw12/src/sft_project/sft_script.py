import os
import logging
import re
import math
from typing import Dict, Any, Optional

import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf, DictConfig, ListConfig # Import ListConfig for type checking
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
    # No longer need explicit scheduler/optimizer imports here
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import wandb
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Loads configuration from a YAML file using OmegaConf."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        conf = OmegaConf.load(config_path)
        logger.info("Configuration loaded successfully.")
        # Validations (keep previous)
        if conf.evaluation.get("base_model_prompt_strategy") == "one_shot":
            if not conf.evaluation.get("one_shot_example") or \
               not conf.evaluation.one_shot_example.get("question") or \
               not conf.evaluation.one_shot_example.get("answer"):
                logger.warning("One-shot incomplete. Defaulting to zero-shot.")
        if conf.get("tuning_method") == "lora":
             if not conf.get("lora_config"): raise ValueError("Missing lora_config for LoRA method.")
             req_lora = ['r', 'lora_alpha', 'target_modules', 'task_type']
             missing = [p for p in req_lora if not conf.lora_config.get(p)]
             if missing: raise ValueError(f"Missing required LoRA parameters: {missing}")
             if conf.lora_config.task_type != "CAUSAL_LM": logger.warning("lora_config.task_type not CAUSAL_LM.")
        # --- Validation for min_lr within lr_scheduler_kwargs ---
        if conf.training.get("lr_scheduler_type") == "cosine_with_min_lr":
             lr_kwargs = conf.training.get("lr_scheduler_kwargs")
             # Check if kwargs exist and contain the expected key 'min_lr'
             if lr_kwargs is None or lr_kwargs.get("min_lr") is None:
                  # Allow None kwargs, scheduler might have default min_lr? Or raise error?
                  # Let's warn for now if type is set but kwargs/min_lr missing
                   logger.warning("lr_scheduler_type is 'cosine_with_min_lr' but 'lr_scheduler_kwargs.min_lr' is missing. Scheduler might use default behavior or fail.")
             elif not isinstance(lr_kwargs.min_lr, (float, int)) or lr_kwargs.min_lr < 0.0:
                  logger.error(f"lr_scheduler_kwargs.min_lr ({lr_kwargs.min_lr}) must be a non-negative number.")
                  raise ValueError("Invalid min_lr value in lr_scheduler_kwargs.")
             elif lr_kwargs.min_lr >= conf.training.learning_rate:
                  logger.warning(f"lr_scheduler_kwargs.min_lr ({lr_kwargs.min_lr}) is >= learning_rate ({conf.training.learning_rate}). Scheduler might not behave as expected.")
        # --- End validation ---
        return conf
    except FileNotFoundError: logger.error(f"Config file not found: {config_path}"); raise
    except Exception as e: logger.error(f"Error loading configuration: {e}"); raise

def init_wandb(cfg: DictConfig):
    # (Implementation remains the same)
    if "wandb" in cfg.training.report_to:
        logger.info("Initializing WandB...")
        try:
            run_name = cfg.wandb.get("run_name", "sft-run")
            if cfg.get("tuning_method"): run_name += f"-{cfg.tuning_method}"
            wandb.init(project=cfg.wandb.project, name=run_name, config=OmegaConf.to_container(cfg, resolve=True), resume="allow")
            watch_log = cfg.wandb.get("watch", "gradients")
            if cfg.get("tuning_method") == "lora" and watch_log == "gradients": logger.warning("WandB watching gradients with LoRA might not be optimal.")
            wandb.watch(models=None, log=watch_log, log_freq=100)
            logger.info("WandB initialized successfully."); return True
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            cfg.training.report_to = [r for r in cfg.training.report_to if r != "wandb"]
            return False
    return False

def extract_gsm8k_answer(completion: str) -> str | None:
    # (Implementation remains the same)
    match = re.search(r"####\s*([\d.,]+)\s*$", completion)
    if match: return match.group(1).replace(",", "")
    return None

# --- Data Loading and Preprocessing ---
def format_prompt(example: Dict[str, Any], cfg: DictConfig) -> Dict[str, str]:
    # (Implementation remains the same)
    prompt = cfg.dataset.prompt_format.format(question=example['question'])
    response = cfg.dataset.response_template.format(answer=example['answer'])
    text = prompt + response
    return {"text": text, "prompt": prompt, "ground_truth_answer": example['answer']}

def load_and_prepare_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    # (Implementation remains the same)
    logger.info(f"Loading dataset: {cfg.dataset.name} ({cfg.dataset.config_name})")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.config_name, token=cfg.model.access_token)
    logger.info("Formatting prompts...")
    formatted_dataset = dataset.map(lambda x: format_prompt(x, cfg), remove_columns=list(dataset[cfg.dataset.train_split].column_names))
    logger.info("Tokenizing dataset...")
    max_seq_length = cfg.dataset.max_seq_length
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output
    tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_datasets[cfg.dataset.train_split]
    trainer_eval_dataset = tokenized_datasets[cfg.dataset.eval_split].remove_columns(["prompt", "ground_truth_answer"])
    raw_eval_dataset_for_generation = formatted_dataset[cfg.dataset.eval_split]
    logger.info(f"Dataset loaded. Train size: {len(train_dataset)}, Eval size: {len(raw_eval_dataset_for_generation)}")
    return {"train": train_dataset, "eval_for_trainer": trainer_eval_dataset, "raw_eval_for_generation": raw_eval_dataset_for_generation}

# --- Model Loading ---
def load_model_and_tokenizer(cfg: DictConfig) -> (AutoModelForCausalLM, AutoTokenizer):
    # (Implementation remains the same)
    logger.info(f"Loading base model: {cfg.model.name}")
    model_name = cfg.model.name; access_token = cfg.model.access_token; trust_remote_code = cfg.model.trust_remote_code
    model_kwargs = {"trust_remote_code": trust_remote_code, "token": access_token}
    if cfg.training.bf16: logger.info("Using bfloat16 precision."); model_kwargs["torch_dtype"] = torch.bfloat16
    elif cfg.training.fp16: logger.info("Using float16 precision."); model_kwargs["torch_dtype"] = torch.float16
    else: logger.info("Using default float32 precision.")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, trust_remote_code=trust_remote_code, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer setting pad_token=eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    logger.info("Base model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Evaluation ---
@torch.no_grad()
def evaluate_gsm8k( model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Any, cfg: DictConfig, accelerator: Accelerator, is_base_model_eval: bool = False) -> Dict[str, float]:
    # (Implementation remains the same)
    eval_type = "Base Model" if is_base_model_eval else "Fine-Tuned Model"
    logger.info(f"--- Starting GSM8K evaluation for {eval_type} ---"); model.eval()
    prompts_base = dataset["prompt"]; ground_truths = dataset["ground_truth_answer"]
    one_shot_text = ""
    if is_base_model_eval:
        strategy = cfg.evaluation.get("base_model_prompt_strategy", "zero_shot"); logger.info(f"Base model eval strategy: {strategy}")
        if strategy == "one_shot":
            one_shot_example = cfg.evaluation.get("one_shot_example")
            if one_shot_example and one_shot_example.get("question") and one_shot_example.get("answer"):
                one_shot_q = one_shot_example.question; one_shot_a = one_shot_example.answer
                one_shot_text = f"Question: {one_shot_q}\nAnswer: {one_shot_a}\n\n"; logger.info("Using one-shot example.")
            else: logger.warning("One-shot requested but example invalid. Falling back to zero-shot.")
        elif strategy != "zero_shot": logger.warning(f"Unknown strategy '{strategy}'. Defaulting to zero-shot.")
    num_processes = accelerator.num_processes; process_index = accelerator.process_index
    samples_per_process = (len(prompts_base) + num_processes - 1) // num_processes
    start_index = process_index * samples_per_process; end_index = min(start_index + samples_per_process, len(prompts_base))
    logger.info(f"Process {process_index}/{num_processes}: Evaluating samples {start_index} to {end_index-1}")
    generation_kwargs = {"max_new_tokens": cfg.evaluation.max_new_tokens, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "temperature": cfg.evaluation.temperature, "do_sample": cfg.evaluation.do_sample}
    if generation_kwargs["pad_token_id"] is None: generation_kwargs["pad_token_id"] = tokenizer.eos_token_id; logger.warning(f"pad_token_id is None, setting to eos_token_id ({tokenizer.eos_token_id})")
    if generation_kwargs["eos_token_id"] is None: logger.error("eos_token_id is None.")
    local_predictions = []; processed_samples = 0
    if start_index < end_index:
        for i in range(start_index, end_index):
            base_prompt = prompts_base[i]; final_prompt = base_prompt
            if is_base_model_eval and one_shot_text: final_prompt = one_shot_text + base_prompt
            max_input_length = cfg.dataset.max_seq_length - cfg.evaluation.max_new_tokens
            if max_input_length <= 0: logger.error(f"max_seq_length too small."); local_predictions.append("[ERROR: Input too long]"); continue
            inputs = tokenizer(final_prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_input_length); inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            try:
                 outputs = model.generate(**inputs, **generation_kwargs)
                 completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]; completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                 local_predictions.append(completion)
            except Exception as gen_e: logger.error(f"Generation error sample {i}: {gen_e}", exc_info=False); local_predictions.append("[ERROR: Generation Failed]")
            processed_samples += 1
            if processed_samples % 50 == 0: logger.info(f"Process {process_index}: Generated {processed_samples} samples for {eval_type}...")
    logger.info(f"Process {process_index}: Finished generation for {eval_type}. Gathering results..."); all_predictions_list = accelerator.gather_object(local_predictions)
    exact_match_count = 0; total_count = 0; results = {}
    if accelerator.is_main_process:
        logger.info(f"Main process calculating Exact Match accuracy for {eval_type}...")
        all_predictions = [item for sublist in all_predictions_list for item in sublist]
        if len(all_predictions) != len(ground_truths): logger.warning(f"Mismatch in predictions ({len(all_predictions)}) and ground truths ({len(ground_truths)}) for {eval_type}."); total_count = len(all_predictions)
        else: total_count = len(ground_truths)
        for i in range(len(all_predictions)):
            completion = all_predictions[i]; pred_answer = extract_gsm8k_answer(completion)
            if i < len(ground_truths):
                true_answer = extract_gsm8k_answer(ground_truths[i])
                if pred_answer is not None and true_answer is not None:
                    try:
                        if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6: exact_match_count += 1
                    except ValueError:
                        if pred_answer == true_answer: exact_match_count += 1
                elif pred_answer == true_answer: exact_match_count += 1
            else: logger.warning(f"Skipping comparison for prediction index {i} due to ground truth length mismatch.")
        accuracy = (exact_match_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"GSM8K Evaluation Results ({eval_type}): Exact Match = {accuracy:.2f}% ({exact_match_count}/{total_count})"); results = {"gsm8k_exact_match": accuracy}
        if wandb.run: log_key_prefix = "eval/base_model_" if is_base_model_eval else "eval/sft_model_"; wandb.log({f"{log_key_prefix}gsm8k_exact_match": accuracy})
    accelerator.wait_for_everyone(); model.train(); return results

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

    # Load Data
    with accelerator.main_process_first():
        datasets = load_and_prepare_data(cfg, tokenizer)
    train_dataset = datasets["train"]
    trainer_eval_dataset = datasets["eval_for_trainer"]
    raw_eval_dataset_for_generation = datasets["raw_eval_for_generation"]

    # --- Evaluate Base Model ---
    logger.info("--- Evaluating Base Model (Before Any Training/Adapters) ---")
    model_for_eval = model
    eval_model_base = accelerator.prepare(model_for_eval)
    base_model_metrics = evaluate_gsm8k(
        eval_model_base, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=True
    )
    del eval_model_base
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    # --- End Base Model Evaluation ---

    # --- Apply PEFT if configured ---
    tuning_method = cfg.get("tuning_method", "full")
    logger.info(f"Selected tuning method: {tuning_method}")
    if tuning_method == "lora":
         logger.info("Applying LoRA PEFT adapter...")
         # Convert target_modules from OmegaConf ListConfig to Python list if necessary
         target_modules_list = list(cfg.lora_config.target_modules) if isinstance(cfg.lora_config.target_modules, ListConfig) else cfg.lora_config.target_modules
         peft_config = LoraConfig(
             r=cfg.lora_config.r, lora_alpha=cfg.lora_config.lora_alpha,
             target_modules=target_modules_list, # Use Python list
             lora_dropout=cfg.lora_config.lora_dropout, bias=cfg.lora_config.bias,
             task_type=getattr(TaskType, cfg.lora_config.task_type, TaskType.CAUSAL_LM)
         )
         model = get_peft_model(model, peft_config)
         logger.info("LoRA adapter applied for training.")
         model.print_trainable_parameters()
    elif tuning_method != "full":
        logger.error(f"Unknown tuning_method '{tuning_method}'. Exiting.")
        return

    # --- Set up Training Arguments ---
    # Convert OmegaConf dictionary to standard dict for TrainingArguments
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)

    # Handle lr_scheduler_kwargs specifically - ensure it's a dict
    if "lr_scheduler_kwargs" in training_args_dict and training_args_dict["lr_scheduler_kwargs"] is not None:
         if not isinstance(training_args_dict["lr_scheduler_kwargs"], dict):
             try:
                 # Convert if it looks like a dict structure from OmegaConf
                 training_args_dict["lr_scheduler_kwargs"] = dict(training_args_dict["lr_scheduler_kwargs"])
             except (TypeError, ValueError):
                 logger.error("Could not convert lr_scheduler_kwargs to dict. Removing.")
                 del training_args_dict["lr_scheduler_kwargs"]
         # Ensure the key 'min_lr' exists if type is cosine_with_min_lr (validation done in load_config)
         if training_args_dict.get("lr_scheduler_type") == "cosine_with_min_lr":
              if "min_lr" not in training_args_dict["lr_scheduler_kwargs"]:
                   logger.warning("min_lr not found in lr_scheduler_kwargs despite scheduler type. Scheduler might use default.")

    # Remove min_lr from top level if it exists (should be inside kwargs now)
    if "min_lr" in training_args_dict:
        del training_args_dict["min_lr"]

    logger.info("Initializing Training Arguments...")
    training_args = TrainingArguments(**training_args_dict)

    # --- Initialize Trainer (Using standard init) ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args, # Trainer uses args to create optimizer/scheduler
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # optimizers=(optimizer, lr_scheduler) # REMOVED - Trainer handles it
    )

    # --- Train ---
    logger.info(f"--- Starting Fine-Tuning ({tuning_method}) ---")
    try:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")
        metrics = train_result.metrics

        if accelerator.is_main_process:
            logger.info("Saving final model/adapter...")
            # Use output_dir from TrainingArguments which might have method appended
            final_save_path = os.path.join(training_args.output_dir, f"final_checkpoint")
            os.makedirs(final_save_path, exist_ok=True)

            trainer.save_model(final_save_path)
            # Save tokenizer explicitly, especially needed for LoRA where only adapter is saved by default
            tokenizer.save_pretrained(final_save_path)
            try: OmegaConf.save(cfg, os.path.join(final_save_path, "training_config.yaml"))
            except Exception as e_save: logger.error(f"Failed to save training config: {e_save}")

            logger.info(f"Final model/adapter saved to {final_save_path}")
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            if is_wandb_initialized and wandb.run:
                 final_metrics = {"train/final_" + k: v for k,v in metrics.items()}
                 wandb.log(final_metrics)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if accelerator.is_main_process: logger.info("Attempting to save state due to error..."); trainer.save_state()
        raise

    # --- Evaluate Fine-Tuned Model ---
    accelerator.wait_for_everyone()
    logger.info(f"--- Evaluating Fine-Tuned Model ({tuning_method}) ---")
    eval_model_sft = accelerator.prepare(trainer.model)
    sft_model_metrics = evaluate_gsm8k(
        eval_model_sft, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=False
    )

    # --- Clean Up ---
    if is_wandb_initialized and wandb.run and accelerator.is_main_process:
        logger.info("Finishing WandB run..."); wandb.finish()

    logger.info("Script finished successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_model(config)
