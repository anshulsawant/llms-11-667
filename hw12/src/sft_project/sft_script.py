import os
import logging
import re
import math
from typing import Dict, Any, Optional

import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType
from datasets import load_dataset, Dataset # Ensure Dataset is imported
from omegaconf import OmegaConf, DictConfig, ListConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
    SchedulerType
)
try:
    from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
except ImportError:
    get_cosine_with_min_lr_schedule_with_warmup = None

from torch.optim import AdamW
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import wandb
import evaluate

# --- Added tqdm import ---
from tqdm.auto import tqdm
# --- End tqdm import ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def load_config(override_config_path: str, base_config_path: str = "config.yaml") -> DictConfig:
    # (Implementation remains the same - handles merging and validation)
    logger.info(f"Loading base configuration from: {base_config_path}")
    try: base_conf = OmegaConf.load(base_config_path)
    except Exception as e: logger.error(f"Error loading base config: {e}"); raise
    logger.info(f"Loading override configuration from: {override_config_path}")
    try: override_conf = OmegaConf.load(override_config_path)
    except Exception as e: logger.error(f"Error loading override config: {e}"); raise
    logger.info("Merging configurations...")
    try:
        merged_conf = OmegaConf.merge(base_conf, override_conf)
        logger.info("Configurations merged successfully.")
        # --- Validation ---
        logger.info("Validating merged configuration...")
        if merged_conf.evaluation.get("base_model_prompt_strategy") == "one_shot":
             if not merged_conf.evaluation.get("one_shot_example") or \
                not merged_conf.evaluation.one_shot_example.get("question") or \
                not merged_conf.evaluation.one_shot_example.get("answer"):
                 logger.warning("One-shot incomplete. Defaulting to zero-shot.")
        if merged_conf.get("tuning_method") == "lora":
              if not merged_conf.get("lora_config"): raise ValueError("Missing lora_config.")
              req_lora = ['r', 'lora_alpha', 'target_modules', 'task_type']
              missing = [p for p in req_lora if not merged_conf.lora_config.get(p)]
              if missing: raise ValueError(f"Missing LoRA params: {missing}")
              if merged_conf.lora_config.task_type != "CAUSAL_LM": logger.warning("lora_config.task_type not CAUSAL_LM.")
        if merged_conf.training.get("lr_scheduler_type") == "cosine_with_min_lr":
              lr_kwargs = merged_conf.training.get("lr_scheduler_kwargs")
              if lr_kwargs is None or lr_kwargs.get("min_lr") is None: logger.warning("lr_kwargs.min_lr missing.")
              elif not isinstance(lr_kwargs.min_lr, (float, int)) or lr_kwargs.min_lr < 0.0: raise ValueError("Invalid min_lr.")
              elif lr_kwargs.min_lr >= merged_conf.training.learning_rate: logger.warning(f"min_lr >= learning_rate.")
        # --- Rename evaluation_strategy key if present ---
        if "evaluation_strategy" in merged_conf.training and "eval_strategy" not in merged_conf.training:
             logger.warning("Found 'evaluation_strategy' in config, renaming to 'eval_strategy' for TrainingArguments.")
             merged_conf.training.eval_strategy = merged_conf.training.evaluation_strategy
             del merged_conf.training.evaluation_strategy
        # --- Remove eval_steps if eval_strategy is "no" ---
        if merged_conf.training.get("eval_strategy") == "no" and "eval_steps" in merged_conf.training:
             logger.info("Removing 'eval_steps' as 'eval_strategy' is 'no'.")
             del merged_conf.training.eval_steps

        logger.info("Merged configuration validated.")
        # --- End validation ---
        return merged_conf
    except Exception as e: logger.error(f"Error merging/validating config: {e}"); raise

def init_wandb(cfg: DictConfig):
    # (Implementation remains the same)
    if "wandb" in cfg.training.report_to:
        logger.info("Initializing WandB...")
        try:
            run_name = cfg.wandb.get("run_name", "sft-run")
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
    # (Implementation remains the same - handles random subset)
    logger.info(f"Loading dataset: {cfg.dataset.name} ({cfg.dataset.config_name})")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.config_name, token=cfg.model.access_token)
    logger.info("Formatting prompts...")
    formatted_dataset = dataset.map(lambda x: format_prompt(x, cfg), remove_columns=list(dataset[cfg.dataset.train_split].column_names))
    num_train_samples = cfg.dataset.get("num_train_samples", None); num_eval_samples = cfg.dataset.get("num_eval_samples", None)
    eval_random_subset = cfg.dataset.get("eval_random_subset", False)
    train_split_name = cfg.dataset.train_split; eval_split_name = cfg.dataset.eval_split
    selected_train_dataset = formatted_dataset[train_split_name]
    if num_train_samples is not None and num_train_samples > 0:
        logger.info(f"Selecting first {num_train_samples} samples from training split '{train_split_name}'.")
        select_range = range(min(num_train_samples, len(selected_train_dataset))); selected_train_dataset = selected_train_dataset.select(select_range)
    full_eval_dataset = formatted_dataset[eval_split_name]; selected_eval_dataset = full_eval_dataset
    if num_eval_samples is not None and num_eval_samples > 0:
        actual_num_eval_samples = min(num_eval_samples, len(full_eval_dataset))
        if actual_num_eval_samples < len(full_eval_dataset):
            if eval_random_subset:
                logger.info(f"Selecting {actual_num_eval_samples} random samples from eval split '{eval_split_name}' using seed {cfg.training.seed}.")
                selected_eval_dataset = full_eval_dataset.shuffle(seed=cfg.training.seed).select(range(actual_num_eval_samples))
            else:
                logger.info(f"Selecting first {actual_num_eval_samples} samples from eval split '{eval_split_name}'.")
                select_range = range(actual_num_eval_samples); selected_eval_dataset = full_eval_dataset.select(select_range)
        else: logger.info(f"num_eval_samples >= dataset size. Using full evaluation split.")
    raw_eval_dataset_for_generation = selected_eval_dataset
    logger.info("Tokenizing selected dataset splits...")
    max_seq_length = cfg.dataset.max_seq_length
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy(); return tokenized_output
    final_train_dataset = selected_train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "prompt", "ground_truth_answer"])
    final_trainer_eval_dataset = selected_eval_dataset.map(tokenize_function, batched=True, remove_columns=["text", "prompt", "ground_truth_answer"])
    logger.info(f"Dataset loaded. Final Train size: {len(final_train_dataset)}, Final Eval size: {len(raw_eval_dataset_for_generation)}")
    return {"train": final_train_dataset, "eval_for_trainer": final_trainer_eval_dataset, "raw_eval_for_generation": raw_eval_dataset_for_generation}

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
    # (Implementation updated previously for tqdm - remains the same)
    eval_type = "Base Model" if is_base_model_eval else "Fine-Tuned Model"
    logger.info(f"--- Starting GSM8K evaluation for {eval_type} ---"); model.eval()
    if "prompt" not in dataset.column_names or "ground_truth_answer" not in dataset.column_names:
         logger.error("Evaluation dataset missing required columns."); return {"gsm8k_exact_match": 0.0}

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

    generation_kwargs = {"max_new_tokens": cfg.evaluation.max_new_tokens, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "temperature": cfg.evaluation.temperature, "do_sample": cfg.evaluation.do_sample}
    if generation_kwargs["pad_token_id"] is None: generation_kwargs["pad_token_id"] = tokenizer.eos_token_id; logger.warning(f"pad_token_id is None, setting to eos_token_id ({tokenizer.eos_token_id})")
    if generation_kwargs["eos_token_id"] is None: logger.error("eos_token_id is None.")

    local_predictions = []
    if start_index < end_index:
        progress_bar = tqdm(
            range(start_index, end_index), desc=f"Eval P{process_index} {eval_type}",
            position=process_index, disable=not accelerator.is_local_main_process, leave=False
        )
        for i in progress_bar:
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

    logger.info(f"Process {process_index}: Finished generation for {eval_type}. Gathering results...")
    all_predictions_gathered = accelerator.gather_for_metrics(local_predictions)

    exact_match_count = 0; total_count = 0; results = {}
    if accelerator.is_main_process:
        logger.info(f"Main process calculating Exact Match accuracy for {eval_type}...")
        all_predictions = all_predictions_gathered
        actual_eval_size = len(dataset)
        if len(all_predictions) != actual_eval_size:
             logger.warning(f"Mismatch! Gathered predictions ({len(all_predictions)}) != eval dataset size ({actual_eval_size}) for {eval_type}.")
             eval_limit = min(len(all_predictions), actual_eval_size); total_count = eval_limit
             logger.warning(f"Calculating accuracy based on {eval_limit} samples.")
        else: total_count = actual_eval_size; eval_limit = actual_eval_size
        for i in range(eval_limit):
            completion = all_predictions[i]; pred_answer = extract_gsm8k_answer(completion)
            true_answer = extract_gsm8k_answer(ground_truths[i])
            if pred_answer is not None and true_answer is not None:
                try:
                    if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6: exact_match_count += 1
                except ValueError:
                    if pred_answer == true_answer: exact_match_count += 1
            elif pred_answer == true_answer: exact_match_count += 1
        accuracy = (exact_match_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"GSM8K Evaluation Results ({eval_type}): Exact Match = {accuracy:.2f}% ({exact_match_count}/{total_count})"); results = {"gsm8k_exact_match": accuracy}
        # Log with specific prefix based on eval type
        log_key = f"eval/{'base_model' if is_base_model_eval else 'sft_model'}_gsm8k_exact_match"
        # Check if this is the post-training base eval to give it a unique key
        # We need a way to signal this... modify the function slightly or check caller?
        # Let's add another flag for simplicity, although it makes the signature longer.
        # Alternative: Check a global state? Bad practice. Pass a prefix? Better.
        # Let's stick to the flag for now as it's explicit.
        # --> Re-evaluating: Adding another flag is clumsy. Let's log base model eval only once (before training)
        # and the post-training base eval with a distinct key manually after the call.
        if wandb.run:
             wandb.log({log_key: accuracy}) # Log standard base/sft results here

    accelerator.wait_for_everyone(); model.train(); return results


# --- Training ---

def train_model(cfg: DictConfig): # Takes the merged config
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
        # Store original model name for later reloading
        original_model_name = cfg.model.name
        model, tokenizer = load_model_and_tokenizer(cfg)

    # Load Data
    with accelerator.main_process_first():
        datasets = load_and_prepare_data(cfg, tokenizer)
    train_dataset = datasets["train"]
    trainer_eval_dataset = datasets["eval_for_trainer"] # Used only if eval_strategy != "no"
    raw_eval_dataset_for_generation = datasets["raw_eval_for_generation"]

    # --- Evaluate Base Model (BEFORE Training) ---
    logger.info("--- Evaluating Base Model (Before Any Training/Adapters) ---")
    model_for_eval = model
    eval_model_base = accelerator.prepare(model_for_eval)
    base_model_metrics = evaluate_gsm8k(
        eval_model_base, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=True
    )
    # Base model metrics are logged inside evaluate_gsm8k with "eval/base_model_" prefix
    del eval_model_base
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    # --- End Base Model Evaluation ---

    # --- Apply PEFT if configured ---
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

    # --- Set up Training Arguments ---
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    if "evaluation_strategy" in training_args_dict and "eval_strategy" not in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")
        logger.info("Renamed config key 'evaluation_strategy' to 'eval_strategy'.")
    # Ensure eval_steps is removed if eval_strategy is "no"
    if training_args_dict.get("eval_strategy") == "no" and "eval_steps" in training_args_dict:
         del training_args_dict["eval_steps"]
    # Handle lr_scheduler_kwargs
    if "lr_scheduler_kwargs" in training_args_dict and training_args_dict["lr_scheduler_kwargs"] is not None:
         if not isinstance(training_args_dict["lr_scheduler_kwargs"], dict):
             try: training_args_dict["lr_scheduler_kwargs"] = dict(training_args_dict["lr_scheduler_kwargs"])
             except (TypeError, ValueError): logger.error("Could not convert lr_scheduler_kwargs."); del training_args_dict["lr_scheduler_kwargs"]
         if training_args_dict.get("lr_scheduler_type") == "cosine_with_min_lr":
              if "min_lr" not in training_args_dict["lr_scheduler_kwargs"]: logger.warning("min_lr not found in lr_scheduler_kwargs.")
    if "min_lr" in training_args_dict: del training_args_dict["min_lr"]

    logger.info("Initializing Training Arguments...")
    training_args = TrainingArguments(**training_args_dict)

    # --- Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # Pass eval dataset even if strategy is "no", Trainer might use it internally for something? Or set to None? Let's pass it.
        eval_dataset=trainer_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # --- Train ---
    logger.info(f"--- Starting Fine-Tuning ({tuning_method}) ---")
    try:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")
        metrics = train_result.metrics
        # Log training metrics manually at the end if needed (since intermediate logging might be less frequent now)
        if accelerator.is_main_process:
             trainer.log_metrics("train", metrics) # Log final train metrics
             trainer.save_metrics("train", metrics)
             trainer.save_state()
             if is_wandb_initialized and wandb.run:
                  final_metrics = {"train/final_" + k: v for k,v in metrics.items()}
                  wandb.log(final_metrics)

            # --- Save Model/Adapter ---
             logger.info("Saving final model/adapter...")
             final_save_path = os.path.join(training_args.output_dir, f"final_checkpoint")
             os.makedirs(final_save_path, exist_ok=True)
             trainer.save_model(final_save_path) # Handles PEFT adapters automatically
             tokenizer.save_pretrained(final_save_path)
             try: OmegaConf.save(cfg, os.path.join(final_save_path, "training_config_merged.yaml"))
             except Exception as e_save: logger.error(f"Failed to save merged training config: {e_save}")
             logger.info(f"Final model/adapter saved to {final_save_path}")
             # --- End Saving ---

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if accelerator.is_main_process: logger.info("Attempting to save state due to error..."); trainer.save_state()
        raise

    # --- Evaluate Fine-Tuned Model (POST Training) ---
    accelerator.wait_for_everyone()
    logger.info(f"--- Evaluating Fine-Tuned Model ({tuning_method}) ---")
    # Use the model state from the trainer after training
    eval_model_sft = accelerator.prepare(trainer.model)
    sft_model_metrics = evaluate_gsm8k(
        eval_model_sft, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=False
    )
    # SFT model metrics are logged inside evaluate_gsm8k with "eval/sft_model_" prefix
    del eval_model_sft # Clean up wrapped model
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    # --- End SFT Model Evaluation ---


    # --- Evaluate Base Model AGAIN (POST Training) ---
    logger.info("--- Evaluating Base Model AGAIN (Post-Training Comparison) ---")
    # Reload the original base model to ensure it's untuned
    # Use try-except for robustness
    original_base_model = None # Define outside try
    eval_original_base_model = None
    try:
        with accelerator.main_process_first():
             # Reload base using original config details stored earlier
             logger.info(f"Reloading original base model: {original_model_name}")
             original_base_model, _ = load_model_and_tokenizer(cfg) # Reload base

        # Prepare the reloaded base model for evaluation
        eval_original_base_model = accelerator.prepare(original_base_model)

        # Evaluate using the same function, marking it as base model eval
        post_train_base_metrics = evaluate_gsm8k(
            eval_original_base_model, tokenizer, raw_eval_dataset_for_generation, cfg, accelerator, is_base_model_eval=True # Mark as base eval
        )
        # Log manually with a distinct key
        if accelerator.is_main_process:
             logger.info(f"Base Model Evaluation Metrics (Post-Training): {post_train_base_metrics}")
             if is_wandb_initialized and wandb.run:
                  wandb.log({"eval/post_train_base_model_gsm8k_exact_match": post_train_base_metrics.get("gsm8k_exact_match", 0)})

    except Exception as e_reload:
         logger.error(f"Failed to reload and evaluate original base model post-training: {e_reload}", exc_info=True)
    finally:
         # Clean up memory
         del original_base_model, eval_original_base_model
         accelerator.wait_for_everyone()
         torch.cuda.empty_cache()
    # --- End Post-Training Base Model Evaluation ---


    # --- Clean Up WandB ---
    if is_wandb_initialized and wandb.run and accelerator.is_main_process:
        logger.info("Finishing WandB run..."); wandb.finish()

    logger.info("Script finished successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script with Merged Config")
    parser.add_argument("--config", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    args = parser.parse_args()
    config = load_config(override_config_path=args.config, base_config_path=args.base_config)
    train_model(config)
