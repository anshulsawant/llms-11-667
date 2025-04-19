"""Script for evaluating base or fine-tuned models based on config."""

import os
import logging
import re
import math
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf, DictConfig, ListConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed # For reproducible shuffling
)
from peft import PeftModel # Needed for loading adapters
from tqdm.auto import tqdm

# --- Import from utils ---
from .utils import (
    load_config,
    # init_wandb, # Optional: Maybe log eval results separately?
    load_model_and_tokenizer, # Now expects only cfg
    format_prompt,
    extract_gsm8k_answer
)
# --- End imports ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a base or fine-tuned model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the override configuration YAML file defining the model and eval settings.")
    parser.add_argument("--base_config_path", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    parser.add_argument("--use_base_model", action="store_true", help="Force evaluation using the base model specified in the config, ignoring checkpoints.")
    # Removed superfluous arguments: --eval_split, --dataset_config_name, --num_eval_samples, --eval_random_subset
    parser.add_argument("--output_file", type=Path, default=None, help="Optional path to save evaluation metrics as JSON.")
    return parser.parse_args()


def load_and_prepare_eval_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Loads and prepares the evaluation dataset subset based *only* on config."""
    # Get dataset parameters directly from config
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset.config_name
    eval_split = cfg.dataset.eval_split
    num_samples = cfg.dataset.get("num_eval_samples", None)
    random_subset = cfg.dataset.get("eval_random_subset", False)
    seed = cfg.training.seed # Use training seed for reproducibility

    logger.info(f"Loading evaluation dataset: {dataset_name} ({dataset_config}), split: {eval_split}")
    # Load the specified evaluation split
    full_eval_dataset_raw = load_dataset(
        dataset_name,
        dataset_config,
        split=eval_split,
        token=cfg.model.get("access_token", None)
    )

    logger.info("Formatting evaluation prompts...")
    formatted_eval_dataset = full_eval_dataset_raw.map(
        lambda x: format_prompt(x, cfg),
        remove_columns=list(full_eval_dataset_raw.column_names)
    )

    # Select subset if specified in config
    selected_eval_dataset = formatted_eval_dataset
    if num_samples is not None and num_samples > 0:
        actual_num_samples = min(num_samples, len(formatted_eval_dataset))
        if actual_num_samples < len(formatted_eval_dataset):
            if random_subset:
                logger.info(f"Selecting {actual_num_samples} random samples from eval split '{eval_split}' using seed {seed}.")
                selected_eval_dataset = formatted_eval_dataset.shuffle(seed=seed).select(range(actual_num_samples))
            else:
                logger.info(f"Selecting first {actual_num_samples} samples from eval split '{eval_split}'.")
                selected_eval_dataset = formatted_eval_dataset.select(range(actual_num_samples))
        else:
             logger.info(f"num_eval_samples >= dataset size. Using full evaluation split.")

    # No tokenization needed here, evaluate_gsm8k works on raw prompts
    logger.info(f"Evaluation dataset prepared. Final size: {len(selected_eval_dataset)}")
    return selected_eval_dataset


# --- Moved evaluate_gsm8k function here ---
@torch.no_grad()
def evaluate_gsm8k( model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Any, cfg: DictConfig, accelerator: Accelerator, is_base_model_eval: bool = False) -> Dict[str, float]:
    """
    Evaluates the model on the GSM8K dataset using generation and exact match.
    (Implementation is the same as the final version in sft_script.py, uses tqdm, handles base model instructions/few-shot)
    """
    eval_type = "Base Model" if is_base_model_eval else "Fine-Tuned Model" # Label depends on how model was loaded
    logger.info(f"--- Starting GSM8K evaluation for {eval_type} ---"); model.eval()
    if "prompt" not in dataset.column_names or "ground_truth_answer" not in dataset.column_names:
         logger.error("Evaluation dataset missing required columns."); return {"gsm8k_exact_match": 0.0}

    prompts_base = dataset["prompt"]; ground_truths = dataset["ground_truth_answer"]

    instruction_text = ""
    one_shot_text = ""
    if is_base_model_eval: # Apply instructions/few-shot only if evaluating the base model explicitly
        instruction_text = "You are a helpful math assistant... format '#### <number>'.\n\n" # Shortened for brevity
        logger.info("Prepending explicit instructions for base model evaluation.")
        strategy = cfg.evaluation.get("base_model_prompt_strategy", "zero_shot"); logger.info(f"Base model eval strategy: {strategy}")
        if strategy == "one_shot":
            one_shot_example = cfg.evaluation.get("one_shot_example")
            if one_shot_example and one_shot_example.get("question") and one_shot_example.get("answer"):
                one_shot_q = one_shot_example.question; one_shot_a = one_shot_example.answer
                one_shot_text = f"Question: {one_shot_q}\nAnswer: {one_shot_a}\n\n"; logger.info("Using one-shot example.")
            else: logger.warning("One-shot requested but example invalid.")
        elif strategy != "zero_shot": logger.warning(f"Unknown strategy '{strategy}'.")

    num_processes = accelerator.num_processes; process_index = accelerator.process_index
    samples_per_process = (len(prompts_base) + num_processes - 1) // num_processes
    start_index = process_index * samples_per_process; end_index = min(start_index + samples_per_process, len(prompts_base))

    # Use generation params from evaluation section of config
    gen_cfg = cfg.get("evaluation", {})
    generation_kwargs = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": gen_cfg.get("temperature", 0.1),
        "do_sample": gen_cfg.get("do_sample", True)
    }
    if generation_kwargs["pad_token_id"] is None: generation_kwargs["pad_token_id"] = tokenizer.eos_token_id; logger.warning(f"pad_token_id is None, setting to eos_token_id ({tokenizer.eos_token_id})")
    if generation_kwargs["eos_token_id"] is None: logger.error("eos_token_id is None.")

    local_predictions = []
    if start_index < end_index:
        progress_bar = tqdm(
            range(start_index, end_index), desc=f"Eval P{process_index} {eval_type}",
            position=process_index, disable=not accelerator.is_local_main_process, leave=False
        )
        for i in progress_bar:
            base_prompt = prompts_base[i]
            final_prompt = instruction_text + one_shot_text + base_prompt
            max_input_length = cfg.dataset.max_seq_length - generation_kwargs["max_new_tokens"]
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
        ground_truths_to_compare = ground_truths[:eval_limit]
        for i in range(eval_limit):
            completion = all_predictions[i]; pred_answer = extract_gsm8k_answer(completion)
            true_answer = extract_gsm8k_answer(ground_truths_to_compare[i])
            if pred_answer is not None and true_answer is not None:
                try:
                    if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6: exact_match_count += 1
                except ValueError:
                    if pred_answer == true_answer: exact_match_count += 1
            elif pred_answer == true_answer: exact_match_count += 1
        accuracy = (exact_match_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"GSM8K Evaluation Results ({eval_type}): Exact Match = {accuracy:.2f}% ({exact_match_count}/{total_count})"); results = {"gsm8k_exact_match": accuracy}
        # No WandB logging here by default, handled by caller if needed

    accelerator.wait_for_everyone(); model.train(); return results
# --- End evaluate_gsm8k function ---


def main():
    """Main function for evaluation."""
    args = parse_arguments()
    logger.info("Starting evaluation script...")
    logger.info(f"CLI Arguments: {vars(args)}")

    try:
        cfg = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        logger.info(f"Successfully loaded config from: {args.config_path} (overriding {args.base_config_path})")
    except Exception as e: logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load base model and tokenizer using the utility function
    try:
        logger.info("Loading base model and tokenizer using config...")
        # --- Call changed to only use cfg ---
        model, tokenizer = load_model_and_tokenizer(cfg)
        logger.info("Base model and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Exiting due to base model/tokenizer loading failure: {e}", exc_info=True)
        sys.exit(1)

    # --- Logic to load adapter or full checkpoint AFTER base model is loaded ---
    is_base_model_eval = args.use_base_model # Check flag

    if not is_base_model_eval:
        tuning_method = cfg.get("tuning_method", "full")
        output_dir_from_config = Path(cfg.training.output_dir)
        expected_checkpoint_dir = output_dir_from_config / "final_checkpoint"
        logger.info(f"Attempting to load fine-tuned model (method: {tuning_method}) from checkpoint dir: {expected_checkpoint_dir}")

        try:
            if tuning_method == "lora":
                adapter_load_path = str(expected_checkpoint_dir)
                logger.info(f"Loading LoRA adapter from: '{adapter_load_path}'")
                if not os.path.isdir(adapter_load_path):
                    logger.error(f"LoRA adapter path does not exist: {adapter_load_path}")
                    sys.exit(1)
                # Load the PEFT adapter onto the base model
                model = PeftModel.from_pretrained(model, adapter_load_path)
                logger.info("LoRA adapter loaded successfully.")

            elif tuning_method == "full":
                checkpoint_path = str(expected_checkpoint_dir)
                logger.info(f"Loading Full SFT checkpoint from: '{checkpoint_path}'")
                if not os.path.isdir(checkpoint_path):
                    logger.error(f"Full SFT checkpoint path does not exist: {checkpoint_path}")
                    sys.exit(1)

                # Need to reload the model entirely from the checkpoint
                # Reconstruct model kwargs from config (similar to utils.py)
                access_token = cfg.model.get("access_token", None)
                trust_remote_code = cfg.model.get("trust_remote_code", False)
                model_kwargs = {"trust_remote_code": trust_remote_code, "token": access_token}
                attn_impl = cfg.model.get("attn_implementation", None)
                if attn_impl: model_kwargs["attn_implementation"] = attn_impl
                precision = "bf16" if cfg.training.get("bf16") else "fp16" if cfg.training.get("fp16") else "fp32"
                if precision == "bf16": model_kwargs["torch_dtype"] = torch.bfloat16
                elif precision == "fp16": model_kwargs["torch_dtype"] = torch.float16

                # Replace the previously loaded base model
                del model # Release memory if possible
                torch.cuda.empty_cache() # Try to clear cache
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
                logger.info("Full SFT model loaded successfully from checkpoint.")
                # Tokenizer should remain the same as the base model's

            else:
                logger.error(f"Unknown tuning_method '{tuning_method}'. Cannot load fine-tuned model."); sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model/adapter (method: {tuning_method}): {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info(f"Evaluating BASE model specified in config: '{cfg.model.name}'")


    # Load Evaluation Data
    try:
        # Pass tokenizer loaded either from base or potentially reloaded for full SFT (though usually same)
        eval_dataset = load_and_prepare_eval_data(cfg, tokenizer)
    except Exception as e: logger.error(f"Exiting due to eval data loading failure: {e}", exc_info=True); sys.exit(1)

    # Prepare model with accelerator
    # This needs to happen AFTER potential adapter/checkpoint loading
    logger.info("Preparing model with Accelerator...")
    model = accelerator.prepare(model)
    logger.info("Model prepared.")

    # Run evaluation
    logger.info("Starting evaluation...")
    eval_metrics = evaluate_gsm8k(
        model, tokenizer, eval_dataset, cfg, accelerator, is_base_model_eval=is_base_model_eval
    )

    # Save results if output file specified
    if args.output_file and accelerator.is_main_process:
        logger.info(f"Saving evaluation metrics to: {args.output_file}")
        try:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metrics to {args.output_file}: {e}")

    logger.info("Evaluation script finished successfully.")

if __name__ == "__main__":
    main()
