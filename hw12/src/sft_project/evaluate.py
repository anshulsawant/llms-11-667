# src/sft_project/evaluate.py
"""Script for evaluating base or fine-tuned models based on config."""

import os
import logging
import argparse
import json
import re # Import re for slugify
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys
import math # For accuracy calculation if needed beyond exact match

# --- Setup logger FIRST ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End logger setup ---

try:
    import torch
    import transformers
    from accelerate import Accelerator
    from datasets import load_dataset, Dataset # Need load_dataset here
    from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed # For reproducible shuffling
    )
    from peft import PeftModel
    from tqdm.auto import tqdm

    # --- Import from utils ---
    from .utils import (
        load_config,
        load_model_and_tokenizer,
        format_prompt,
        extract_gsm8k_answer # Specific to GSM8K evaluation
        # read_jsonl, write_jsonl are not typically needed for eval metrics saving
    )
    # --- End imports ---

except ImportError as e:
    logger.error(f"Failed to import required libraries or functions from .utils: {e}. Make sure all dependencies are installed and utils.py is correct and accessible.")
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a base or fine-tuned model using config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config_path", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    parser.add_argument("--use_base_model", action="store_true", help="Force evaluation using the base model specified in the config, ignoring checkpoints.")
    # Removed CLI overrides for dataset params, use config only
    return parser.parse_args()

def slugify(value: str) -> str:
    """Normalizes string, removes invalid chars, and converts spaces to hyphens."""
    if not isinstance(value, str): value = str(value) # Ensure string type
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    if not value: return "na" # Handle empty slugs
    return value

def load_and_prepare_eval_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Loads and prepares the evaluation dataset subset based *only* on config."""
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset.config_name
    eval_split = cfg.dataset.eval_split
    num_samples = cfg.dataset.get("num_eval_samples", None)
    random_subset = cfg.dataset.get("eval_random_subset", False)
    seed = cfg.training.seed

    logger.info(f"Loading evaluation dataset: {dataset_name} ({dataset_config}), split: {eval_split}")
    try:
        full_eval_dataset_raw = load_dataset(
            dataset_name,
            dataset_config,
            split=eval_split,
            token=cfg.model.get("access_token", None)
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}/{dataset_config} split {eval_split}: {e}", exc_info=True)
        raise # Re-raise after logging

    logger.info("Formatting evaluation prompts...")
    # Ensure format_prompt returns ground truth needed for comparison
    try:
         # Assuming format_prompt adds 'prompt' and 'ground_truth_answer' keys
        formatted_eval_dataset = full_eval_dataset_raw.map(
            lambda x: format_prompt(x, cfg),
            remove_columns=[col for col in full_eval_dataset_raw.column_names if col not in ['prompt', 'ground_truth_answer']] # Keep only needed cols
        )
        # Verify required columns exist after formatting
        if "prompt" not in formatted_eval_dataset.column_names or \
           "ground_truth_answer" not in formatted_eval_dataset.column_names:
            logger.error("Formatted dataset missing 'prompt' or 'ground_truth_answer' column. Check `format_prompt` in utils.py.")
            raise ValueError("Formatted dataset missing required columns.")

    except Exception as e:
        logger.error(f"Error during prompt formatting for evaluation: {e}", exc_info=True)
        raise # Re-raise after logging


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

    logger.info(f"Evaluation dataset prepared. Final size: {len(selected_eval_dataset)}")
    return selected_eval_dataset


@torch.no_grad()
def run_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    cfg: DictConfig,
    accelerator: Accelerator,
    is_base_model_eval: bool = False
) -> Dict[str, Any]:
    """
    Runs evaluation on the provided dataset, calculates metrics.
    Specifically tailored for GSM8K exact match, but adaptable.
    """
    model.eval()
    eval_results_list = [] # Store individual results if needed for detailed logging

    # --- Get Generation Parameters (Using evaluation section primarily) ---
    gen_cfg = cfg.get("evaluation", cfg.get("inference", {})) # Prioritize evaluation settings
    generation_kwargs = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": gen_cfg.get("temperature", 0.1),
        "do_sample": gen_cfg.get("do_sample", True)
    }
    if generation_kwargs["pad_token_id"] is None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        logger.warning(f"pad_token_id is None, setting to eos_token_id ({tokenizer.eos_token_id})")
    if generation_kwargs["eos_token_id"] is None:
        logger.error("eos_token_id is None. Generation might not stop correctly.")
    logger.info(f"Using evaluation generation parameters: {generation_kwargs}")

    # --- Prepare base prompts from dataset ---
    try:
        base_prompts = eval_dataset["prompt"]
        ground_truths = eval_dataset["ground_truth_answer"]
    except KeyError:
         logger.error("Evaluation dataset missing 'prompt' or 'ground_truth_answer' column.", exc_info=True)
         return {"error": "Missing required dataset columns"}

    # --- Evaluation Loop ---
    local_results = [] # Store dicts: {'prediction': str, 'ground_truth': str, 'is_correct': bool}
    progress_bar = tqdm(total=len(base_prompts), desc="Running Evaluation", disable=not accelerator.is_local_main_process)

    for i, base_prompt_text in enumerate(base_prompts):
        instruction_text = ""
        few_shot_text = ""
        final_prompt_text = base_prompt_text

        if is_base_model_eval:
            # Use evaluation config section for base model instructions/few-shot
            instruction_text = cfg.evaluation.get("base_model_instruction", "You are a helpful assistant.")
            if instruction_text: instruction_text += "\n\n"
            strategy = cfg.evaluation.get("base_model_prompt_strategy", "zero_shot")
            if strategy == "one_shot":
                one_shot_example = cfg.evaluation.get("one_shot_example", None)
                if one_shot_example and isinstance(one_shot_example, DictConfig) and one_shot_example.get("question") and one_shot_example.get("answer"):
                    q = one_shot_example.question; a = one_shot_example.answer
                    few_shot_text = f"Question: {q}\nAnswer: {a}\n\n"
                    if i == 0: logger.info("Using one-shot example for base model evaluation.")
                elif i == 0: logger.warning("Eval one-shot requested but 'one_shot_example' invalid/missing.")
            elif strategy != "zero_shot" and i == 0: logger.warning(f"Unknown eval strategy '{strategy}'. Using zero-shot.")
            final_prompt_text = instruction_text + few_shot_text + base_prompt_text

        buffer = 10
        max_model_len = int(cfg.dataset.get('max_seq_length', 2048))
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 256))
        if max_new_tokens >= max_model_len:
             if i == 0: logger.warning(f"Eval max_new_tokens >= max_model_len. Adjusting.")
             max_new_tokens = max_model_len // 2
        max_input_length = max_model_len - max_new_tokens - buffer

        if max_input_length <= 0:
            logger.error(f"Eval input length calc failed (<=0) sample {i}. Skipping."); prediction = "[ERROR: Input length calculation failed]"; is_correct = False
            local_results.append({'prediction': prediction, 'ground_truth': ground_truths[i], 'is_correct': is_correct}); progress_bar.update(1); continue

        try:
            inputs = tokenizer(final_prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=int(max_input_length))
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        except OverflowError as oe:
             logger.error(f"Eval Tokenization OverflowError sample {i}. Error: {oe}", exc_info=True); prediction = "[ERROR: Tokenization Overflow]"; is_correct = False
             local_results.append({'prediction': prediction, 'ground_truth': ground_truths[i], 'is_correct': is_correct}); progress_bar.update(1); continue
        except Exception as e:
             logger.error(f"Eval Tokenization error sample {i}: {e}", exc_info=True); prediction = "[ERROR: Tokenization Failed]"; is_correct = False
             local_results.append({'prediction': prediction, 'ground_truth': ground_truths[i], 'is_correct': is_correct}); progress_bar.update(1); continue

        try:
            outputs = model.generate(**inputs, **generation_kwargs)
            completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(completion_tokens, skip_special_tokens=True)

            # --- Task-specific Post-processing and Comparison (GSM8K Example) ---
            pred_answer = extract_gsm8k_answer(prediction)
            true_answer = extract_gsm8k_answer(ground_truths[i]) # Extract from ground truth as well for fair comparison
            is_correct = False
            if pred_answer is not None and true_answer is not None:
                try: # Compare numerically if possible
                    if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6:
                        is_correct = True
                except ValueError: # Fallback to string comparison if not numbers
                    if pred_answer == true_answer:
                        is_correct = True
            elif pred_answer == true_answer: # Handle cases where both might be None or identical strings
                 is_correct = True
            # --- End Task-specific ---

            local_results.append({'prediction': prediction, 'ground_truth': ground_truths[i], 'is_correct': is_correct})

        except Exception as e:
            logger.error(f"Eval Generation error sample {i}: {e}", exc_info=False); prediction = "[ERROR: Generation Failed]"; is_correct = False
            local_results.append({'prediction': prediction, 'ground_truth': ground_truths[i], 'is_correct': is_correct})

        progress_bar.update(1)

    progress_bar.close()

    # --- Aggregate Metrics ---
    # Gather results from all processes
    logger.info(f"Process {accelerator.process_index}: Finished evaluation loop. Gathering results...")
    all_results_gathered = accelerator.gather_for_metrics(local_results)
    logger.info(f"Process {accelerator.process_index}: Results gathered.")


    final_metrics = {}
    if accelerator.is_main_process:
        logger.info("Main process calculating final metrics...")
        correct_count = 0
        total_count = len(all_results_gathered)

        if total_count == 0:
             logger.warning("No evaluation results gathered. Cannot calculate metrics.")
             return {"error": "No results gathered"}

        # Check for size mismatch (might happen with uneven distribution)
        if total_count != len(eval_dataset):
             logger.warning(f"Mismatch! Gathered results ({total_count}) != eval dataset size ({len(eval_dataset)}). Using gathered count.")

        for result in all_results_gathered:
            if result.get('is_correct', False): # Safely check the 'is_correct' flag
                correct_count += 1

        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        final_metrics = {
            "exact_match_accuracy": accuracy,
            "correct_count": correct_count,
            "total_samples": total_count
            # Add other metrics if needed
        }
        logger.info(f"Evaluation Metrics: {final_metrics}")
        # Optionally add detailed results if needed for analysis
        # final_metrics["detailed_results"] = all_results_gathered

    # Ensure all processes wait before returning results from main process
    accelerator.wait_for_everyone()

    return final_metrics


def main():
    """Main function for evaluation."""
    args = parse_arguments()
    logger.info("Starting evaluation script...")
    logger.info(f"CLI Arguments: {vars(args)}")

    try:
        cfg = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        logger.info(f"Successfully loaded config from: {args.config_path} (overriding {args.base_config_path})")
    except Exception as e: logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)

    accelerator = Accelerator()

    try:
        logger.info("Loading base model and tokenizer using config...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        logger.info("Base model and tokenizer loaded.")
    except Exception as e: logger.error(f"Exiting due to base model/tokenizer loading failure: {e}", exc_info=True); sys.exit(1)

    is_base_model_eval = args.use_base_model
    model_type_str = "base"

    if not is_base_model_eval:
        tuning_method = cfg.get("tuning_method", "full")
        model_type_str = slugify(tuning_method)
        output_dir_from_config = Path(cfg.training.output_dir)
        expected_checkpoint_dir = output_dir_from_config / "final_checkpoint"
        logger.info(f"Attempting to load fine-tuned model (method: {tuning_method}) from checkpoint dir: {expected_checkpoint_dir}")
        try:
            if tuning_method == "lora":
                adapter_load_path = str(expected_checkpoint_dir)
                if not os.path.isdir(adapter_load_path): logger.error(f"LoRA path missing: {adapter_load_path}"); sys.exit(1)
                model = PeftModel.from_pretrained(model, adapter_load_path); logger.info("LoRA adapter loaded.")
            elif tuning_method == "full":
                checkpoint_path = str(expected_checkpoint_dir)
                if not os.path.isdir(checkpoint_path): logger.error(f"Full SFT path missing: {checkpoint_path}"); sys.exit(1)
                access_token = cfg.model.get("access_token", None); trust_remote_code = cfg.model.get("trust_remote_code", False)
                model_kwargs = {"trust_remote_code": trust_remote_code, "token": access_token}
                attn_impl = cfg.model.get("attn_implementation", None);
                if attn_impl: model_kwargs["attn_implementation"] = attn_impl
                precision = "bf16" if cfg.training.get("bf16") else "fp16" if cfg.training.get("fp16") else "fp32"
                if precision == "bf16": model_kwargs["torch_dtype"] = torch.bfloat16
                elif precision == "fp16": model_kwargs["torch_dtype"] = torch.float16
                del model; torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs); logger.info("Full SFT model loaded.")
            else: logger.error(f"Unknown tuning_method '{tuning_method}'."); sys.exit(1)
        except Exception as e: logger.error(f"Failed to load fine-tuned model/adapter: {e}", exc_info=True); sys.exit(1)
    else: logger.info(f"Running evaluation with BASE model: '{cfg.model.name}'")

    try:
        eval_dataset = load_and_prepare_eval_data(cfg, tokenizer)
    except Exception as e: logger.error(f"Exiting due to eval data loading failure: {e}", exc_info=True); sys.exit(1)

    logger.info("Preparing model with Accelerator...")
    # Note: Prepare model *after* loading adapters/checkpoints
    # For evaluation, dataset preparation happens before prepare, which is fine.
    model = accelerator.prepare(model)
    # accelerator.prepare() can also handle DataLoaders if batching is implemented later
    logger.info("Model prepared.")

    logger.info("Starting evaluation process...")
    eval_metrics = run_evaluation(model, tokenizer, eval_dataset, cfg, accelerator, is_base_model_eval)

    # --- Construct Output File Path and Save Metrics ---
    if accelerator.is_main_process:
        output_file_path = None
        try:
            output_dir = Path("./eval_results") # Dedicated folder
            output_dir.mkdir(parents=True, exist_ok=True)

            config_path = Path(args.config_path)
            config_stem = slugify(config_path.stem)
            base_model_name = cfg.model.get("name", "unknown_model")
            base_model_slug = slugify(base_model_name.split('/')[-1])
            dataset_config_name = slugify(cfg.dataset.get("config_name", "unknown_dataset"))
            eval_split = slugify(cfg.dataset.get("eval_split", "unknown_split"))

            output_filename = f"{config_stem}_eval_{eval_split}_{model_type_str}_{base_model_slug}_{dataset_config_name}_metrics.json"
            output_file_path = output_dir / output_filename
            logger.info(f"Constructed evaluation metrics path: {output_file_path}")

        except Exception as e:
            logger.error(f"Failed to construct output file path for evaluation metrics: {e}", exc_info=True)

        # Save metrics if available and path is valid
        if eval_metrics and "error" not in eval_metrics and output_file_path:
            logger.info(f"Saving evaluation metrics to: {output_file_path}")
            try:
                with open(output_file_path, 'w') as f:
                    json.dump(eval_metrics, f, indent=4)
                logger.info("Evaluation metrics saved successfully.")
            except Exception as e:
                logger.error(f"Failed to write evaluation metrics to {output_file_path}: {e}", exc_info=True)
        elif not eval_metrics or "error" in eval_metrics:
            logger.warning("Evaluation metrics were not generated or contained errors. Output file not created.")
        else: # Metrics exist but path is None
             logger.error("Output file path could not be constructed. Evaluation metrics were not saved.")

    accelerator.wait_for_everyone()
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()
