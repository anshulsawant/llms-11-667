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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
# --- End logger setup ---

try:
    import torch
    from torch.utils.data import DataLoader # Import DataLoader
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed, # For reproducible shuffling
        DataCollatorWithPadding # Use basic padding collator for inputs
    )
    from datasets import load_dataset, Dataset
    from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
    from accelerate import Accelerator
    from peft import PeftModel
    from tqdm.auto import tqdm

    # --- Import from utils ---
    from .utils import (
        load_config,
        load_model_and_tokenizer,
        format_prompt,
        extract_gsm8k_answer, # Specific to GSM8K evaluation
        slugify
    )
    # --- End imports ---

except ImportError as e:
    logger.error(f"Failed to import required libraries or functions from .utils: {e}.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a base or fine-tuned model using config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config_path", type=str, default="configs/config.yaml", help="Path to the base configuration YAML file.")
    parser.add_argument("--use_base_model", action="store_true", help="Force evaluation using the base model specified in the config, ignoring checkpoints.")
    # Removed unused CLI overrides for dataset params - use config file's `dataset` section
    return parser.parse_args()

# slugify function remains the same as before
def slugify(value: str) -> str:
    """Normalizes string, removes invalid chars, and converts spaces to hyphens."""
    if not isinstance(value, str): value = str(value) # Ensure string type
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    if not value: return "na" # Handle empty slugs
    return value

def load_and_prepare_eval_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Loads and prepares the evaluation dataset subset based *only* on config."""
    # --- Dataset Loading ---
    dataset_name = cfg.dataset.name
    dataset_config_name = cfg.dataset.config_name
    eval_split = cfg.dataset.eval_split
    num_samples = cfg.dataset.get("num_eval_samples", None)
    random_subset = cfg.dataset.get("eval_random_subset", False)
    seed = cfg.training.seed # Use training seed for reproducibility here too

    logger.info(f"Loading evaluation dataset: {dataset_name} ({dataset_config_name}), split: {eval_split}")
    try:
        full_eval_dataset_raw = load_dataset(
            dataset_name,
            dataset_config_name,
            split=eval_split,
            token=cfg.model.get("access_token", None)
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}/{dataset_config_name} split {eval_split}: {e}", exc_info=True)
        raise

    # --- Select subset ---
    selected_eval_dataset = full_eval_dataset_raw
    if num_samples is not None and num_samples > 0:
        actual_num_samples = min(num_samples, len(full_eval_dataset_raw))
        if actual_num_samples < len(full_eval_dataset_raw):
            if random_subset:
                logger.info(f"Selecting {actual_num_samples} random samples from eval split '{eval_split}' using seed {seed}.")
                selected_eval_dataset = full_eval_dataset_raw.shuffle(seed=seed).select(range(actual_num_samples))
            else:
                logger.info(f"Selecting first {actual_num_samples} samples from eval split '{eval_split}'.")
                selected_eval_dataset = full_eval_dataset_raw.select(range(actual_num_samples))
        else:
             logger.info(f"num_eval_samples >= dataset size. Using full evaluation split.")

    # --- Format prompts ---
    logger.info("Formatting evaluation prompts...")
    try:
        original_columns = selected_eval_dataset.column_names
        formatted_eval_dataset = selected_eval_dataset.map(
            lambda x: format_prompt(x, cfg),
            # Keep original columns temporarily if needed by format_prompt, will remove later
        )
        if "prompt" not in formatted_eval_dataset.column_names or \
           "ground_truth_answer" not in formatted_eval_dataset.column_names:
            logger.error("Formatted dataset missing 'prompt' or 'ground_truth_answer'. Check utils.format_prompt.")
            raise ValueError("Formatted dataset missing required columns.")
        logger.info(f"Formatting complete. Current columns: {formatted_eval_dataset.column_names}")
    except Exception as e:
        logger.error(f"Error during prompt formatting: {e}", exc_info=True)
        raise

    logger.info(f"Evaluation dataset prepared for tokenization. Size: {len(formatted_eval_dataset)}")
    return formatted_eval_dataset


@torch.no_grad()
def run_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset, # Dataset now has 'prompt' and 'ground_truth_answer' + maybe originals
    cfg: DictConfig,
    accelerator: Accelerator,
    is_base_model_eval: bool = False
) -> Dict[str, Any]:
    """
    Runs batched evaluation on the provided dataset, calculates metrics.
    """
    model.eval()
    max_seq_length = cfg.dataset.max_seq_length

    # --- Pre-calculate base model instructions/few-shot (if needed) ---
    instruction_text = ""
    few_shot_text = ""
    if is_base_model_eval:
        instruction_text = cfg.evaluation.get("base_model_instruction", "You are a helpful assistant.")
        if instruction_text: instruction_text += "\n\n"
        strategy = cfg.evaluation.get("base_model_prompt_strategy", "zero_shot")
        if strategy == "one_shot":
            one_shot_example = cfg.evaluation.get("one_shot_example", None)
            if one_shot_example and isinstance(one_shot_example, DictConfig) and one_shot_example.get("question") and one_shot_example.get("answer"):
                q = one_shot_example.question; a = one_shot_example.answer
                few_shot_text = f"Question: {q}\nAnswer: {a}\n\n"
                logger.info("Using one-shot example for base model evaluation.")
            else: logger.warning("Eval one-shot requested but 'one_shot_example' invalid/missing.")
        elif strategy != "zero_shot": logger.warning(f"Unknown eval strategy '{strategy}'. Using zero-shot.")

    # --- Define Preprocessing Function (Prompt Construction + Tokenization) ---
    def preprocess_function(examples):
        final_prompts = []
        for prompt in examples['prompt']:
            prompt_str = str(prompt if prompt is not None else "")
            final_prompt = (instruction_text + few_shot_text + prompt_str) if is_base_model_eval else prompt_str
            final_prompts.append(final_prompt)

        gen_cfg = cfg.get("evaluation", cfg.get("inference", {}))
        max_new_tokens = gen_cfg.get("max_new_tokens", 256)
        buffer = 10
        max_input_length = max_seq_length - max_new_tokens - buffer
        if max_input_length <= 0:
             logger.warning(f"Max sequence length ({max_seq_length}) too small for max_new_tokens ({max_new_tokens}). Setting max_input_length to {max_seq_length // 2}.")
             max_input_length = max_seq_length // 2

        model_inputs = tokenizer(
            final_prompts,
            max_length=max_input_length,
            padding=False, # Collator handles padding
            truncation=True
        )
        # --- FIX: Explicitly include ground_truth_answer in the output ---
        # This ensures it's available after the map removes other columns.
        model_inputs["ground_truth_answer"] = examples["ground_truth_answer"]
        # --- End FIX ---
        return model_inputs

    logger.info("Preprocessing and tokenizing evaluation dataset...")
    # --- FIX: Remove all columns that existed *before* this map step ---
    columns_to_remove = eval_dataset.column_names
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove + ['ground_truth_answer'] # Remove all prior columns
    )
    # --- End FIX ---
    logger.info(f"Tokenization complete. Final columns: {tokenized_eval_dataset.column_names}")
    # Check if expected columns are present
    if not all(col in tokenized_eval_dataset.column_names for col in ['input_ids', 'attention_mask', 'ground_truth_answer']):
         logger.error(f"Tokenized dataset missing required columns for eval. Found: {tokenized_eval_dataset.column_names}")
         raise ValueError("Tokenized dataset missing required columns.")

    # --- Define Data Collator ---
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        pad_to_multiple_of=8
    )
    logger.info("Data collator defined.")

    # --- Create DataLoader ---
    eval_batch_size = cfg.evaluation.get("batch_size", cfg.training.get("per_device_eval_batch_size", 4))
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    logger.info(f"DataLoader created with batch size {eval_batch_size}.")

    # --- Prepare Model and DataLoader ---
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    logger.info("Model and DataLoader prepared with Accelerator.")

    # --- Get Generation Parameters ---
    gen_cfg = cfg.get("evaluation", cfg.get("inference", {}))
    generation_kwargs = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": gen_cfg.get("temperature", 0.1),
        "do_sample": gen_cfg.get("do_sample", True)
    }
    logger.info(f"Using evaluation generation parameters: {generation_kwargs}")

    # --- Evaluation Loop (Batched) ---
    local_results = []
    progress_bar = tqdm(total=len(eval_dataloader), desc="Running Batched Evaluation", disable=not accelerator.is_local_main_process)

    for batch in eval_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # Ground truth should now be correctly passed if it was returned by preprocess_function
        # DataCollatorWithPadding typically passes through columns it doesn't process
        try:
            ground_truths_batch = batch['ground_truth_answer']
        except KeyError:
            logger.error("Ground truth answers ('ground_truth_answer') not found in DataLoader batch.")
            batch_size_fallback = input_ids.shape[0]
            for i in range(batch_size_fallback):
                 local_results.append({'prediction': '[ERROR: GT Missing]', 'ground_truth': 'N/A', 'is_correct': False})
            progress_bar.update(1)
            continue

        try:
            gen_kwargs_batch = generation_kwargs.copy()
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs_batch)
            input_lengths = torch.sum(attention_mask, dim=1)
            generated_tokens = [output_seq[input_len:] for output_seq, input_len in zip(outputs, input_lengths)]
            predictions_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for i in range(len(predictions_batch)):
                prediction = predictions_batch[i]
                ground_truth = ground_truths_batch[i]
                pred_answer = extract_gsm8k_answer(prediction)
                true_answer = extract_gsm8k_answer(ground_truth)
                is_correct = False
                if pred_answer is not None and true_answer is not None:
                    try:
                        if abs(float(pred_answer.replace(',','')) - float(true_answer.replace(',',''))) < 1e-6: is_correct = True
                    except ValueError:
                        if pred_answer == true_answer: is_correct = True
                elif pred_answer == true_answer: is_correct = True
                local_results.append({'prediction': prediction, 'ground_truth': ground_truth, 'is_correct': is_correct})

        except Exception as e:
            logger.error(f"Error during generation or processing batch: {e}", exc_info=True)
            batch_size = len(ground_truths_batch) if 'ground_truths_batch' in locals() else input_ids.shape[0]
            for i in range(batch_size):
                 gt = ground_truths_batch[i] if 'ground_truths_batch' in locals() else 'N/A'
                 local_results.append({'prediction': '[ERROR: Batch Failed]', 'ground_truth': gt, 'is_correct': False})
        progress_bar.update(1)
    progress_bar.close()

    # --- Aggregate Metrics ---
    logger.info(f"Process {accelerator.process_index}: Finished evaluation loop. Gathering results...")
    all_results_gathered = accelerator.gather_for_metrics(local_results)
    logger.info(f"Process {accelerator.process_index}: Results gathered ({len(all_results_gathered)} total).")

    final_metrics = {}
    if accelerator.is_main_process:
        logger.info("Main process calculating final metrics...")
        correct_count = 0
        total_count = len(all_results_gathered)
        if total_count == 0: logger.warning("No results gathered."); return {"error": "No results gathered"}
        for result in all_results_gathered:
            if result.get('is_correct', False): correct_count += 1
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        final_metrics = {"exact_match_accuracy": accuracy, "correct_count": correct_count, "total_samples_processed": total_count}
        logger.info(f"Evaluation Metrics: {final_metrics}")

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
    set_seed(cfg.training.seed)

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
        eval_dataset_formatted = load_and_prepare_eval_data(cfg, tokenizer)
    except Exception as e: logger.error(f"Exiting due to eval data loading/prep failure: {e}", exc_info=True); sys.exit(1)

    logger.info("Starting evaluation process...")
    eval_metrics = run_evaluation(model, tokenizer, eval_dataset_formatted, cfg, accelerator, is_base_model_eval)

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
            logger.error(f"Failed to construct output file path: {e}", exc_info=True)

        if eval_metrics and "error" not in eval_metrics and output_file_path:
            logger.info(f"Saving evaluation metrics to: {output_file_path}")
            try:
                with open(output_file_path, 'w') as f:
                    json.dump(eval_metrics, f, indent=4)
                logger.info("Evaluation metrics saved successfully.")
            except Exception as e:
                logger.error(f"Failed to write metrics to {output_file_path}: {e}", exc_info=True)
        elif not eval_metrics or "error" in eval_metrics:
            logger.warning("Metrics not generated or contained errors. Output file not created.")
        else:
             logger.error("Output path invalid. Metrics not saved.")

    accelerator.wait_for_everyone()
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()
