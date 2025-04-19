# src/sft_project/inference.py
"""Script for running inference with base or fine-tuned models."""

import os
import logging
import argparse
import json
import re # Import re for slugify
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# --- Setup logger FIRST ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End logger setup ---

try:
    import torch
    import transformers
    from accelerate import Accelerator
    from datasets import Dataset # Keep for potential future use/consistency
    from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import PeftModel
    from tqdm.auto import tqdm

    # --- Import from utils ---
    from .utils import (
        load_config,
        load_model_and_tokenizer,
        format_prompt,
        extract_gsm8k_answer, # Keep even if not used directly here, utils standard
        read_jsonl,
        write_jsonl
    )
    # --- End imports ---

except ImportError as e:
    logger.error(f"Failed to import required libraries or functions from .utils: {e}. Make sure all dependencies are installed and utils.py is correct and accessible.")
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a base or fine-tuned model using config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the override configuration YAML file.")
    parser.add_argument("--base_config_path", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    parser.add_argument("--use_base_model", action="store_true", help="Force inference using the base model specified in the config, ignoring checkpoints.")
    return parser.parse_args()

def slugify(value: str) -> str:
    """Normalizes string, removes invalid chars, and converts spaces to hyphens."""
    if not isinstance(value, str): value = str(value) # Ensure string type
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    if not value: return "na" # Handle empty slugs
    return value

@torch.no_grad()
def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_data: List[Dict[str, Any]],
    cfg: DictConfig,
    accelerator: Accelerator,
    is_base_model_eval: bool = False
) -> List[Dict[str, Any]]:
    """
    Performs inference on the input data. Handles prompt formatting, generation,
    and prepends base model instructions/few-shot examples if needed.
    """
    model.eval()
    results = []
    gen_cfg = cfg.get("inference", cfg.get("evaluation", {}))
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
    logger.info(f"Using generation parameters: {generation_kwargs}")

    try:
        formatted_samples = [format_prompt(sample, cfg) for sample in input_data]
        base_prompts = [sample["prompt"] for sample in formatted_samples]
        logger.info(f"Formatted {len(base_prompts)} base prompts.")
    except KeyError:
        logger.error("Failed to format prompts.", exc_info=True); return []
    except Exception as e:
        logger.error(f"Error during prompt formatting: {e}", exc_info=True); return []

    progress_bar = tqdm(total=len(base_prompts), desc="Running Inference", disable=not accelerator.is_local_main_process)
    for i, base_prompt_text in enumerate(base_prompts):
        instruction_text = ""
        few_shot_text = "" # Variable name kept for simplicity, logic uses one_shot_example key
        final_prompt_text = base_prompt_text

        if is_base_model_eval:
            instruction_text = cfg.inference.get("base_model_instruction", cfg.evaluation.get("base_model_instruction", "You are a helpful assistant."))
            if instruction_text: instruction_text += "\n\n"
            strategy = cfg.inference.get("base_model_prompt_strategy", cfg.evaluation.get("base_model_prompt_strategy", "zero_shot"))
            if strategy == "one_shot":
                one_shot_example = cfg.inference.get("one_shot_example", cfg.evaluation.get("one_shot_example", None))
                if one_shot_example and isinstance(one_shot_example, DictConfig) and one_shot_example.get("question") and one_shot_example.get("answer"):
                    q = one_shot_example.question; a = one_shot_example.answer
                    few_shot_text = f"Question: {q}\nAnswer: {a}\n\n"
                    if i == 0: logger.info("Using one-shot example for base model.") # Log only once
                elif i == 0: logger.warning("One-shot requested but 'one_shot_example' invalid/missing in config.")
            elif strategy != "zero_shot" and i == 0: logger.warning(f"Unknown base model strategy '{strategy}'. Using zero-shot.")
            final_prompt_text = instruction_text + few_shot_text + base_prompt_text

        buffer = 10
        max_model_len = int(cfg.dataset.get('max_seq_length', 2048))
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 256))
        if max_new_tokens >= max_model_len:
            if i == 0: logger.warning(f"max_new_tokens >= max_model_len. Adjusting.")
            max_new_tokens = max_model_len // 2
        max_input_length = max_model_len - max_new_tokens - buffer

        if max_input_length <= 0:
            logger.error(f"Input length calc failed (<=0) for sample {i}. Skipping."); output_sample = {**input_data[i], "completion": "[ERROR: Input length calculation failed]"}; results.append(output_sample); progress_bar.update(1); continue

        try:
            inputs = tokenizer(final_prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=int(max_input_length))
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        except OverflowError as oe:
             logger.error(f"Tokenization OverflowError sample {i}. max_input_length={max_input_length}. Error: {oe}", exc_info=True); output_sample = {**input_data[i], "completion": "[ERROR: Tokenization Overflow]"}; results.append(output_sample); progress_bar.update(1); continue
        except Exception as e:
             logger.error(f"Tokenization error sample {i}: {e}", exc_info=True); output_sample = {**input_data[i], "completion": "[ERROR: Tokenization Failed]"}; results.append(output_sample); progress_bar.update(1); continue

        try:
            outputs = model.generate(**inputs, **generation_kwargs)
            completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            output_sample = {**input_data[i], "completion": completion}
            results.append(output_sample)
        except Exception as e:
            logger.error(f"Generation error sample {i}: {e}", exc_info=False); output_sample = {**input_data[i], "completion": "[ERROR: Generation Failed]"}; results.append(output_sample)
        progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Inference completed for {len(results)} samples.")
    return results


def main():
    """Main function for inference."""
    args = parse_arguments()
    logger.info("Starting inference script...")
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
    model_type_str = "base" # Default for filename

    if not is_base_model_eval:
        tuning_method = cfg.get("tuning_method", "full")
        model_type_str = slugify(tuning_method) # Use 'lora' or 'full'
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
    else: logger.info(f"Running inference with BASE model: '{cfg.model.name}'")

    try:
        input_file_path_str = cfg.inference.input_file
        input_file_path = Path(input_file_path_str)
        logger.info(f"Loading input data from config path: {input_file_path}")
    except MissingMandatoryValue: logger.error("Missing 'inference.input_file' in configuration!"); sys.exit(1)
    except Exception as e: logger.error(f"Error accessing input file path from config: {e}"); sys.exit(1)

    if not input_file_path.exists(): logger.error(f"Input file not found: {input_file_path}"); sys.exit(1)
    try:
        input_data_dicts = read_jsonl(str(input_file_path))
        if not input_data_dicts: logger.warning(f"Input file empty: {input_file_path}"); sys.exit(0)
        logger.info(f"Loaded {len(input_data_dicts)} samples.")
    except Exception as e: logger.error(f"Failed to read input file {input_file_path}: {e}", exc_info=True); sys.exit(1)

    # --- Construct Output File Path (New Logic) ---
    output_file_path = None
    try:
        config_path = Path(args.config_path)
        config_stem = slugify(config_path.stem)
        data_stem = slugify(input_file_path.stem)
        base_model_name = cfg.model.get("name", "unknown_model")
        base_model_slug = slugify(base_model_name.split('/')[-1])
        dataset_config_name = slugify(cfg.dataset.get("config_name", "unknown_dataset"))
        output_filename = f"{config_stem}_{data_stem}_{model_type_str}_{base_model_slug}_{dataset_config_name}_results.jsonl"
        output_file_path = input_file_path.parent / output_filename # Save in same dir as input
        logger.info(f"Constructed output path: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to construct output file path: {e}", exc_info=True)
    # --- End Output File Path Construction ---

    logger.info("Preparing model with Accelerator...")
    model = accelerator.prepare(model)
    logger.info("Model prepared.")

    logger.info("Starting inference process...")
    inference_results = run_inference(model, tokenizer, input_data_dicts, cfg, accelerator, is_base_model_eval)

    if accelerator.is_main_process:
        if inference_results and output_file_path:
            logger.info(f"Saving inference results to: {output_file_path}")
            try:
                write_jsonl(str(output_file_path), inference_results)
                logger.info("Results saved successfully.")
            except Exception as e: logger.error(f"Failed to write results to {output_file_path}: {e}", exc_info=True)
        elif not inference_results: logger.warning("No results generated. Output file not created.")
        else: logger.error("Output path invalid. Results not saved.")

    accelerator.wait_for_everyone()
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()
