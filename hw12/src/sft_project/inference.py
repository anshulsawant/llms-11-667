"""Script for running inference with base or fine-tuned models."""

import os
import logging
import argparse
import json
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
    from datasets import Dataset # Although not directly used, good for consistency
    from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import PeftModel
    from tqdm.auto import tqdm

    # --- Import from utils ---
    # Ensure these functions exist in your utils.py!
    from .utils import (
        load_config,
        load_model_and_tokenizer, # Expects only cfg
        format_prompt,
        extract_gsm8k_answer, # Optional: if you need answer extraction post-inference
        read_jsonl, # Expects path, returns List[Dict]
        write_jsonl # Expects path and List[Dict]
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
    # No --input_file or --output_file arguments needed
    return parser.parse_args()

@torch.no_grad()
def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_data: List[Dict[str, Any]],
    cfg: DictConfig,
    accelerator: Accelerator,
    is_base_model_eval: bool = False # Reusing this flag name for consistency
) -> List[Dict[str, Any]]:
    """
    Performs inference on the input data.

    Args:
        model: The loaded model (potentially adapted).
        tokenizer: The loaded tokenizer.
        input_data: A list of dictionaries, each representing an input sample.
                    Expected to be compatible with `format_prompt`.
        cfg: The configuration object.
        accelerator: The Accelerator object.
        is_base_model_eval: Flag indicating if the base model is being used.

    Returns:
        A list of dictionaries, where each dictionary contains the original
        input data merged with the generated output ('completion').
    """
    model.eval() # Set model to evaluation mode
    results = []

    # --- Get Generation Parameters ---
    gen_cfg = cfg.get("inference", cfg.get("evaluation", {})) # Use inference settings if available
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

    # --- Prepare Prompts ---
    try:
        formatted_prompts = [format_prompt(sample, cfg)["prompt"] for sample in input_data]
        logger.info(f"Formatted {len(formatted_prompts)} prompts.")
    except KeyError:
        logger.error("Failed to format prompts. Ensure `format_prompt` returns a dict with a 'prompt' key and input data has required fields.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during prompt formatting: {e}", exc_info=True)
        return []

    # --- Inference Loop ---
    progress_bar = tqdm(total=len(formatted_prompts), desc="Running Inference", disable=not accelerator.is_local_main_process)

    for i, prompt_text in enumerate(formatted_prompts):
        # --- Tokenization ---
        buffer = 10
        max_model_len = getattr(tokenizer, 'model_max_length', cfg.dataset.get('max_seq_length', 2048))
        max_input_length = max_model_len - generation_kwargs["max_new_tokens"] - buffer

        if max_input_length <= 0:
            logger.error(f"Model max length ({max_model_len}) is too small for max_new_tokens ({generation_kwargs['max_new_tokens']}). Skipping sample {i}.")
            output_sample = {**input_data[i], "completion": "[ERROR: Input too long]"}
            results.append(output_sample)
            progress_bar.update(1)
            continue

        inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_input_length)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        # --- Generation ---
        try:
            outputs = model.generate(**inputs, **generation_kwargs)
            completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

            # --- Store Result ---
            output_sample = {**input_data[i], "completion": completion}
            results.append(output_sample)

        except Exception as e:
            logger.error(f"Generation error for sample {i}: {e}", exc_info=False)
            output_sample = {**input_data[i], "completion": "[ERROR: Generation Failed]"}
            results.append(output_sample)

        progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Inference completed for {len(results)} samples.")
    return results


def main():
    """Main function for inference."""
    args = parse_arguments()
    logger.info("Starting inference script...")
    logger.info(f"CLI Arguments: {vars(args)}")

    # --- Load Config ---
    try:
        cfg = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        logger.info(f"Successfully loaded config from: {args.config_path} (overriding {args.base_config_path})")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)

    # --- Initialize Accelerator ---
    accelerator = Accelerator()

    # --- Load Base Model and Tokenizer ---
    try:
        logger.info("Loading base model and tokenizer using config...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        logger.info("Base model and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Exiting due to base model/tokenizer loading failure: {e}", exc_info=True); sys.exit(1)

    # --- Load Fine-Tuned Model/Adapter if necessary ---
    is_base_model_eval = args.use_base_model

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
                    logger.error(f"LoRA adapter path does not exist: {adapter_load_path}"); sys.exit(1)
                model = PeftModel.from_pretrained(model, adapter_load_path)
                logger.info("LoRA adapter loaded successfully.")

            elif tuning_method == "full":
                checkpoint_path = str(expected_checkpoint_dir)
                logger.info(f"Loading Full SFT checkpoint from: '{checkpoint_path}'")
                if not os.path.isdir(checkpoint_path):
                    logger.error(f"Full SFT checkpoint path does not exist: {checkpoint_path}"); sys.exit(1)

                access_token = cfg.model.get("access_token", None)
                trust_remote_code = cfg.model.get("trust_remote_code", False)
                model_kwargs = {"trust_remote_code": trust_remote_code, "token": access_token}
                attn_impl = cfg.model.get("attn_implementation", None)
                if attn_impl: model_kwargs["attn_implementation"] = attn_impl
                precision = "bf16" if cfg.training.get("bf16") else "fp16" if cfg.training.get("fp16") else "fp32"
                if precision == "bf16": model_kwargs["torch_dtype"] = torch.bfloat16
                elif precision == "fp16": model_kwargs["torch_dtype"] = torch.float16

                del model; torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
                logger.info("Full SFT model loaded successfully from checkpoint.")

            else:
                logger.error(f"Unknown tuning_method '{tuning_method}'."); sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model/adapter (method: {tuning_method}): {e}", exc_info=True); sys.exit(1)
    else:
        logger.info(f"Running inference with BASE model specified in config: '{cfg.model.name}'")

    # --- Load Input Data from Config ---
    try:
        input_file_path_str = cfg.inference.input_file
        input_file_path = Path(input_file_path_str)
        logger.info(f"Loading input data from config path: {input_file_path}")
    except MissingMandatoryValue:
        logger.error("Missing 'inference.input_file' in configuration!"); sys.exit(1)
    except Exception as e:
         logger.error(f"Error accessing input file path from config: {e}"); sys.exit(1)

    if not input_file_path.exists():
        logger.error(f"Input file specified in config not found: {input_file_path}"); sys.exit(1)
    try:
        input_data = read_jsonl(str(input_file_path))
        if not input_data:
             logger.warning(f"Input file {input_file_path} is empty or could not be read."); sys.exit(0)
        logger.info(f"Loaded {len(input_data)} samples from input file.")
    except Exception as e:
        logger.error(f"Failed to read input file {input_file_path}: {e}", exc_info=True); sys.exit(1)

    # --- Construct Output File Path ---
    try:
        config_path = Path(args.config_path)
        config_stem = config_path.stem
        data_stem = input_file_path.stem
        # Place output file in the same directory as the input data file
        output_filename = f"{config_stem}_{data_stem}_results.jsonl"
        output_file_path = input_file_path.parent / output_filename
        logger.info(f"Output will be saved to: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to construct output file path: {e}", exc_info=True)
        # Decide if we should exit or proceed without saving
        output_file_path = None # Set to None to prevent saving attempt later


    # --- Prepare Model with Accelerator ---
    logger.info("Preparing model with Accelerator...")
    model = accelerator.prepare(model)
    logger.info("Model prepared.")

    # --- Run Inference ---
    logger.info("Starting inference process...")
    inference_results = run_inference(
        model, tokenizer, input_data, cfg, accelerator, is_base_model_eval
    )

    # --- Save Results (only on main process) ---
    if accelerator.is_main_process:
        if inference_results and output_file_path: # Check if results exist and path was constructed
            logger.info(f"Saving inference results to: {output_file_path}")
            try:
                # write_jsonl handles directory creation now, but double-check if needed
                # output_file_path.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(str(output_file_path), inference_results)
                logger.info("Results saved successfully.")
            except Exception as e:
                logger.error(f"Failed to write results to {output_file_path}: {e}", exc_info=True)
        elif not inference_results:
            logger.warning("Inference process completed, but no results were generated. Output file will not be created.")
        else: # Results exist but output_file_path is None
             logger.error("Output file path could not be constructed. Results were not saved.")


    accelerator.wait_for_everyone()
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()
