# src/sft_project/inference.py
"""Script for running inference with base or fine-tuned models."""

import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# --- Setup logger FIRST ---
# Changed default level to INFO, but DEBUG can be useful for length issues
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
    # Use inference settings if available, otherwise fallback to evaluation settings
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

    # --- Prepare Prompts (initial formatting) ---
    try:
        # Get the base formatted prompts first
        formatted_samples = [format_prompt(sample, cfg) for sample in input_data]
        base_prompts = [sample["prompt"] for sample in formatted_samples] # Extract the prompt part
        logger.info(f"Formatted {len(base_prompts)} base prompts.")
    except KeyError:
        logger.error("Failed to format prompts. Ensure `format_prompt` returns a dict with a 'prompt' key and input data has required fields.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during prompt formatting: {e}", exc_info=True)
        return []

    # --- Inference Loop ---
    progress_bar = tqdm(total=len(base_prompts), desc="Running Inference", disable=not accelerator.is_local_main_process)

    for i, base_prompt_text in enumerate(base_prompts):

        # --- Prepend Base Model Instructions/Few-Shot if needed ---
        instruction_text = ""
        few_shot_text = ""
        final_prompt_text = base_prompt_text # Default to base prompt

        if is_base_model_eval:
            # Get instruction from config (inference > evaluation > default)
            instruction_text = cfg.inference.get("base_model_instruction",
                                                 cfg.evaluation.get("base_model_instruction",
                                                                    "You are a helpful assistant.")) # Generic default
            if instruction_text:
                 instruction_text += "\n\n" # Add spacing
                 logger.debug(f"Using base model instruction text.")

            # Check for few-shot strategy (inference > evaluation > default)
            strategy = cfg.inference.get("base_model_prompt_strategy",
                                         cfg.evaluation.get("base_model_prompt_strategy", "zero_shot"))
            logger.debug(f"Base model inference strategy: {strategy}")

            if strategy == "one_shot":
                # Get example from config (inference > evaluation > default=None)
                one_shot_example = cfg.inference.get("few_shot_example",
                                                     cfg.evaluation.get("few_shot_example", None))
                if one_shot_example and isinstance(one_shot_example, DictConfig) and \
                   one_shot_example.get("question") and one_shot_example.get("answer"):
                    # Format the one-shot example (assuming question/answer keys like GSM8K)
                    q = one_shot_example.question
                    a = one_shot_example.answer
                    # Adapt keys if your few-shot example uses different ones (e.g., 'input', 'output')
                    few_shot_text = f"Question: {q}\nAnswer: {a}\n\n" # Match GSM8K format
                    logger.info("Using one-shot example for base model.")
                else:
                    logger.warning("One-shot strategy requested for base model, but 'few_shot_example' is missing, invalid, or lacks 'question'/'answer' in config.")
            elif strategy != "zero_shot":
                 logger.warning(f"Unknown base_model_prompt_strategy '{strategy}'. Using zero-shot.")

            # Construct the final prompt for the base model
            final_prompt_text = instruction_text + few_shot_text + base_prompt_text

        # --- Tokenization ---
        buffer = 10 # Add a small buffer for safety

        # Determine max_model_len *directly from config*
        max_model_len = int(cfg.dataset.get('max_seq_length', 2048)) # Default to 2048 if not in config
        logger.debug(f"Using max_seq_length from config: {max_model_len}")

        # Ensure max_new_tokens is reasonable compared to max_model_len
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 256)) # Ensure int
        if max_new_tokens >= max_model_len:
            logger.warning(f"max_new_tokens ({max_new_tokens}) >= max_model_len ({max_model_len}). Setting max_new_tokens to {max_model_len // 2} to allow for input.")
            max_new_tokens = max_model_len // 2

        # Calculate max_input_length for truncation
        max_input_length = max_model_len - max_new_tokens - buffer
        logger.debug(f"Calculated max_input_length: {max_input_length} (max_model_len={max_model_len}, max_new_tokens={max_new_tokens}, buffer={buffer})")

        # Check if calculated max_input_length is valid
        if max_input_length <= 0:
            logger.error(f"Calculated max_input_length ({max_input_length}) is zero or negative. Skipping sample {i}.")
            output_sample = {**input_data[i], "completion": "[ERROR: Input length calculation failed]"}
            results.append(output_sample)
            progress_bar.update(1)
            continue

        # Tokenize the final prompt text (potentially with instructions prepended)
        try:
            inputs = tokenizer(
                final_prompt_text, # Use the potentially modified prompt
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=int(max_input_length) # Ensure it's an integer
            )
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        except OverflowError as oe:
             logger.error(f"OverflowError during tokenization for sample {i}. max_input_length={max_input_length}. Error: {oe}", exc_info=True)
             output_sample = {**input_data[i], "completion": "[ERROR: Tokenization Overflow]"}
             results.append(output_sample)
             progress_bar.update(1)
             continue
        except Exception as e:
             logger.error(f"Unexpected error during tokenization for sample {i}: {e}", exc_info=True)
             output_sample = {**input_data[i], "completion": "[ERROR: Tokenization Failed]"}
             results.append(output_sample)
             progress_bar.update(1)
             continue


        # --- Generation ---
        try:
            outputs = model.generate(**inputs, **generation_kwargs)
            completion_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

            # --- Store Result ---
            output_sample = {**input_data[i], "completion": completion} # Use original input_data[i] for merging
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
    is_base_model_eval = args.use_base_model # Determine if evaluating base model

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
        # Read the original input data dictionaries
        input_data_dicts = read_jsonl(str(input_file_path))
        if not input_data_dicts:
             logger.warning(f"Input file {input_file_path} is empty or could not be read."); sys.exit(0)
        logger.info(f"Loaded {len(input_data_dicts)} samples from input file.")
    except Exception as e:
        logger.error(f"Failed to read input file {input_file_path}: {e}", exc_info=True); sys.exit(1)

    # --- Construct Output File Path ---
    output_file_path = None # Initialize to None
    try:
        config_path = Path(args.config_path)
        config_stem = config_path.stem
        data_stem = input_file_path.stem
        output_filename = f"{config_stem}_{data_stem}_results.jsonl"
        output_file_path = input_file_path.parent / output_filename
        logger.info(f"Output will be saved to: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to construct output file path: {e}", exc_info=True)


    # --- Prepare Model with Accelerator ---
    logger.info("Preparing model with Accelerator...")
    model = accelerator.prepare(model)
    logger.info("Model prepared.")

    # --- Run Inference ---
    logger.info("Starting inference process...")
    # Pass the original dictionaries to run_inference
    inference_results = run_inference(
        model, tokenizer, input_data_dicts, cfg, accelerator, is_base_model_eval
    )

    # --- Save Results (only on main process) ---
    if accelerator.is_main_process:
        if inference_results and output_file_path: # Check if results exist and path was constructed
            logger.info(f"Saving inference results to: {output_file_path}")
            try:
                write_jsonl(str(output_file_path), inference_results)
                logger.info("Results saved successfully.")
            except Exception as e:
                logger.error(f"Failed to write results to {output_file_path}: {e}", exc_info=True)
        elif not inference_results:
            logger.warning("Inference process completed, but no results were generated. Output file will not be created.")
        else: # Results exist but output_file_path is None
             logger.error("Output file path could not be constructed or was invalid. Results were not saved.")


    accelerator.wait_for_everyone() # Ensure all processes finish before exiting
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()
