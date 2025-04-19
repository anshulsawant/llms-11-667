import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from omegaconf import DictConfig # Keep DictConfig for type hint
from tqdm.auto import tqdm

# --- Import load_config from utils ---
try:
    # Use relative import assuming utils.py is in the same directory
    from .utils import load_config, load_model_and_tokenizer, read_jsonl, write_jsonl, extract_gsm8k_answer
except ImportError:
    logger.error("Failed to import functions from .utils. Make sure utils.py is in the same directory (src/sft_project) and run as part of the package.")
    # If running scripts directly without installing the package, imports might fail.
    # Consider adding the src directory to PYTHONPATH if needed for direct execution.
    raise
# --- End import ---


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# GSM8K prompt format (Consider moving to utils or config if used elsewhere)
PROMPT_FORMAT = "Question: {question}\nAnswer: "


# --- Removed copied load_config function definition ---


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference using a configuration file.")
    parser.add_argument( "--config_path", type=str, required=True, help="Path to the specific override configuration YAML file.")
    parser.add_argument( "--base_config_path", type=str, default="config.yaml", help="Path to the base configuration YAML file.")
    parser.add_argument( "--use_base_model", action="store_true", help="Force inference using the base model specified in the config.")
    # Removed other CLI args - now read from config's inference section
    args = parser.parse_args()
    return args

# --- load_model_tokenizer is now imported from utils ---
# --- read_jsonl is now imported from utils ---
# --- write_jsonl is now imported from utils ---

# --- run_inference remains here as it's specific to this script's workflow ---
@torch.no_grad()
def run_inference(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: List[Dict[str, Any]],
    prompt_field: str, output_field: str, batch_size: int,
    max_new_tokens: int, temperature: float, do_sample: bool,
    config_max_seq_len: int # Max sequence length from config
) -> List[Dict[str, Any]]:
    """Runs inference on the data and adds generations."""
    model.eval(); all_prompts = []; valid_indices = []
    logger.info(f"Using prompt field: '{prompt_field}', Output field: '{output_field}'")
    for i, item in enumerate(data):
        if prompt_field in item and isinstance(item[prompt_field], str) and item[prompt_field]:
            # Use the format_prompt function if available and applicable?
            # Or stick to the simpler PROMPT_FORMAT here? Let's stick to simpler for now.
            # formatted_prompt_obj = format_prompt(item, cfg) # Requires cfg access
            formatted_prompt = PROMPT_FORMAT.format(question=item[prompt_field])
            all_prompts.append(formatted_prompt); valid_indices.append(i)
        else:
            logger.warning(f"Skipping record {i+1}: Missing, empty, or invalid prompt field '{prompt_field}'. Found: {item.get(prompt_field)}")

    if not all_prompts: logger.warning("No valid prompts found."); return data

    logger.info(f"Processing {len(all_prompts)} valid prompts in batches of {batch_size}...")
    generations = []; num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": do_sample, "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id}
    if gen_kwargs["eos_token_id"] is None: logger.warning("EOS token ID is None.")

    progress_bar = tqdm(range(num_batches), desc="Generating", disable=False)
    for i in progress_bar:
        batch_start = i * batch_size; batch_end = min((i + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]

        max_input_len = config_max_seq_len - max_new_tokens
        if max_input_len <= 0: logger.error(f"Config max_seq_length ({config_max_seq_len}) too small for max_new_tokens ({max_new_tokens})."); generations.extend(["[ERROR: Input too long]"] * len(batch_prompts)); continue

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len).to(model.device)

        try:
            outputs = model.generate(**inputs, **gen_kwargs)
            batch_completions = []
            for j, output in enumerate(outputs):
                # Ensure slicing doesn't go out of bounds if generation is short
                input_len = inputs['input_ids'][j].shape[0]
                if output.shape[0] > input_len:
                     completion_tokens = output[input_len:]
                     completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                else:
                     completion = "" # Handle case where nothing new is generated
                batch_completions.append(completion)
            generations.extend(batch_completions)
        except Exception as gen_e: logger.error(f"Generation error batch {i+1}: {gen_e}", exc_info=False); generations.extend(["[ERROR: Generation Failed]"] * len(batch_prompts))

    output_data = data # Start with original data
    if len(generations) != len(valid_indices): logger.error(f"Mismatch generations ({len(generations)}) vs valid prompts ({len(valid_indices)}). Output alignment error.")

    gen_idx = 0
    # Add generation to corresponding original item using valid_indices
    for data_idx in valid_indices:
        if gen_idx < len(generations):
            output_data[data_idx][output_field] = generations[gen_idx]
            gen_idx += 1
        else:
            # Should not happen if lengths match check passes, but defensively add error
            if data_idx < len(output_data):
                 output_data[data_idx][output_field] = "[ERROR: Missing Generation]"

    # Mark items that were skipped initially
    for i, item in enumerate(output_data):
        if i not in valid_indices and output_field not in item:
             item[output_field] = "[SKIPPED: Invalid Input]"

    logger.info("Finished generating completions.")
    return output_data


def main():
    """Main function to orchestrate loading, inference, and writing."""
    args = parse_arguments()
    logger.info("Starting inference script...")
    logger.info(f"CLI Arguments: {vars(args)}")

    try:
        # Use the imported load_config function
        cfg = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        logger.info(f"Successfully loaded config from: {args.config_path} (overriding {args.base_config_path})")
    except Exception as e: logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)

    # Determine Model and Adapter Paths
    model_load_path: str; adapter_load_path: Optional[str] = None; model_identifier_for_output: str
    if args.use_base_model:
        model_load_path = cfg.model.name; adapter_load_path = None
        model_identifier_for_output = f"{Path(cfg.model.name).name.replace('/','_')}__base"
        logger.info(f"Using BASE model specified in config: '{model_load_path}'")
    else:
        tuning_method = cfg.get("tuning_method", "full")
        output_dir_from_config = Path(cfg.training.output_dir)
        expected_checkpoint_dir = output_dir_from_config / "final_checkpoint"
        model_identifier_for_output = Path(args.config_path).stem
        if tuning_method == "lora":
            model_load_path = cfg.model.name; adapter_load_path = str(expected_checkpoint_dir)
            logger.info(f"Inferred LoRA setup: Base='{model_load_path}', Adapter='{adapter_load_path}'")
            if not os.path.isdir(adapter_load_path): logger.warning(f"LoRA adapter path does not exist: {adapter_load_path}")
        elif tuning_method == "full":
            model_load_path = str(expected_checkpoint_dir); adapter_load_path = None
            logger.info(f"Inferred Full SFT setup: Model='{model_load_path}'")
            if not os.path.isdir(model_load_path): logger.error(f"Full SFT checkpoint path does not exist: {model_load_path}"); sys.exit(1)
        else: logger.error(f"Unknown tuning_method '{tuning_method}'."); sys.exit(1)

    # Get parameters ONLY from config
    inference_cfg = cfg.get("inference"); dataset_cfg = cfg.get("dataset", {})
    if not inference_cfg: logger.error("Config missing required 'inference' section."); sys.exit(1)
    input_file_str = inference_cfg.get("input_file")
    output_dir_str = inference_cfg.get("output_dir", ".")
    prompt_field = inference_cfg.get("prompt_field", "question")
    output_field = inference_cfg.get("output_field", "generation")
    if not input_file_str: logger.error("Missing 'inference.input_file' in configuration."); sys.exit(1)
    input_file = Path(input_file_str); output_dir = Path(output_dir_str)
    os.makedirs(output_dir, exist_ok=True)
    precision = inference_cfg.get("precision", "bf16"); batch_size = inference_cfg.get("batch_size", 4)
    max_new_tokens = inference_cfg.get("max_new_tokens", 256); temperature = inference_cfg.get("temperature", 0.1)
    do_sample = inference_cfg.get("do_sample", True); hf_token = cfg.model.get("access_token", None)
    trust_remote_code = cfg.model.get("trust_remote_code", False)
    config_max_seq_len = dataset_cfg.get("max_seq_length", 1024)

    logger.info(f"Using parameters from config: input='{input_file}', output_dir='{output_dir}', prompt='{prompt_field}', output='{output_field}'")
    logger.info(f"Generation params: precision={precision}, batch_size={batch_size}, max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}, config_max_seq_len={config_max_seq_len}")

    if not input_file.is_file(): logger.error(f"Input file not found: {input_file}"); sys.exit(1)

    try:
        # Use imported load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer( model_load_path, precision, hf_token, trust_remote_code, adapter_load_path)
    except Exception: logger.error("Exiting due to model loading failure."); sys.exit(1)

    try:
        # Use imported read_jsonl
        input_data = read_jsonl(input_file)
    except Exception: logger.error("Exiting due to input file reading failure."); sys.exit(1)
    if not input_data: logger.warning("Input file empty. Exiting."); sys.exit(0)

    try:
        output_data = run_inference( model, tokenizer, input_data, prompt_field, output_field, batch_size, max_new_tokens, temperature, do_sample, config_max_seq_len)
    except Exception as e: logger.error(f"Inference error: {e}", exc_info=True); sys.exit(1)

    # Generate output filename
    input_filename_stem = input_file.stem
    output_filename = f"{input_filename_stem}__{model_identifier_for_output}__generations.jsonl"
    output_path = output_dir / output_filename

    # Write results using imported write_jsonl
    try: write_jsonl(output_path, output_data)
    except Exception: logger.error("Exiting due to output file writing failure."); sys.exit(1)

    logger.info("Inference script finished successfully.")

if __name__ == "__main__":
    main()
