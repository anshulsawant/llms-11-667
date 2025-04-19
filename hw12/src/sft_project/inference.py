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

# Import load_config from sft_script
try:
    from .sft_script import load_config
except ImportError:
    logger.error("Failed to import 'load_config' from sft_script. Make sure scripts are in the same package or run as part of the installed module.")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# GSM8K prompt format
PROMPT_FORMAT = "Question: {question}\nAnswer: "


# --- load_config function (imported or copied) remains here ---
# (Assuming load_config is available via import as per previous step)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference using a configuration file.")

    # --- Essential Arguments ---
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the specific override configuration YAML file (e.g., config_full_main.yaml)."
    )
    parser.add_argument(
        "--base_config_path", type=str, default="config.yaml",
        help="Path to the base configuration YAML file."
    )
    # --- New flag to force using the base model ---
    parser.add_argument(
        "--use_base_model", action="store_true",
        help="Force inference using the base model specified in the config, ignoring any fine-tuning checkpoints or adapters."
    )
    # --- Arguments now read from config ---
    # input_file, output_dir, prompt_field, output_field

    # --- Removed Optional Overrides ---

    args = parser.parse_args()
    return args

# --- load_model_tokenizer remains the same ---
def load_model_tokenizer(
    model_name_or_path: str,
    precision: str,
    hf_token: Optional[str],
    # force_cpu: bool, # Removed
    trust_remote_code: bool,
    adapter_path: Optional[str] = None
) -> (AutoModelForCausalLM, AutoTokenizer):
    """Loads the specified base model and tokenizer, optionally applying LoRA adapters."""
    logger.info(f"Loading model/tokenizer: {model_name_or_path}")
    if adapter_path: logger.info(f"Will apply LoRA adapter from: {adapter_path}")

    if precision == "bf16": dtype = torch.bfloat16
    elif precision == "fp16": dtype = torch.float16
    else: dtype = torch.float32
    logger.info(f"Using precision: {precision} ({dtype})")

    # Determine device map automatically
    if torch.cuda.is_available():
        device_map = "auto"; logger.info("CUDA available.")
    else:
        device_map = "cpu"; logger.info("CUDA not available, using CPU.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            token=hf_token,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )
        tokenizer_load_path = model_name_or_path
        # For LoRA, tokenizer should still come from base model path
        if adapter_path: logger.info(f"Loading tokenizer from base model path: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, token=hf_token, padding_side="right")

        if adapter_path:
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
            except Exception as peft_e:
                 logger.error(f"Failed to load PEFT adapter: {peft_e}", exc_info=True); logger.warning("Proceeding with base model only.")

        if tokenizer.pad_token is None:
            logger.warning("Tokenizer setting pad_token=eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            if hasattr(model, 'config'):
                 model.config.pad_token_id = model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else tokenizer.eos_token_id

        logger.info("Model and tokenizer ready.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
        raise


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    # (Implementation remains the same)
    data = []; logger.info(f"Reading input file: {file_path}")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip(): data.append(json.loads(line))
        logger.info(f"Read {len(data)} records.")
        return data
    except FileNotFoundError: logger.error(f"Input file not found: {file_path}"); raise
    except json.JSONDecodeError as e: logger.error(f"Error decoding JSON line {i+1} in {file_path}: {e}"); raise
    except Exception as e: logger.error(f"Error reading input file {file_path}: {e}"); raise

def write_jsonl(file_path: Path, data: List[Dict[str, Any]]):
    # (Implementation remains the same)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('w', encoding='utf-8') as f:
            for item in data: f.write(json.dumps(item) + '\n')
        logger.info(f"Successfully wrote {len(data)} records to: {file_path}")
    except Exception as e: logger.error(f"Error writing output file {file_path}: {e}"); raise

@torch.no_grad()
def run_inference(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: List[Dict[str, Any]],
    prompt_field: str, output_field: str, batch_size: int,
    max_new_tokens: int, temperature: float, do_sample: bool
) -> List[Dict[str, Any]]:
    # (Implementation remains the same)
    model.eval(); all_prompts = []; valid_indices = []
    for i, item in enumerate(data):
        if prompt_field in item and isinstance(item[prompt_field], str):
            formatted_prompt = PROMPT_FORMAT.format(question=item[prompt_field])
            all_prompts.append(formatted_prompt); valid_indices.append(i)
        else: logger.warning(f"Skipping record {i+1}: Invalid prompt field '{prompt_field}'.")
    if not all_prompts: logger.warning("No valid prompts found."); return data
    logger.info(f"Processing {len(all_prompts)} valid prompts in batches of {batch_size}...")
    generations = []; num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": do_sample, "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id}
    if gen_kwargs["eos_token_id"] is None: logger.warning("EOS token ID is None.")
    progress_bar = tqdm(range(num_batches), desc="Generating", disable=False)
    for i in progress_bar:
        batch_start = i * batch_size; batch_end = min((i + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        tokenizer_max_len = getattr(tokenizer, 'model_max_length', 1024); max_input_len = tokenizer_max_len - max_new_tokens
        if max_input_len <= 0: logger.error(f"Tokenizer max length too small."); generations.extend(["[ERROR: Input too long]"] * len(batch_prompts)); continue
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len).to(model.device)
        try:
            outputs = model.generate(**inputs, **gen_kwargs)
            batch_completions = []
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]; completion_tokens = output[input_length:]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                batch_completions.append(completion)
            generations.extend(batch_completions)
        except Exception as gen_e: logger.error(f"Generation error batch {i+1}: {gen_e}", exc_info=False); generations.extend(["[ERROR: Generation Failed]"] * len(batch_prompts))
    output_data = data
    if len(generations) != len(valid_indices): logger.error(f"Mismatch generations ({len(generations)}) vs valid prompts ({len(valid_indices)}).")
    gen_idx = 0
    for data_idx in valid_indices:
        if gen_idx < len(generations): output_data[data_idx][output_field] = generations[gen_idx]; gen_idx += 1
        else: output_data[data_idx][output_field] = "[ERROR: Missing Generation]"
    for i, item in enumerate(output_data):
        if i not in valid_indices and output_field not in item: item[output_field] = "[SKIPPED: Invalid Input]"
    logger.info("Finished generating completions.")
    return output_data

# --- Updated main function ---
def main():
    """Main function to orchestrate loading, inference, and writing."""
    args = parse_arguments()
    logger.info("Starting inference script...")
    logger.info(f"CLI Arguments: {vars(args)}")

    try:
        cfg = load_config(override_config_path=args.config_path, base_config_path=args.base_config_path)
        logger.info(f"Successfully loaded config from: {args.config_path} (overriding {args.base_config_path})")
    except Exception as e: logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)

    # --- Determine Model/Adapter Paths based on config and --use_base_model flag ---
    model_load_path: str; adapter_load_path: Optional[str] = None
    model_identifier_for_output: str # For naming the output file

    if args.use_base_model:
        model_load_path = cfg.model.name
        adapter_load_path = None
        model_identifier_for_output = f"{Path(cfg.model.name).name.replace('/','_')}__base"
        logger.info(f"Using BASE model specified in config: '{model_load_path}'")
    else:
        # Infer paths from training output defined in the config
        tuning_method = cfg.get("tuning_method", "full")
        output_dir_from_config = Path(cfg.training.output_dir)
        expected_checkpoint_dir = output_dir_from_config / "final_checkpoint"
        # Use the config file stem as the identifier for fine-tuned models
        model_identifier_for_output = Path(args.config_path).stem

        if tuning_method == "lora":
            model_load_path = cfg.model.name # Base model path
            adapter_load_path = str(expected_checkpoint_dir) # Adapter path
            logger.info(f"Inferred LoRA setup: Base='{model_load_path}', Adapter='{adapter_load_path}'")
            if not os.path.isdir(adapter_load_path): logger.warning(f"LoRA adapter path does not exist: {adapter_load_path}")
        elif tuning_method == "full":
            model_load_path = str(expected_checkpoint_dir) # Full SFT checkpoint path
            adapter_load_path = None
            logger.info(f"Inferred Full SFT setup: Model='{model_load_path}'")
            if not os.path.isdir(model_load_path): logger.error(f"Full SFT checkpoint path does not exist: {model_load_path}"); sys.exit(1)
        else: logger.error(f"Unknown tuning_method '{tuning_method}'."); sys.exit(1)
    # --- End Path Inference ---

    # --- Get parameters ONLY from config's 'inference' section ---
    inference_cfg = cfg.get("inference")
    if not inference_cfg: logger.error("Config missing required 'inference' section."); sys.exit(1)

    # Required inference params from config
    input_file_str = inference_cfg.get("input_file")
    output_dir_str = inference_cfg.get("output_dir", ".")
    prompt_field = inference_cfg.get("prompt_field", "question")
    output_field = inference_cfg.get("output_field", "generation")
    if not input_file_str: logger.error("Missing 'inference.input_file' in configuration."); sys.exit(1)
    input_file = Path(input_file_str)
    output_dir = Path(output_dir_str)
    os.makedirs(output_dir, exist_ok=True)

    # Other inference params from config (with defaults)
    precision = inference_cfg.get("precision", "bf16")
    batch_size = inference_cfg.get("batch_size", 4)
    max_new_tokens = inference_cfg.get("max_new_tokens", 256)
    temperature = inference_cfg.get("temperature", 0.1)
    do_sample = inference_cfg.get("do_sample", True)
    hf_token = cfg.model.get("access_token", None) # Get token from model config
    trust_remote_code = cfg.model.get("trust_remote_code", False)

    logger.info(f"Using parameters from config: input='{input_file}', output_dir='{output_dir}', prompt='{prompt_field}', output='{output_field}'")
    logger.info(f"Generation params: precision={precision}, batch_size={batch_size}, max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")
    # --- End Parameter Determination ---

    if not input_file.is_file(): logger.error(f"Input file not found: {input_file}"); sys.exit(1)

    try:
        # Pass config-derived parameters, force_cpu removed
        model, tokenizer = load_model_tokenizer(
            model_load_path, precision, hf_token,
            trust_remote_code, adapter_load_path
        )
    except Exception: logger.error("Exiting due to model loading failure."); sys.exit(1)

    try: input_data = read_jsonl(input_file)
    except Exception: logger.error("Exiting due to input file reading failure."); sys.exit(1)
    if not input_data: logger.warning("Input file empty. Exiting."); sys.exit(0)

    try:
        output_data = run_inference(
            model, tokenizer, input_data, prompt_field, output_field,
            batch_size, max_new_tokens, temperature, do_sample
        )
    except Exception as e: logger.error(f"Inference error: {e}", exc_info=True); sys.exit(1)

    # --- Generate output filename using appropriate identifier ---
    input_filename_stem = input_file.stem
    # model_identifier_for_output determined earlier based on --use_base_model flag
    output_filename = f"{input_filename_stem}__{model_identifier_for_output}__generations.jsonl"
    output_path = output_dir / output_filename
    # --- End filename generation ---

    # --- Write results to the NEW file ---
    try: write_jsonl(output_path, output_data)
    except Exception: logger.error("Exiting due to output file writing failure."); sys.exit(1)
    # --- End writing results ---

    logger.info("Inference script finished successfully.")

if __name__ == "__main__":
    main()
