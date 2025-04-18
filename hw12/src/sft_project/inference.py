import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional # Added Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# --- Added PEFT import ---
from peft import PeftModel
# --- End PEFT import ---


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# GSM8K prompt format
PROMPT_FORMAT = "Question: {question}\nAnswer: "

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a base or fine-tuned model.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model name OR path to the BASE model checkpoint (e.g., 'google/gemma-2-9b-it'). For LoRA, provide base model here."
    )
    # --- New argument for LoRA adapters ---
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional path to the trained LoRA adapter checkpoint (e.g., './sft_results/final_checkpoint_lora/'). If provided, adapters will be loaded onto the base model."
    )
    # --- End LoRA argument ---
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input JSON Lines (.jsonl) file."
    )
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="question",
        help="Field name in the JSON object containing the input question/prompt."
    )
    parser.add_argument(
        "--output_field",
        type=str,
        default="generation",
        help="Field name to add to the JSON object with the model's generation."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face access token."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision (bf16, fp16, or fp32)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts to process in parallel."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature."
    )
    parser.add_argument(
        "--do_sample",
        action='store_true',
        help="Enable sampling during generation."
    )
    parser.add_argument(
        "--force_cpu",
        action='store_true',
        help="Force loading the model on CPU."
    )

    return parser.parse_args()

def load_model_tokenizer(
    model_name_or_path: str,
    precision: str,
    hf_token: Optional[str],
    force_cpu: bool,
    adapter_path: Optional[str] = None # Added adapter_path
) -> (AutoModelForCausalLM, AutoTokenizer):
    """Loads the specified base model and tokenizer, optionally applying LoRA adapters."""
    logger.info(f"Loading base model: {model_name_or_path}")

    if precision == "bf16": dtype = torch.bfloat16
    elif precision == "fp16": dtype = torch.float16
    else: dtype = torch.float32
    logger.info(f"Using precision: {precision} ({dtype})")

    if force_cpu: device_map = "cpu"; logger.info("Forcing model loading onto CPU.")
    elif torch.cuda.is_available(): device_map = "auto"; logger.info("CUDA available. Using device_map='auto'.")
    else: device_map = "cpu"; logger.info("CUDA not available. Loading model onto CPU.")

    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            token=hf_token,
            device_map=device_map,
            trust_remote_code=True
        )
        # Load tokenizer associated with the base model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token, padding_side="right")

        # --- Load LoRA adapter if path is provided ---
        if adapter_path:
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
                # You might want to merge adapters if needed for faster inference, but requires more memory
                # logger.info("Merging LoRA adapter...")
                # model = model.merge_and_unload()
                # logger.info("Adapter merged.")
            except Exception as peft_e:
                 logger.error(f"Failed to load PEFT adapter from {adapter_path}: {peft_e}", exc_info=True)
                 logger.warning("Proceeding with the base model only.")
        # --- End LoRA loading ---

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

# read_jsonl and write_jsonl remain the same
def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Reads a JSON Lines file into a list of dictionaries."""
    data = []
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Read {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading input file {file_path}: {e}")
        raise

def write_jsonl(file_path: Path, data: List[Dict[str, Any]]):
    """Writes a list of dictionaries to a JSON Lines file, overwriting it."""
    try:
        with file_path.open('w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Successfully wrote {len(data)} records to {file_path}")
    except Exception as e:
        logger.error(f"Error writing output file {file_path}: {e}")
        raise


# run_inference remains the same - it works on the loaded model (base or adapted)
@torch.no_grad()
def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[Dict[str, Any]],
    prompt_field: str,
    output_field: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool
) -> List[Dict[str, Any]]:
    """Runs inference on the data and adds generations."""
    model.eval()
    all_prompts = []
    valid_indices = []

    for i, item in enumerate(data):
        if prompt_field in item and isinstance(item[prompt_field], str):
            formatted_prompt = PROMPT_FORMAT.format(question=item[prompt_field])
            all_prompts.append(formatted_prompt)
            valid_indices.append(i)
        else:
            logger.warning(f"Skipping record {i+1}: Missing or invalid prompt field '{prompt_field}'. Found: {item.get(prompt_field)}")

    if not all_prompts:
        logger.warning("No valid prompts found.")
        return data

    logger.info(f"Processing {len(all_prompts)} valid prompts in batches of {batch_size}...")
    generations = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    # Determine generation kwargs, ensuring critical tokens exist
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen_kwargs["eos_token_id"] is None:
        logger.warning("EOS token ID is None, generation might not stop correctly.")
        # Optionally remove if it causes errors: del gen_kwargs["eos_token_id"]


    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]

        logger.info(f"Processing batch {i+1}/{num_batches} (samples {batch_start+1}-{batch_end})")

        # Use tokenizer's model_max_length if available, otherwise fallback (e.g., 1024)
        tokenizer_max_len = getattr(tokenizer, 'model_max_length', 1024)
        max_input_len = tokenizer_max_len - max_new_tokens
        if max_input_len <= 0:
             logger.error(f"Tokenizer max length ({tokenizer_max_len}) too small for max_new_tokens ({max_new_tokens}). Cannot generate.")
             generations.extend(["[ERROR: Input too long]"] * len(batch_prompts))
             continue

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len).to(model.device)

        try:
            outputs = model.generate(**inputs, **gen_kwargs)
            batch_completions = []
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                completion_tokens = output[input_length:]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                batch_completions.append(completion)
            generations.extend(batch_completions)
        except Exception as gen_e:
             logger.error(f"Error during generation for batch {i+1}: {gen_e}", exc_info=True)
             generations.extend(["[ERROR: Generation Failed]"] * len(batch_prompts))

        logger.info(f"Finished batch {i+1}/{num_batches}")

    output_data = data
    if len(generations) != len(valid_indices):
         logger.error(f"Mismatch between generations ({len(generations)}) and valid prompts ({len(valid_indices)}). Output alignment error.")

    gen_idx = 0
    for data_idx in valid_indices:
        if gen_idx < len(generations):
            output_data[data_idx][output_field] = generations[gen_idx]
            gen_idx += 1
        else:
            output_data[data_idx][output_field] = "[ERROR: Missing Generation]"

    for i, item in enumerate(output_data):
        if i not in valid_indices and output_field not in item:
             item[output_field] = "[SKIPPED: Invalid Input]"

    logger.info("Finished generating completions.")
    return output_data


def main():
    """Main function to orchestrate loading, inference, and writing."""
    args = parse_arguments()
    logger.info("Starting inference script...")
    logger.info(f"Arguments: {vars(args)}")

    if not args.input_file.is_file():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        # Pass adapter path to loading function
        model, tokenizer = load_model_tokenizer(
            args.model_name_or_path,
            args.precision,
            args.hf_token,
            args.force_cpu,
            args.adapter_path # Pass adapter path here
        )
    except Exception:
        logger.error("Exiting due to model loading failure.")
        sys.exit(1)

    try:
        input_data = read_jsonl(args.input_file)
    except Exception:
        logger.error("Exiting due to input file reading failure.")
        sys.exit(1)

    if not input_data:
        logger.warning("Input file is empty or contains no valid JSON objects. Exiting.")
        sys.exit(0)

    try:
        output_data = run_inference(
            model, tokenizer, input_data,
            args.prompt_field, args.output_field, args.batch_size,
            args.max_new_tokens, args.temperature, args.do_sample
        )
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}", exc_info=True)
        sys.exit(1)

    logger.warning(f"Overwriting the input file '{args.input_file}' with results...")
    try:
        write_jsonl(args.input_file, output_data)
    except Exception:
        logger.error("Exiting due to output file writing failure.")
        sys.exit(1)

    logger.info("Inference script finished successfully.")

if __name__ == "__main__":
    main()
