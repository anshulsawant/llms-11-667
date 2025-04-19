# src/sft_project/utils.py
"""Utility functions for the SFT project."""

import os
import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import transformers
from omegaconf import OmegaConf, DictConfig, open_dict # Import open_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
# Note: PeftModel import is NOT needed here, as adapters are loaded in the main scripts.

# --- Added WandB import ---
try:
    import wandb
except ImportError:
    wandb = None
# --- End WandB import ---


# Configure logging (optional, could let main scripts handle logging)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use logger for potential warnings/errors in utils

# --- Configuration Loading ---

def load_config(override_config_path: str, base_config_path: str = "config.yaml") -> DictConfig:
    """Loads base config and merges override config."""
    try:
        base_cfg = OmegaConf.load(base_config_path)
        override_cfg = OmegaConf.load(override_config_path)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
        # Optional: Add validation logic here if needed
        # logger.debug(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg)}") # Debug log
        return cfg
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading or merging configuration files: {e}")
        raise

# --- Model and Tokenizer Loading ---

def load_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the base model and tokenizer based on the configuration.
    Handles precision, access token, trust_remote_code, and attn_implementation.
    NOTE: This function ONLY loads the base model specified in cfg.model.name.
          Adapter/checkpoint loading happens in the calling script (evaluate.py/inference.py).
    """
    model_name = cfg.model.name
    access_token = cfg.model.get("access_token", None)
    trust_remote_code = cfg.model.get("trust_remote_code", False)
    attn_impl = cfg.model.get("attn_implementation", None) # e.g., "flash_attention_2"

    # Determine precision
    precision = "bf16" if cfg.training.get("bf16") else "fp16" if cfg.training.get("fp16") else "fp32"
    torch_dtype = torch.float32 # Default
    if precision == "bf16":
        torch_dtype = torch.bfloat16
        logger.info(f"Loading model {model_name} in bfloat16 precision.")
    elif precision == "fp16":
        torch_dtype = torch.float16
        logger.info(f"Loading model {model_name} in float16 precision.")
    else:
        logger.info(f"Loading model {model_name} in float32 precision.")


    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": access_token,
        "torch_dtype": torch_dtype,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
        logger.info(f"Using attention implementation: {attn_impl}")

    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info(f"Base model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading base model {model_name}: {e}", exc_info=True)
        raise # Re-raise the exception

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=access_token,
            trust_remote_code=trust_remote_code,
            padding_side='left' # Often set to left for causal LMs
        )
        logger.info(f"Tokenizer for '{model_name}' loaded successfully.")

        # Set pad token if missing
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("Tokenizer missing pad_token, setting to eos_token.")
            else:
                # Add a default pad token if EOS is also missing (less common)
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings if new token added
                logger.warning("Tokenizer missing pad_token and eos_token. Added '[PAD]' as pad_token.")


    except Exception as e:
        logger.error(f"Error loading tokenizer for {model_name}: {e}", exc_info=True)
        raise # Re-raise the exception

    return model, tokenizer

# --- Prompt Formatting ---

def format_prompt(sample: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    Formats a data sample into a prompt string based on config.

    Args:
        sample: A dictionary representing a single data point (e.g., from dataset).
        cfg: The configuration object, expected to have `cfg.dataset.prompt_format`.

    Returns:
        A dictionary containing at least 'prompt' and potentially 'ground_truth_answer'.
        The exact structure depends on the prompt format.
    """
    prompt_format = cfg.dataset.get("prompt_format", "instruction") # Default format
    formatted_sample = {}

    # --- GSM8K Example Format ---
    if cfg.dataset.name == "gsm8k":
        question = sample.get("question", "")
        answer = sample.get("answer", "") # Keep the ground truth if available

        # Basic instruction format - assumes model generates reasoning + answer
        # prompt_format might be defined in config like "Question: {question}\nAnswer: "
        try:
             prompt = cfg.dataset.prompt_format.format(question=question)
             formatted_sample["prompt"] = prompt
             formatted_sample["ground_truth_answer"] = answer # Pass ground truth along
        except KeyError as e:
             logger.warning(f"KeyError formatting GSM8K prompt. Check `prompt_format` in config and sample keys. Error: {e}")
             formatted_sample["prompt"] = f"Question: {question}\nAnswer:" # Fallback
             formatted_sample["ground_truth_answer"] = answer
        except Exception as e:
             logger.error(f"Unexpected error formatting GSM8K prompt: {e}", exc_info=True)
             formatted_sample["prompt"] = f"Question: {question}\nAnswer:" # Fallback
             formatted_sample["ground_truth_answer"] = answer


    # --- Add other dataset format handlers here ---
    # elif cfg.dataset.name == "alpaca":
    #     instruction = sample.get("instruction", "")
    #     input_text = sample.get("input", "")
    #     # ... (Alpaca formatting logic) ...
    #     formatted_sample["prompt"] = prompt
    #     formatted_sample["ground_truth_answer"] = sample.get("output", "")

    # --- Default/Placeholder Format ---
    else:
        # Assume a simple instruction format if not specified
        instruction = sample.get("instruction", sample.get("question", "")) # Try common keys
        input_text = sample.get("input", "")
        output_text = sample.get("output", sample.get("answer", "")) # Try common keys
        try:
            # Try formatting based on config if available
            prompt = cfg.dataset.prompt_format.format(instruction=instruction, input=input_text)
        except:
             # Fallback to a generic format
            if input_text:
                 prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
            else:
                 prompt = f"Instruction: {instruction}\nOutput:"
        formatted_sample["prompt"] = prompt
        formatted_sample["ground_truth_answer"] = output_text

    if not formatted_sample.get("prompt"):
         logger.warning(f"Could not format prompt for sample: {sample}")

    return formatted_sample


# --- Answer Extraction ---

def extract_gsm8k_answer(completion: str) -> Optional[str]:
    """Extracts the final numerical answer from a GSM8K completion string."""
    if not isinstance(completion, str): return None # Handle non-string input

    # Regex to find the final answer marked with ####
    match = re.search(r"####\s*([\d\.,]+)", completion)
    if match:
        answer = match.group(1).strip().replace(',', '')
        # Optional: Add validation if it's a number?
        try:
            float(answer) # Check if it can be converted to float
            return answer
        except ValueError:
             logger.warning(f"Extracted GSM8K answer '{answer}' is not a valid number.")
             return None # Return None if it's not a valid number despite matching pattern
    else:
        # Fallback: Try to find the last number in the string
        numbers = re.findall(r"[\d\.]+", completion)
        if numbers:
            logger.debug(f"GSM8K '####' pattern not found, using last number: {numbers[-1]}")
            return numbers[-1]
        else:
            logger.debug(f"Could not extract GSM8K answer from: {completion[:100]}...") # Log snippet
            return None # Return None if the pattern is not found


# --- JSON Lines Read/Write ---

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Reads a JSON Lines file and returns a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip(): # Ensure line is not empty
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {line_num} in {file_path}: {line.strip()} - Error: {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        raise
    return data

def write_jsonl(file_path: str, data: List[Dict[str, Any]]):
    """Writes a list of dictionaries to a JSON Lines file."""
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                try:
                    f.write(json.dumps(item) + '\n')
                except TypeError as e:
                     logger.warning(f"Skipping item due to JSON serialization error: {item} - Error: {e}")
    except IOError as e:
         logger.error(f"Could not write to file {file_path}: {e}", exc_info=True)
         raise
    except Exception as e:
        logger.error(f"An unexpected error occurred writing to {file_path}: {e}", exc_info=True)
        raise

# --- WandB Initialization ---

def init_wandb(cfg: DictConfig, accelerator: Accelerator) -> Optional[Any]:
    """Initializes Weights & Biases run based on configuration."""
    if not wandb:
        logger.warning("WandB library not found. Skipping WandB initialization.")
        return None

    # Check if WandB reporting is enabled in the config
    report_to = cfg.training.get("report_to", [])
    if "wandb" not in report_to:
        logger.info("WandB reporting not enabled in config (training.report_to). Skipping.")
        return None

    # Proceed only on the main process
    if accelerator.is_main_process:
        try:
            wandb_cfg = cfg.get("wandb")
            if not wandb_cfg:
                logger.warning("`wandb:` section not found in config. Skipping WandB initialization.")
                return None

            # Ensure run_name is set, potentially combining project/model info
            run_name = wandb_cfg.get("run_name")
            if not run_name:
                 model_slug = slugify(cfg.model.get("name", "unknown_model").split('/')[-1])
                 dataset_slug = slugify(cfg.dataset.get("name", "unknown_dataset"))
                 run_name = f"{model_slug}_{dataset_slug}_{cfg.get('tuning_method','sft')}"
                 logger.info(f"WandB run name not set, generated: {run_name}")
                 # Optionally update the config object if needed elsewhere (use with caution)
                 # with open_dict(cfg): cfg.wandb.run_name = run_name

            # Initialize WandB
            run = wandb.init(
                project=wandb_cfg.get("project", "sft_project"), # Default project name
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False), # Log config
                dir=cfg.training.get("output_dir", "./wandb_runs"), # Set WandB directory
                resume="allow", # Allow resuming runs
                # Add other wandb.init options if needed (e.g., entity, tags)
                # entity=wandb_cfg.get("entity", None),
                # tags=wandb_cfg.get("tags", []),
            )
            logger.info(f"WandB run initialized successfully. Run name: {run_name}, Project: {wandb_cfg.get('project')}")
            return run
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
            return None
    else:
        # Return None on non-main processes
        return None

# --- Slugify Helper ---
def slugify(value: str) -> str:
    """Normalizes string, removes invalid chars, and converts spaces to hyphens."""
    if not isinstance(value, str): value = str(value) # Ensure string type
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    if not value: return "na" # Handle empty slugs
    return value

# --- Optional: Add other utility functions as needed ---
