"""Utility functions for the SFT project."""

import os
import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import transformers
from omegaconf import OmegaConf, DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
# Note: PeftModel import is NOT needed here, as adapters are loaded in the main scripts.

# Configure logging (optional, could let main scripts handle logging)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# --- Configuration Loading ---

def load_config(override_config_path: str, base_config_path: str = "config.yaml") -> DictConfig:
    """Loads base config and merges override config."""
    base_cfg = OmegaConf.load(base_config_path)
    override_cfg = OmegaConf.load(override_config_path)
    cfg = OmegaConf.merge(base_cfg, override_cfg)
    # Optional: Add validation logic here if needed
    # print(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg)}") # Debug print
    return cfg

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
        print(f"Loading model {model_name} in bfloat16 precision.") # Use logger if configured
    elif precision == "fp16":
        torch_dtype = torch.float16
        print(f"Loading model {model_name} in float16 precision.") # Use logger if configured
    else:
        print(f"Loading model {model_name} in float32 precision.") # Use logger if configured


    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": access_token,
        "torch_dtype": torch_dtype,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
        print(f"Using attention implementation: {attn_impl}") # Use logger if configured

    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print(f"Base model '{model_name}' loaded successfully.") # Use logger if configured
    except Exception as e:
        print(f"Error loading base model {model_name}: {e}") # Use logger if configured
        raise # Re-raise the exception

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=access_token,
            trust_remote_code=trust_remote_code,
            # Add padding side if needed, often depends on model/fine-tuning
            # padding_side='left' # Or 'right' depending on model/task
        )
        print(f"Tokenizer for '{model_name}' loaded successfully.") # Use logger if configured

        # Set pad token if missing
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("Tokenizer missing pad_token, setting to eos_token.") # Use logger if configured
            else:
                # Add a default pad token if EOS is also missing (less common)
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings if new token added
                print("Tokenizer missing pad_token and eos_token. Added '[PAD]' as pad_token.") # Use logger if configured


    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}") # Use logger if configured
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

        # Basic instruction format
        prompt = f"Question: {question}\nAnswer:" # Model should generate the reasoning and final answer
        formatted_sample["prompt"] = prompt
        formatted_sample["ground_truth_answer"] = answer # Pass ground truth along

    # --- Add other dataset format handlers here ---
    # elif cfg.dataset.name == "alpaca":
    #     instruction = sample.get("instruction", "")
    #     input_text = sample.get("input", "")
    #     if input_text:
    #         prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    #     else:
    #         prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    #     formatted_sample["prompt"] = prompt
    #     formatted_sample["ground_truth_answer"] = sample.get("output", "")

    # --- Default/Placeholder Format ---
    else:
        # Assume a simple instruction format if not specified
        instruction = sample.get("instruction", sample.get("question", "")) # Try common keys
        input_text = sample.get("input", "")
        if input_text:
             prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
             prompt = f"Instruction: {instruction}\nOutput:"
        formatted_sample["prompt"] = prompt
        formatted_sample["ground_truth_answer"] = sample.get("output", sample.get("answer", ""))

    if not formatted_sample.get("prompt"):
         print(f"Warning: Could not format prompt for sample: {sample}") # Use logger if configured

    return formatted_sample


# --- Answer Extraction ---

def extract_gsm8k_answer(completion: str) -> Optional[str]:
    """Extracts the final numerical answer from a GSM8K completion string."""
    # Regex to find the final answer marked with ####
    match = re.search(r"####\s*([\d\.,]+)", completion)
    if match:
        answer = match.group(1).strip().replace(',', '')
        # Optional: Add validation if it's a number?
        try:
            float(answer) # Check if it can be converted to float
            return answer
        except ValueError:
            return None # Return None if it's not a valid number despite matching pattern
    else:
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
                        # Handle potential errors in specific lines if needed
                        print(f"Warning: Skipping invalid JSON line {line_num} in {file_path}: {line.strip()} - Error: {e}") # Use logger if configured
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}") # Use logger if configured
        # Decide if you want to raise the error or return empty list
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}") # Use logger if configured
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
                     print(f"Warning: Skipping item due to JSON serialization error: {item} - Error: {e}") # Use logger if configured
    except IOError as e:
         print(f"Error: Could not write to file {file_path} - Error: {e}") # Use logger if configured
         # Decide if you want to raise the error
         raise
    except Exception as e:
        print(f"An unexpected error occurred writing to {file_path}: {e}") # Use logger if configured
        raise

# --- Optional: Add other utility functions as needed ---
# For example: WandB initialization, dataset preprocessing steps, etc.
