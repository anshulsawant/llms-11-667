# src/sft_project/utils.py
"""Utility functions shared between training and evaluation scripts."""

import logging
import re
from typing import Dict, Any, Optional

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig, MissingMandatoryValue
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

logger = logging.getLogger(__name__) # Use logger defined in calling script or configure here

# --- Configuration Loading ---
def load_config(override_config_path: str, base_config_path: str = "config.yaml") -> DictConfig:
    """
    Loads a base configuration and merges it with an override configuration.
    Performs validation on the merged configuration.
    """
    logger.info(f"Loading base configuration from: {base_config_path}")
    try: base_conf = OmegaConf.load(base_config_path)
    except Exception as e: logger.error(f"Error loading base config: {e}"); raise
    logger.info(f"Loading override configuration from: {override_config_path}")
    try: override_conf = OmegaConf.load(override_config_path)
    except Exception as e: logger.error(f"Error loading override config: {e}"); raise
    logger.info("Merging configurations...")
    try:
        merged_conf = OmegaConf.merge(base_conf, override_conf)
        logger.info("Configurations merged successfully.")
        # --- Validation ---
        logger.info("Validating merged configuration...")
        # Basic checks - add more as needed
        if not merged_conf.get("model") or not merged_conf.model.get("name"): raise ValueError("Config missing 'model.name'.")
        if not merged_conf.get("training") or not merged_conf.training.get("output_dir"): raise ValueError("Config missing 'training.output_dir'.")
        if merged_conf.get("tuning_method") == "lora":
              if not merged_conf.get("lora_config"): raise ValueError("Missing lora_config.")
              # Add more checks if needed
        # Add other validation checks from previous sft_script.py version if desired
        logger.info("Merged configuration validated.")
        # --- End validation ---
        return merged_conf
    except Exception as e: logger.error(f"Error merging/validating config: {e}"); raise

# --- WandB Initialization ---
def init_wandb(cfg: DictConfig, job_type: str = "train"):
    """Initializes Weights & Biases if enabled in config."""
    if "wandb" in cfg.training.get("report_to", []) and cfg.get("wandb"):
        logger.info(f"Initializing WandB for job type: {job_type}...")
        try:
            run_name = cfg.wandb.get("run_name", f"sft-run-{job_type}")
            # Append tuning method if not already in name from override
            if cfg.get("tuning_method") and cfg.tuning_method not in run_name:
                 run_name += f"-{cfg.tuning_method}"

            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
                job_type=job_type
            )
            # Only watch during training
            if job_type == "train":
                watch_log = cfg.wandb.get("watch", "gradients")
                if cfg.get("tuning_method") == "lora" and watch_log == "gradients":
                    logger.warning("WandB watching gradients with LoRA might not be optimal.")
                wandb.watch(models=None, log=watch_log, log_freq=100)

            logger.info("WandB initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            # Optionally modify cfg to remove wandb from report_to if init fails
            # cfg.training.report_to = [r for r in cfg.training.report_to if r != "wandb"]
            return False
    else:
        logger.info("WandB reporting not configured.")
        return False

# --- Model Loading ---
def load_model_and_tokenizer(cfg: DictConfig) -> (AutoModelForCausalLM, AutoTokenizer):
    """Loads the base model and tokenizer based on the configuration."""
    logger.info(f"Loading base model: {cfg.model.name}")
    model_name = cfg.model.name
    access_token = cfg.model.get("access_token", None) # Use get for safety
    trust_remote_code = cfg.model.get("trust_remote_code", False)

    model_kwargs = {"trust_remote_code": trust_remote_code, "token": access_token}
    attn_impl = cfg.model.get("attn_implementation", None)
    if attn_impl: logger.info(f"Setting attn_implementation='{attn_impl}'"); model_kwargs["attn_implementation"] = attn_impl

    # Determine precision from training args (or inference args if separate)
    precision = "bf16" if cfg.training.get("bf16") else "fp16" if cfg.training.get("fp16") else "fp32"
    if precision == "bf16": logger.info("Using bfloat16 precision."); model_kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "fp16": logger.info("Using float16 precision."); model_kwargs["torch_dtype"] = torch.float16
    else: logger.info("Using default float32 precision.")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=access_token,
        trust_remote_code=trust_remote_code,
        padding_side="left", # Use left padding
        use_fast=True,
    )

    # Set pad token if missing - essential for left padding
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Base model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Prompt Formatting ---
def format_prompt(example: Dict[str, Any], cfg: DictConfig) -> Dict[str, str]:
    """Formats a single example using the templates from the config."""
    # Ensure dataset section and keys exist
    if not cfg.get("dataset") or not cfg.dataset.get("prompt_format") or not cfg.dataset.get("response_template"):
         raise ValueError("Config missing dataset.prompt_format or dataset.response_template")
    if 'question' not in example or 'answer' not in example:
         raise ValueError("Input example missing 'question' or 'answer' field")

    prompt = cfg.dataset.prompt_format.format(question=example['question'])
    response = cfg.dataset.response_template.format(answer=example['answer'])
    text = prompt + response # Used for training tokenization
    return {"text": text, "prompt": prompt, "ground_truth_answer": example['answer']}

# --- Answer Extraction ---
def extract_gsm8k_answer(completion: str) -> str | None:
    """Extracts the final numerical answer from GSM8K generated text."""
    if not isinstance(completion, str): return None # Handle non-string input
    match = re.search(r"####\s*([\d.,]+)\s*$", completion)
    if match:
        return match.group(1).replace(",", "")
    # Optional: Add fallback logic if #### marker is missing?
    # E.g., find last number in the string? More brittle.
    # numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
    # if numbers: return numbers[-1].replace(",", "")
    return None
