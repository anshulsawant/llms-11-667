from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)

from dataclasses import dataclass, field
import logging
import torch
from datasets import load_dataset
import math
import sys

import shutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if len(sys.argv) == 1:
    raise ValueError("Missing configuration file. Please follow the instructions in the README.")

config_file = sys.argv[1] if len(sys.argv) == 2 else sys.argv[2] # deepspeed will add --local_rank as arg

@dataclass
class ModelConfig:
    model_to_train: str = field(default="models/GPTNeoX-160M")
    seq_len: int = field(default=512)
    attention_type: str = field(default="flash_attention_2")
    dataset: str = field(default="wikitext")

    def __post_init__(self):
        if self.seq_len not in {512, 2048}:
            raise ValueError("seq_len must be either 512 or 2048.")
        if self.dataset not in {"wikitext", "minipile"}:
            raise ValueError("dataset must be either wikitext or minipile.")
        if self.dataset == "minipile" and self.seq_len != 2048:
            raise ValueError("minipile dataset only has 2048 seq len split.")

        
parser = HfArgumentParser((ModelConfig, TrainingArguments))
model_config, training_args = parser.parse_json_file(json_file=config_file)

logger.info(f"Base model: {model_config.model_to_train}")
logger.info(f"Saving to: {training_args.output_dir}")

model = AutoModelForCausalLM.from_pretrained(model_config.model_to_train, 
                                             torch_dtype=torch.bfloat16, 
                                             attn_implementation=model_config.attention_type,
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_config.model_to_train)
tokenizer.pad_token = tokenizer.eos_token

if model_config.dataset == "wikitext":
    train_dataset = load_dataset("cjiao/wikitext-tokenized", split=f"train_{model_config.seq_len}")

    valid_dataset_512 = load_dataset("cjiao/wikitext-tokenized", split="validation_512")
    valid_dataset_2048 = load_dataset("cjiao/wikitext-tokenized", split="validation_2048")

    valid_during_training = {
        512: valid_dataset_512,
        2048: valid_dataset_2048
    }

elif model_config.dataset == "minipile":
    full_dataset = load_dataset("cjiao/minipile-train", split=f"train_{model_config.seq_len}")

    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    train_dataset = dataset_split["train"]
    valid_dataset_2048 = dataset_split["test"]  # minipile only has 2048 split

    valid_during_training = {
        2048: valid_dataset_2048
    }


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_during_training[model_config.seq_len],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)


if model_config.dataset == "wikitext":
    results_512 = trainer.evaluate(eval_dataset=valid_dataset_512)
    print(f"Perplexity (512): {math.exp(results_512['eval_loss']):.2f}")

    results_2048 = trainer.evaluate(eval_dataset=valid_dataset_2048)
    print(f"Perplexity (2048): {math.exp(results_2048['eval_loss']):.2f}")

else:
    results_2048 = trainer.evaluate(eval_dataset=valid_dataset_2048)
    print(f"Perplexity (2048): {math.exp(results_2048['eval_loss']):.2f}")


source_file = os.path.join(model_config.model_to_train, "modeling_custom.py")
destination_file = os.path.join(training_args.output_dir, "modeling_custom.py")

# HACK: huggingface recent versions have a bug when saving custom modeling files. this ensures it is properly saved.
try:
    shutil.copy2(source_file, destination_file)
except FileNotFoundError:
    logger.error(f"File not found: {source_file}")
except Exception as e:
    logger.error(f"Error copying file: {e}")