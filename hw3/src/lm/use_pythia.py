import argparse
import json
import os

import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm.utils import determine_device, enable_tf32


def initialize_pythia(
    model_name: str,
    device: str,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Initialize pythia tokenizer and model

    Args:
        model_name: the name of the model to load
        device: device to put the model on

    Returns:
        tokenizer: the tokenizer
        model: the model

    Note: you should learn how to load tokenizer and model using the `transformers` library. For Pythia, use `eos_token` for `pad_token`, and set `padding_side` to "left" in the tokenizer.
    """

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return tokenizer, model


@torch.inference_mode()
def generate_pythia(
    model: AutoModelForCausalLM,
    device: str,
    tokenizer: AutoTokenizer,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax; should be greater than 0 for sampling

    Returns:
        generations: a list of strings (continuations to prefixes)

    Note: a batched version of this function by directly using the `model.generate` method.
    """

    generations = []

    # do batched generation
    for i in trange(0, len(prefixes), batch_size):
        # tokenize the prefixes
        batch_prefixes = prefixes[i : i + batch_size]
        tokenized_prefixes = tokenizer(
            batch_prefixes, return_tensors="pt", padding=True
        ).to(device)

        outputs = model.generate(
            **tokenized_prefixes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuations = outputs[:, tokenized_prefixes["input_ids"].shape[1] :]

        # decode the generated tokens
        batch_generations = tokenizer.batch_decode(
            continuations, skip_special_tokens=True
        )
        generations.extend(batch_generations)

    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="number of prefixes to batch together during generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory to save the generated outputs",
    )

    args = parser.parse_args()
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    batch_size = args.batch_size
    output_dir = args.output_dir
    device = determine_device()

    # initialize pythia tokenizer and model
    model_name = "EleutherAI/pythia-1.4b"
    tokenizer, model = initialize_pythia(model_name, device)

    # generate and save outputs
    model.eval()
    generations = generate_pythia(
        model,
        device,
        tokenizer,
        prefixes,
        batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(output_dir, "generation_pythia.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()