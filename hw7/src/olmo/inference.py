"""
This is an example code. Feel free to modify it when needed.
"""

import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from olmo.utils import determine_device
from huggingface_hub import login


MODEL_MAP = {
    "pretrained": "allenai/OLMo-7B-hf",
    "sft": "allenai/OLMo-7B-SFT-hf",
    "instruct": "allenai/OLMo-7B-Instruct-hf",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "tulu": "allenai/llama-3.1-tulu-2-8b",
    "qwen" : "Qwen/Qwen2.5-7B-Instruct"
}


def prepare_messages(file_path: str, model_type: str) -> List[Union[str, Dict[str, str]]]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if model_type == "pretrained":
        return [item["prompt"] for item in data]
    else:
        return [{"role": "user", "content": item["prompt"]} for item in data]


@torch.inference_mode()
def generate_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Union[str, Dict[str, str]]],
    output_file: str,
    model_type: str,
    device: str
):
    with open(output_file, "w") as f:
        for message in tqdm(messages):
            prompt = message if isinstance(message, str) else message["content"]
            
            if model_type == "pretrained":
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
            else:
                inputs = tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=True, return_tensors="pt")

            if model_type != "pretrained" and not isinstance(inputs, dict):
                inputs = {'input_ids': inputs}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = model.generate(**inputs, max_new_tokens=300)
            response_decoded = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

            if model_type != "pretrained":
                response_decoded = response_decoded.split("\n<|assistant|>\n")[-1]

            json.dump({"prompt": prompt, "output": response_decoded}, f)
            f.write("\n")


def main():
    login()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", 
        type=str,
        required=True, 
        choices=MODEL_MAP.keys(),
        help="The type of the model: pretrained, sft, or instruct"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        default="data/prompts.jsonl",
        help="A jsonl file where each line has an input prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="A jsonl file where each line will have the input and the corresponding output"
    )
    args = parser.parse_args()

    device = determine_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model_type], 
        torch_dtype="float16"  # we need half-precision to fit into the AWS machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_type])

    messages = prepare_messages(args.prompts, args.model_type)
    generate_output(model, tokenizer, messages, args.output, args.model_type, device)

    print("Done!")


if __name__ == "__main__":
    main()
