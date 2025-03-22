import argparse
import json
import os

import random 

import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM

def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    Note: you should implement this function and learn how to load tokenizer and model using the `transformers` library. For Pythia, use `eos_token` for `pad_token`, and set `padding_side` to "left" in the tokenizer.
    """

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to(device)

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

    Note: you should implement a batched version of this function by directly using the `model.generate` method.
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

def get_rag(query_id, doc2text, query2docs, top_n, shuffle):
    """Returns the text of the top-documents associated with each query.

    Args:
        query_id (_type_): query identifier
        doc2text (dict): maps document identifier to text
        query2docs (dict): maps query identifier to top-N doc ids
        top_n (int): number of top-docs to return
        shuffle (bool): whether or not to shuffle the top-N


    Returns:
        str: white-space separeted text of top-N documents.
    """
    raise NotImplementedError()


def apply_prompt(prefixes, trec_run, doc2text, query2docs):

    new_prefixes = []

    for prefix, query_id in prefixes:
        if trec_run is not None:
            rag = get_rag(query_id, doc2text, query2docs, top_n=1, shuffle=False)
        else:
            rag = ""
        
        new_prefix = f"{rag}<|user|>\nAnswer this question: {prefix}\n<|assistant|>\n"

        new_prefixes.append(new_prefix)

    return new_prefixes


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
        "--temperature", type=float, default=1, help="temperature in sampling"
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
    parser.add_argument(
        "--augmentation_run",
        type=str,
        default=None,
        help="trec formatted run for RAG",
    )

    args = parser.parse_args()

    # If necessary, load passage corpus for RAG, and get top-N document for each question.
    if args.augmentation_run is not None:
        from datasets import load_dataset

        dataset = load_dataset('jmvcoelho/toy-corpus', split='train')
        
        docid_to_text = {}
        raise NotImplementedError()
        #TODO: parse the huggingface dataset, and populate docid_to_text, mapping the document identifier to its repective content.

        qid_to_topdocs = {}

        with open(args.augmentation_run, 'r') as h:
            raise NotImplementedError()
            #TODO: Read the run file, and populate qid_to_topdocs, mapping query ids to a list of top-document ids.


    else: # this means that RAG won't be performed.
        docid_to_text = None
        qid_to_topdocs = None

    with open(args.prefixes) as f:
        prefixes = [(json.loads(line)["prefix"],  json.loads(line)["qid"])for line in f]
        prefixes = apply_prompt(prefixes, args.augmentation_run, docid_to_text, qid_to_topdocs)

    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    batch_size = args.batch_size
    output_dir = args.output_dir
    device = determine_device()

    # initialize pythia tokenizer and model
    model_name = "allenai/open-instruct-pythia-6.9b-tulu"
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

    out_file_name = "generation_pythia.jsonl" if args.augmentation_run is None else "generation_pythia_with_RAG.jsonl"

    generation_path = os.path.join(output_dir, out_file_name)
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()