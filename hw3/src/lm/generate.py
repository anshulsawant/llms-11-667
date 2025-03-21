import argparse
import json
import os
import math

import tiktoken
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    
    return F.softmax(logits / temperature, -1)


class FakeModel:
    """ For dev. """
    def __init__(self, vocab_size):
        self.i = 0
        self.vocab_size = vocab_size
        
    def forward(self, encoded, mask):
        B = encoded.size(0)
        S = encoded.size(-1)
        token = torch.arange(self.i, self.i + B).view(encoded.size(0), -1)
        y = F.one_hot(token.repeat((1, S)), num_classes=self.vocab_size)
        return y


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """

    N = len(prefixes)
    B = max(batch_size, N)
    neg_log_prob_sum = 0
    generations = []

    for s in range(0, N, B):
        bs = min(s + B, N) - s
        encoded = tokenizer.encode_batch(prefixes[s:s + bs])
        lens = [len(p) for p in prefixes[s:s + bs]]
        min_len = min(lens)
        max_len = max(lens)
        encoded = [[tokenizer.eot_token] * (max_len - len(e)) + e for e in encoded]
        encoded = torch.tensor(encoded).to(device=device)
        mask = (encoded != tokenizer.eot_token).to(dtype=torch.float32).to(device=device)
        gen = torch.zeros(bs, max_new_tokens).to(device=device)
        for i in range(max_len, max_len + max_new_tokens):
            logits = model.forward(encoded, mask)[:, i-1, :]
            probs = softmax_with_temperature(logits, temperature).view(bs, -1)
            samples = torch.multinomial(probs, num_samples=1).view(bs, -1)
            neg_log_prob_sum += -torch.log(probs[torch.arange(bs).view(samples.size()), samples]).sum()
            encoded = torch.cat((encoded, samples), 1)
            mask = torch.cat((mask, torch.ones_like(samples)), 1) 
        lens = torch.tensor(lens).view(-1, 1).to(device=device)
        generated_tokens = encoded[:, max_len:(max_len + max_new_tokens)]
        batch_gen = tokenizer.decode_batch(generated_tokens.tolist()) 
        generations += batch_gen

    perplexity = torch.exp(neg_log_prob_sum/(N * max_new_tokens)).item()

    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
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

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
