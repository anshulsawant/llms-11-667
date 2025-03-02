import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from calculator.utils import (
    load_asdiv,
    can_use_calculator,
    use_calculator,
    extract_label,
)
from tqdm.auto import tqdm
from datasets import Dataset


def main():
    """Initialize the pre-trained Pythia model"""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=16,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    """ Initialize a rank-16 LoRA module """
    model = get_peft_model(model, peft_config)

    dataset = load_asdiv()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = "<|padding|>"

    train(model, tokenizer, dataset["train"])
    evaluate(model, tokenizer, dataset["test"])

    print("Done!")


def evaluate(
    model: PeftModelForCausalLM, tokenizer: AutoTokenizer, test_dataset: Dataset
):
    test_data = test_dataset.to_pandas()
    test_data["label"] = pd.to_numeric(test_data["label"])

    generations_calc = []
    labels_calc = []
    generations_no_calc = []
    labels_no_calc = []

    for prefix in tqdm(test_data["text"]):
        answer_calc = inference(model, tokenizer, prefix, calculator=True)
        answer_no_calc = inference(model, tokenizer, prefix, calculator=False)
        generations_calc.append(answer_calc)
        generations_no_calc.append(answer_no_calc)

        labels_calc.append(extract_label(answer_calc))
        labels_no_calc.append(extract_label(answer_no_calc))

    test_data["answer-calc"] = generations_calc
    test_data["answer-no-calc"] = generations_no_calc
    test_data["label-calc"] = labels_calc
    test_data["label-no-calc"] = labels_no_calc
    test_data.to_json("pythia-1b-asdiv/eval.jsonl", lines=True, orient="records")

    acc_calc = np.isclose(test_data["label-calc"], test_data["label"]).mean()
    acc_no_calc = np.isclose(test_data["label-no-calc"], test_data["label"]).mean()
    print(
        f"test accuracy with calculator: {acc_calc:.1%}",
    )
    print(
        f"test accuracy without calculator: {acc_no_calc:.1%}",
    )
    print("Done!")


def train(
    model: PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    grad_acc_steps: int = 1,
    batch_size: int = 32,
    epochs: int = 5,
) -> None:
    tokenized_dataset = train_dataset.map(
        lambda x: {
            "input_ids": tokenizer.encode(x["text"] + x["target"])
            + [tokenizer.eos_token_id]
        }
    ).remove_columns(["text", "target", "label"])

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    step = 0
    for epoch_num in range(epochs):
        for batch in (pbar := tqdm(dataloader, desc=f"epoch {epoch_num+1}/{epochs}")):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            outputs["loss"].backward()

            if (step + 1) % grad_acc_steps == 0:
                opt.step()
                opt.zero_grad()

            pbar.set_postfix({"loss": outputs["loss"].item()})
            step += 1

    model.save_pretrained("pythia-1b-asdiv")


@torch.inference_mode(True)
def inference(
    model: PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    calculator: bool = True,
    max_tokens: int = 40,
) -> str:
    for i in range(max_tokens):
        if calculator and can_use_calculator(prefix):
            prefix = use_calculator(prefix)

        input_ids = tokenizer(prefix)["input_ids"]
        outputs = model(
            input_ids=torch.tensor([input_ids], dtype=torch.int64, device="cuda")
        )
        next_token_id = outputs["logits"][0][-1].argmax().item()
        if next_token_id == tokenizer.eos_token_id:
            break
        prefix = tokenizer.decode(input_ids + [next_token_id])
    return prefix


if __name__ == "__main__":
    main()
