import os
import torch
import re # Using 're' instead of 'regex' for standard library regex
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from tqdm import tqdm # For progress bar

# --- Configuration ---
base_model_id = "google/gemma-2-3b"
# *** IMPORTANT: Update this path if your tuned model is saved elsewhere ***
tuned_model_path = "./gemma-2-3b-gsm8k-full-sft"
dataset_name = "gsm8k"
dataset_config = "main" # Or "socratic"
compute_dtype = torch.bfloat16 # Or torch.float16 if bf16 is not supported

# --- One-Shot Example (for base model prompting) ---
one_shot_example = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May. #### 72
"""
# Add a separator between the one-shot example and the actual question
one_shot_separator = "\n---\n\n"


# --- Answer Extraction Function ---
def extract_final_answer(text):
    """
    Extracts the final numerical answer from text formatted like '... #### <answer>'.
    Handles potential commas in numbers.
    """
    # Search for the pattern '#### <number>' at the end of the string
    match = re.search(r"####\s*([\d\.,]+)$", text)
    if match:
        answer_str = match.group(1).replace(",", "") # Remove commas
        try:
            # Convert to float for consistent comparison
            return float(answer_str)
        except ValueError:
            print(f"Warning: Could not convert extracted answer '{answer_str}' to float.")
            return None
    else:
        # print(f"Warning: Could not find '#### <answer>' pattern in text: '{text}'")
        return None # Pattern not found

# --- Evaluation Function ---
def evaluate_model(model, tokenizer, test_dataset, device, use_one_shot=False):
    """
    Evaluates the model on the GSM8K test set.
    Includes option for one-shot prompting.
    """
    model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    # Construct the prompt prefix if using one-shot
    prompt_prefix = (one_shot_example + one_shot_separator) if use_one_shot else ""

    # Use tqdm for a progress bar
    for example in tqdm(test_dataset, desc=f"Evaluating (One-shot: {use_one_shot})"):
        question = example['question']
        reference_answer_text = example['answer']

        # Format the actual question part of the prompt
        question_prompt = f"Question: {question}\nAnswer:"
        # Combine prefix and question prompt
        full_prompt = prompt_prefix + question_prompt

        # Tokenize the full prompt
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=450).to(device) # Leave room for generation
        input_length = inputs["input_ids"].shape[1]

        # Generate the answer
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=150, # Max tokens for the answer part
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )

        # Decode only the newly generated tokens, skipping special tokens
        output_ids = outputs[0][input_length:] # Get only generated token IDs
        generated_answer_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Extract final numerical answers
        predicted_answer = extract_final_answer(generated_answer_text)
        reference_answer = extract_final_answer(reference_answer_text)

        # Compare answers
        if predicted_answer is not None and reference_answer is not None:
            if predicted_answer == reference_answer:
                correct_predictions += 1
        elif reference_answer is None:
             print(f"Warning: Could not extract reference answer from: {reference_answer_text}")
        # else: # predicted_answer is None
             # print(f"Warning: Could not extract predicted answer from: '{generated_answer_text}'")


        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# --- Main Evaluation Logic ---

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: No GPU detected. Evaluation will run on CPU and be very slow.")
    device = torch.device("cpu")

# Load the test dataset
print(f"\nLoading dataset: {dataset_name} ({dataset_config}) test split...")
gsm8k_test = load_dataset(dataset_name, name=dataset_config, split="test")
print(f"Loaded {len(gsm8k_test)} examples from the test split.")

# --- 1. Evaluate Untuned Model (with One-Shot Prompt) ---
print(f"\n--- Evaluating Untuned Model: {base_model_id} (Using One-Shot Prompt) ---")
try:
    # Load base model and tokenizer
    print("Loading untuned model and tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    print("Model and tokenizer loaded.")

    # Run evaluation with use_one_shot=True
    base_accuracy = evaluate_model(base_model, base_tokenizer, gsm8k_test, device, use_one_shot=True)
    print(f"\nUntuned Model Accuracy (One-Shot) on GSM8K Test Set: {base_accuracy:.4f}")

    # Clean up memory before loading the next model
    print("Cleaning up base model resources...")
    del base_model
    del base_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception as e:
    print(f"Error evaluating untuned model: {e}")
    print("Skipping untuned model evaluation.")

# --- 2. Evaluate Tuned Model (Zero-Shot Prompt) ---
print(f"\n--- Evaluating Tuned Model: {tuned_model_path} (Using Zero-Shot Prompt) ---")
if not os.path.exists(tuned_model_path):
    print(f"Error: Tuned model path not found: {tuned_model_path}")
    print("Skipping tuned model evaluation. Ensure the path is correct and the model was saved.")
else:
    try:
        # Load tuned model and tokenizer
        print("Loading fine-tuned model and tokenizer...")
        tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)

        tuned_model = AutoModelForCausalLM.from_pretrained(
            tuned_model_path,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        print("Model and tokenizer loaded.")

        # Run evaluation with use_one_shot=False (default)
        tuned_accuracy = evaluate_model(tuned_model, tuned_tokenizer, gsm8k_test, device, use_one_shot=False)
        print(f"\nFine-tuned Model Accuracy (Zero-Shot) on GSM8K Test Set: {tuned_accuracy:.4f}")

        # Clean up memory (optional, as script ends here)
        print("Cleaning up tuned model resources...")
        del tuned_model
        del tuned_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating fine-tuned model: {e}")
        print("Ensure the tuned model path is correct and contains a valid saved model.")

print("\nEvaluation script finished.")
