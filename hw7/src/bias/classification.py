import argparse
import numpy as np
from dotenv import load_dotenv
import os
import litellm
import logging
from tqdm import tqdm

from bias.utils import read_data, create_prompt, get_responses, get_label_probs, calibrate, eval_accuracy


litellm.set_verbose = False
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("litellm").propagate = False
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

load_dotenv()


def main(
    seed: int,
    num_shots: int,
    q_prefix: str,
    a_prefix: str,
    train_size: int,
    test_size: int,
    content_free_input: str
) -> None:
    """
    Main function to run a binary sentiment classification using few-shot learning and calibration.
    
    Args:
        seed (int): Random seed for reproducibility
        num_shots (int): Number of examples to use for few-shot learning
        q_prefix (str): Prefix of the review
        a_prefix (str): Prefix of the sentiment
        train_size (int): Number of training samples to select from
        test_size (int): Number of testing samples to evaluate
        content_free_input (str): Content-free input text used for calibration
        
    Notes:
        Prints accuracy before and after calibration, and calibration vector
    """

    # Load dataset
    train_sentences, train_labels, test_sentences, test_labels = read_data(seed, train_size, test_size)

    # Select few-shot samples
    np.random.seed(seed)
    idxs = np.random.choice(len(train_sentences), num_shots, replace=False)
    few_shot_sentences = [train_sentences[i] for i in idxs]
    few_shot_labels = [train_labels[i] for i in idxs]

    # Create prompts
    prompts = []
    for test_sentence in tqdm(test_sentences, desc="Create prompts"):
        prompt = create_prompt(q_prefix, a_prefix, few_shot_sentences, few_shot_labels, test_sentence)
        prompts.append(prompt)
    
    # Get raw responses from the model
    raw_responses = get_responses(prompts)

    # Get probabilities of each label
    all_label_probs = get_label_probs(raw_responses, few_shot_sentences, few_shot_labels, test_sentences, q_prefix, a_prefix)

    # Calibration algorithm
    p_cf = calibrate(content_free_input, few_shot_sentences, few_shot_labels, q_prefix, a_prefix)

    # Calculate classification accuracy
    acc_before = eval_accuracy(all_label_probs, test_labels)
    acc_after = eval_accuracy(all_label_probs, test_labels, p_cf)

    print(f"Accuracy before calibration: {acc_before}")
    print(f"Accuracy after calibration: {acc_after}")
    print(f"Calibration vector: {p_cf}")


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_shots', type=int, default=4, help='Number of shots for few-shot learning')
    parser.add_argument('--q_prefix', type=str, default="Review: ", help='Prefix of the review')
    parser.add_argument('--a_prefix', type=str, default="Sentiment: ", help='Prefix of the sentiment')
    parser.add_argument('--train_size', type=int, default=1000, help='Number of training samples to select from')
    parser.add_argument('--test_size', type=int, default=100, help='Number of testing samples')
    parser.add_argument('--content_free_input', type=str, default="N/A", help="Content-free input")
    args = parser.parse_args()
    args = vars(args)

    # Run classification on SST-2 with OpenAI GPT-3 API
    main(**args)