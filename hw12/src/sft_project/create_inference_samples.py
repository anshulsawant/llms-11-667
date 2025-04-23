import json
import argparse
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_gsm8k_test_sample(num_samples: int, output_file: str):
    """
    Loads the gsm8k test split, selects samples, extracts id and question,
    and saves them to a JSON file.

    Args:
        num_samples: The number of samples to extract from the test split.
        output_file: The path to save the output JSON file.
    """
    try:
        logger.info(f"Loading gsm8k dataset (main config, test split)...")
        # Load the test split directly
        dataset = load_dataset("gsm8k", "main", split="test")
        logger.info(f"Dataset loaded. Total test samples: {len(dataset)}")

        # Determine the number of samples to process
        actual_num_samples = min(num_samples, len(dataset))
        if num_samples > len(dataset):
            logger.warning(
                f"Requested {num_samples} samples, but test split only has {len(dataset)}. Using {actual_num_samples} samples."
            )
        else:
            logger.info(f"Selecting {actual_num_samples} samples.")

        # Select the samples
        sampled_dataset = dataset.shuffle().select(range(actual_num_samples))

        # Prepare data for JSON output
        output_data = []
        logger.info("Extracting id and question fields...")
        for i, example in enumerate(sampled_dataset):
            try:
                # Create a simple ID based on the index
                sample_id = f"gsm8k_test_{i:04d}"
                question = example.get("question", None)

                if question is None:
                    logger.warning(
                        f"Sample {i} missing 'question' field. Skipping.")
                    continue

                output_data.append({"id": sample_id, "question": question})
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")

        # Write data to JSON file
        logger.info(f"Writing {len(output_data)} samples to {output_file}...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved sample file to {output_file}")
        except IOError as e:
            logger.error(f"Failed to write to output file {output_file}: {e}")
        except TypeError as e:
            logger.error(f"Data serialization error: {e}. Check data format.")

    except Exception as e:
        logger.error(
            f"An error occurred during dataset loading or processing: {e}",
            exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sample JSON file from the gsm8k test split.")
    parser.add_argument("-n",
                        "--num_samples",
                        type=int,
                        default=10,
                        help="Number of samples to extract from the test set.")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default="gsm8k_test_samples.json",
                        help="Path to the output JSON file.")

    args = parser.parse_args()

    create_gsm8k_test_sample(args.num_samples, args.output_file)
