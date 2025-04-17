# Import the necessary function from the datasets library
from datasets import load_dataset

# Define the dataset name and the specific configuration ('main' or 'socratic')
dataset_name = "gsm8k"
config_name = "main"  # Or use "socratic" for the socratic version

try:
    # Load the dataset (specify the configuration using the 'name' parameter)
    # This will download the dataset if it's not already cached
    print(f"Loading dataset '{dataset_name}' with configuration '{config_name}'...")
    gsm8k_dataset = load_dataset(dataset_name, name=config_name)
    print("Dataset loaded successfully.")

    # The dataset is usually loaded as a DatasetDict containing splits like 'train' and 'test'
    print("\nAvailable splits:", gsm8k_dataset.keys())

    # --- Accessing the Training Split ---
    if "train" in gsm8k_dataset:
        train_split = gsm8k_dataset["train"]
        print(f"\nNumber of examples in the train split: {len(train_split)}")

        # Access the first example in the training split
        if len(train_split) > 0:
            print("\n--- First Training Example ---")
            first_train_example = train_split[0]
            print(f"Question: {first_train_example['question']}")
            print(f"Answer: {first_train_example['answer']}")
        else:
            print("Train split is empty.")

    # --- Accessing the Test Split ---
    if "test" in gsm8k_dataset:
        test_split = gsm8k_dataset["test"]
        print(f"\nNumber of examples in the test split: {len(test_split)}")

        # Access the first example in the test split
        if len(test_split) > 0:
            print("\n--- First Test Example ---")
            first_test_example = test_split[0]
            print(f"Question: {first_test_example['question']}")
            print(f"Answer: {first_test_example['answer']}")
        else:
            print("Test split is empty.")

    # --- Iterating through a split ---
    if "train" in gsm8k_dataset and len(gsm8k_dataset["train"]) > 0:
        print("\n--- Iterating through the first 3 training examples ---")
        for i, example in enumerate(gsm8k_dataset["train"]):
            if i >= 3:  # Limit to the first 3 for demonstration
                break
            print(f"\nExample {i+1}:")
            print(f"  Question: {example['question']}")
            print(f"  Answer: {example['answer']}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure you have the 'datasets' library installed (`pip install datasets`)")
    print("And that you have a working internet connection to download the dataset.")

