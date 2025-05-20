# 11967 Homework 9: Building a RAG System

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 
You could use the same instance for all the assignments. We will specify in the homework instruction and README if you need a different machine.

### Python Environment
1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw9 python=3.11
    conda activate cmu-11967-hw9
    pip install -r requirements.txt
    # Note: faiss-gpu might require a specific CUDA version. Adjust if necessary.
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0
    pip install ninja
    pip install flash-attn --no-build-isolation
    pip install wandb
    pip install -e .
    ```
3.  **Login to Weights & Biases:**
    *(You will need a [Weights & Biases account](https://wandb.ai/login).)*
    ```bash
    wandb login
    ```

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

## Testing

You can test your solutions by running `pytest tests/` in the project directory as you did in HW1.
Initially all test cases will fail, and you should check your implementation
against the test cases as you are working through the assignment.

## Code submission

1. Run `bash zip_submission.sh`. It fails if mandatory files are missing.
2. A `submission.zip` file should be created. Upload this file to Gradescope.

## Note
This homework builds upon the retriever model trained in HW8. Ensure you have access to your HW8 model.
