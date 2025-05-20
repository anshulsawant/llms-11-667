# 11967 Homework 3: Text Generation & Perplexity Analysis

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 

### Python Environment
1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw3 python=3.11
    conda activate cmu-11967-hw3
    pip install -r requirements.txt
    pip install -e .
    ```

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

### Data Setup
Download and unzip the data:
```bash
curl https://huggingface.co/datasets/yimingzhang/llms-hw3/resolve/main/data.zip -o data/data.zip -L
unzip data/data.zip -d data/
# Note: You might need to install unzip first, e.g., sudo apt-get install unzip
```

## Testing

You can test your solutions by running `pytest tests/` in the project directory as you did in HW1.
Initially all test cases will fail, and you should check your implementation
against the test cases as you are working through the assignment.


## Training
In HW3, you should use your final model from HW2 to do generation. If you need to train again, please refer to HW2 README.

## Code submission

1. Run `scripts/zip-submission.sh`. It fails if mandatory files are missing.
3. A `submission.zip` file should be created. Upload this file to Gradescope.

## Acknowledgement

This code contains adaptations from [nanoGPT](https://github.com/karpathy/nanoGPT)
([license](copyright/nanoGPT)) and [PyTorch](https://pytorch.org/)
([license](copyright/pytorch)).
