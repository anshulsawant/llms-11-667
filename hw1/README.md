# 11967 Homework 1: Implementing a Transformer

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?usp=sharing)

### Python Environment
1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw1 python=3.11
    conda activate cmu-11967-hw1
    pip install -r requirements.txt
    pip install -e .
    ```

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

## Testing

You can test your model implementation by running `pytest tests/test_model.py` in the project directory.
Initially all test cases will fail, and you should check your implementation
against the test cases as you are working through the assignment.

## Code submission

1. Run `scripts/zip-submission.sh`. It fails if mandatory files are missing.
2. A `submission.zip` file should be created. Upload this file to Gradescope.

## Acknowledgement

This code contains adaptations from [nanoGPT](https://github.com/karpathy/nanoGPT)
([license](copyright/nanoGPT)) and [PyTorch](https://pytorch.org/)
([license](copyright/pytorch)).
