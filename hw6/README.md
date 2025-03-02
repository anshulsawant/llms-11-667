# 11967 Homework 6: Tool Use

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual
  machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 
You could use the same instance for all the assignments. We will specify in the homework instruction and README if you need a different machine.

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-11967-hw6 python=3.11
conda activate cmu-11967-hw6
pip install -r requirements.txt
pip install -e .
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
