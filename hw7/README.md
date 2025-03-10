# 11967 Homework 7: Comparing Models and Mitigating Bias

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual
  machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 
You could use the same instance for all the assignments. We will specify in the homework instruction and README if you need a different machine.

*Note: please make sure your machine have enough space for HW7 since we will load and inference with a 7B model.*

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-11967-hw7 python=3.11
conda activate cmu-11967-hw7
pip install -r requirements.txt
pip install -e .
```

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

## Classification bias calibration
In this problem, you will use the GPT-3 model (davinci-002) with OpenAI API. This is done by using [LiteLLM](https://www.litellm.ai/) API proxy. Please first request an API key from [OpenAI](https://platform.openai.com/api-keys), create a `.env` file, and set the `OPENAI_API_KEY`. After completing `utils.py`, you may run the algorithm in `classification.py`, and test your implementation with `pytest tests/test_calibration.py`.

*Note: please never push your api key to a public space.*

*Hint: to complete this homework, it's really helpful if you understand the API [usage](https://platform.openai.com/docs/api-reference/completions), including the request body and the returned object, and what roles they play in our calibration method.*

## Testing
You can test your solutions by running `pytest` in the project directory. Initially all test cases will fail, and you should check your implementation against the test cases as you are working through the assignment.

## Code submission
1. Run `zip_submission.sh`
2. Upload the generated `submission.zip` to Gradescope


## Acknowledgement
This code contains adaptations from [few-shot-learning](https://github.com/tonyzhaozh/few-shot-learning) ([license](copyright/few-shot-learning)).