# 11967 Homework 7: Comparing Models and Mitigating Bias

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 
You could use the same instance for all the assignments. We will specify in the homework instruction and README if you need a different machine.

*Note: Please ensure your machine has enough disk space for HW7, as it involves loading and running inference with a 7B model.*

### Python Environment
1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw7 python=3.11
    conda activate cmu-11967-hw7
    pip install -r requirements.txt
    pip install -e .
    ```
    *Note: This homework uses LiteLLM to interact with the OpenAI API. The necessary packages (`litellm`, `openai`) are included in `requirements.txt`.*

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

### API Key Setup (for Classification Bias Calibration)
This homework involves using the GPT-3 model (davinci-002) via the OpenAI API, proxied by LiteLLM.
1.  Request an API key from [OpenAI](https://platform.openai.com/api-keys).
2.  Create a `.env` file in the `hw7` project root directory.
3.  Add your API key to the `.env` file like this:
    ```env
    OPENAI_API_KEY="your_api_key_here"
    ```
*Important: Never commit your `.env` file or API key to a public repository.*

*Hint: To complete this homework, it's really helpful if you understand the OpenAI API [usage](https://platform.openai.com/docs/api-reference/completions), including the request body and the returned object, and what roles they play in our calibration method.*

## Testing
You can test your solutions by running `pytest` in the project directory. Initially all test cases will fail, and you should check your implementation against the test cases as you are working through the assignment.

## Code submission
1. Run `zip_submission.sh`
2. Upload the generated `submission.zip` to Gradescope


## Acknowledgement
This code contains adaptations from [few-shot-learning](https://github.com/tonyzhaozh/few-shot-learning) ([license](copyright/few-shot-learning)).