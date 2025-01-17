# 11967 Homework 2: Training a Transformer

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
conda create -n cmu-11967-hw2 python=3.11
conda activate cmu-11967-hw2
pip install -r requirements.txt
pip install -e .
```
3. Run `wandb login` to finish setting up weights & biases for experiment tracking (you will need to have a [weights & biases account](https://wandb.ai/login)).
4. Download pre-tokenized training data: `curl https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz -o data/tokens.npz -L`

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

## Testing

You can test your solutions by running `pytest tests/` in the project directory as you did in HW1.
Initially all test cases will fail, and you should check your implementation
against the test cases as you are working through the assignment.


## Training

### Configuration

We use the YAML format to configure model training.
You should modify the example below to explore different hyperparameters.

```yaml
output_dir: outputs/GPT-tiny  # <- where the output files are written
tokenizer_encoding: gpt2      # <- the tokenizer encoding, used by tiktoken (YOU SHOULD NOT CHANGE THIS)
model_config:
  n_embd: 32                  # <- dimension of token and positional embeddings 
  n_head: 2                   # <- number of attention heads in multihead attention
  n_positions: 128            # <- the maximum number of tokens that the model can take
  n_layer: 2                  # <- number of decoder blocks
device: auto                  # <- which device to put the model on (YOU DO NOT NEED TO CHANGE THIS)
batch_size: 32                # <- number of sequences to feed into the model at a time
seq_len: 128                  # <- length of each sequence in training and evaluation, <= model_config.n_positions
num_warmup_steps: 10          # <- number of warmup steps in cosine annealing
num_training_steps: 2000      # <- number of training steps in cosine annealing
grad_accumulation_steps: 1    # <- number of micro steps of gradient accumulation before every model update
min_lr: 1e-4                  # <- minimum learning rate in cosine annealing
max_lr: 5e-4                  # <- maximum learning rate in cosine annealing
```

### Train a model

After implementing `src/lm/model.py` and `src/lm/train.py`, you can run
  `python src/lm/train.py configs/GPT-tiny.yaml` to train a tiny (~2M parameters!)
  model.

You will write additional config files for hyperparameters tuning and training
  the final model.

### Efficient training

Although you will not be graded on this, you will need to write an efficient
  implementation of the transformers architecture so you can train language
  models in a reasonable amount of time.
Part of this means that you should never use for loops for matrix operations -
  look for PyTorch functions that do the job.

For reference, our implementation reached around 15.1 TFLOPS (teraflops per
  second) during training, and the full training run (~1e+17 FLOPs) took less
  than 2 hours on an AWS `g5.2xlarge` instance.

*Note: FLOPs per second will be low for small models (e.g., training on
  `configs/GPT-tiny.yaml` reaches 1.8 TFLOPS and runs in about a minute).*

### A note on einops (optional)

By default, we have made the `einops` package available for you.
Using it is completely optional (PyTorch can do everything it does), but it
  makes certain tensor manipulations simple and intuitive.
Below are two examples where you may find `einops.rearrange` useful, and you
  can find a tutorial [here](https://einops.rocks/1-einops-basics/).

```
B, S, V = 16, 256, 50257

x = torch.rand(B, S, V)

# flatten axes B and V
y = einops.rearrange(x, "B S V -> (B S) V")
y = x.reshape(B * S, V)

# transpose axes S and V
z = einops.rearrange(x, "B S V -> B V S")
z = torch.permute(x, (0, 2, 1))
```

## Code submission

1. Set an environment variable `export MODEL_DIR="[path to your trained model]"` (for example, `outputs/GPT-tiny`), so the submission preparation script knows where to look for your models.
It's okay to skip this if you do not intend to submit a model for perplexity grading (Question 1.5).
2. Run `scripts/zip-submission.sh`. It fails if mandatory files are missing.
3. A `submission.zip` file should be created. Upload this file to Gradescope.

## Acknowledgement

This code contains adaptations from [nanoGPT](https://github.com/karpathy/nanoGPT)
([license](copyright/nanoGPT)) and [PyTorch](https://pytorch.org/)
([license](copyright/pytorch)).
