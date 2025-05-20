# HW11 - Efficient Inference

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training. Note that for this homework, different types of instances might be needed; follow the recommendations on the handout.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 

### Python Environment
1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw11 python=3.11
    conda activate cmu-11967-hw11
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
    pip install -r requirements.txt
    pip install ninja
    pip install flash-attn==2.6.3 --no-build-isolation
    pip install wandb
    pip install -U "huggingface_hub[cli]"
    ```
3.  **Login to Services:**
    *(You will need accounts for Weights & Biases and Hugging Face.)*
    ```bash
    wandb login
    huggingface-cli login
    ```

### Model Setup
Download the initial model checkpoint:
```bash
chmod +x get_initial_model.sh  # Ensure it's executable
./get_initial_model.sh
```

## Contents
This repo contains a simple huggingface-based pre-training script, supporting two `wikitext` datasets. Each split contains 50M tokens. Both splits are pre-tokenized for your convinience, one set with sequences of 512 tokens, and the other 2048.


## Pre-training

The folder ```scripts``` contains access points to the pre-training code. All scripts under there can be called as follows:

```./scripts/launch_<name>.sh <path_to_config>```, where ```<path_to_config>``` points to a model configuration under ```configs```.

