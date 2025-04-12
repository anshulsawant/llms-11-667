# HW10 - Efficient Training and Inference


## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training. Note that for this homework different types of instances are needed. Follow the recommendations on the handout.  
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1zNOkS8GmtJxMQ74g41610RVe-ZYNkGwkZfq18mr78ME/edit?usp=sharing) 

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`

2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-11967-hw10 python=3.11
conda activate cmu-11967-hw10
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
pip install -r requirements.txt
pip install ninja
pip install flash-attn==2.6.3 --no-build-isolation
pip install wandb
pip install -U "huggingface_hub[cli]"
```
3. Run `wandb login` to finish setting up weights & biases for experiment tracking (you will need to have a [weights & biases account](https://wandb.ai/login)).  

4. Run `huggingface-cli login` to allow downloading and uploading to Huggingface hub.  

5. Run `chmod +x get_initial_model.sh` if you run into permission error, then `./get_initial_model.sh` to download the model starting point.  

## Contents
This repo contains a simple huggingface-based pre-training script, supporting two `wikitext` datasets. Each split contains 50M tokens. Both splits are pre-tokenized for your convinience, one set with sequences of 512 tokens, and the other 2048.


## Pre-training

The folder ```scripts``` contains access points to the pre-training code. All scripts under there can be called as follows:

```./scripts/launch_<name>.sh <path_to_config>```, where ```<path_to_config>``` points to a model configuration under ```configs```.

