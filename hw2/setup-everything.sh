mkdir -p ~/cmu-llms/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/cmu-llms/miniconda3/miniconda.sh
bash ~/cmu-llms/miniconda3/miniconda.sh -b -u -p ~/cmu-llms/miniconda3
rm -rf ~/cmu-llms/miniconda3/miniconda.sh
~/cmu-llms/miniconda3/bin/conda init bash
source ~/.bashrc
conda create --prefix=~/cmu-llms/cmu-11967-hw2 python=3.11
conda config --append envs_dirs ~/cmu-llms
conda activate cmu-11967-hw2
pip install -r requirements.txt
pip install -e .
wandb login
mkdir data
curl https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz -o data/tokens.npz -L
