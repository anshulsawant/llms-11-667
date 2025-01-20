~/cmu-llms/miniconda3/bin/conda init bash
source ~/.bashrc
conda config --append envs_dirs ~/cmu-llms
conda activate cmu-11967-hw2
pip install -e .
wandb login
