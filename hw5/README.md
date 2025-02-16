# 11967 Homework 5: Data Processing

## Setting up the environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-11967-hw5 python=3.11
conda activate cmu-11967-hw5
pip install -r requirements.txt
```

## Instructions
1. You will be modifying only `homework.py` and `mini_ccc.py`. To download the data that you'll be working on the following command:
   ```bash
   bash download_data.sh
   ```
2. Then follow the assignment pdf to complete this homework.
3. We have provided `test_clean.py` and `test_dataset.py` to help you test your implementation. You can run the tests using the following command:
   ```bash
   pytest test_clean.py
   pytest test_dataset.py
   ```
4. To submit your homework, upload the following files to Gradescope (do not change the filenames):
   - `homework.py`
   - `mini_ccc.py`