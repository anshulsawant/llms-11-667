# 11967 Homework 5: Data Processing

## Setting up the Environment

1.  **Install Conda:**
    ```bash
    bash setup-conda.sh && source ~/.bashrc
    ```
2.  **Create and Activate Conda Environment:**
    *(Note: If you encounter an `UnavailableInvalidChannel` error during environment creation, run `conda config --remove channels <offending_channel>` and ensure `conda config --add channels defaults` is set.)*
    ```bash
    conda create -n cmu-11967-hw5 python=3.11
    conda activate cmu-11967-hw5
    pip install -r requirements.txt
    ```

## Data Setup
To download the data you'll be working on, run the following command:
```bash
bash download_data.sh
```

## Instructions
1. You will be modifying only `homework.py` and `mini_ccc.py`.
2. Then follow the assignment pdf to complete this homework.
3. We have provided `test_clean.py` and `test_dataset.py` to help you test your implementation. You can run the tests using the following command:
   ```bash
   pytest test_clean.py
   pytest test_dataset.py
   ```
4. To submit your homework, upload the following files to Gradescope (do not change the filenames):
   - `homework.py`
   - `mini_ccc.py`