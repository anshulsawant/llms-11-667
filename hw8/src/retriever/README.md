## Dense retriever training

The code in this repo is based on [Tevatron](https://github.com/texttron/tevatron/tree/main).

## Organization

- ```retriever/arguments.py```: Command line arguments to run scripts.
- ```retriever/collator.py```: Dataloader collator, responsible for query and passage tokenization.
- ```retriever/dataset.py```: HuggingFace dataset interface, to read the train and test datasets.
- ```retriever/searcher.py```: Vector search interface using FAISS, supports GPU search.
- ```retriever/trainer.py```: Overrides Huggingface's Trainer interface to support dense retrieval training.

- ```retriever/modeling/encoder.py```: Models a dense retriever. **This is the only file you need to change**
- ```retriever/driver/*.py```: Script entry points: training, encoding, and searching.

## Instructions

### Implementing missing functionality

Follow the statement and the comments in the code to implement the required functions under  ```retriever/modeling/encoder.py```.

- Do not change other files.
- Do not change functions arguments and return values, as it may break automated test cases.

You can check your implementation by running ```pytest tests/test_encoder.py``` in the root folder of the homework.

### Training a dense retriever

Under ```/retriever```, run: ```./scripts/train_msmarco.sh ```. 
- This will train a dense retriever in a small subset of the data from the [MS-MARCO](https://arxiv.org/abs/1611.09268) passage retrieval dataset. 
- Model will be saved under ```./data/model/pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft```.  

### Encoding a collection

The MARCO collection is large-ish, so you will use a different, smaller corpus to check the performance of your model.  

Under ```/retriever```, run: ```./scripts/encode_fiqa.sh ```.
- This will generate embeddings for the queries and passages of the [FiQA](https://sites.google.com/view/fiqa/) collection. 
- Corpus and query embeddings will be saved under ```./data/embeddings```.

### Searching 

Under ```/retriever```, run: ```./scripts/search_fiqa.sh```.
- This will return the top100 documents for each query, and provide evaluation metrics. 
- After running, ```./data/embeddings``` will contain a txt file with evaluation metrics.  


**How to intrepret runs**

```./data/embeddings/{model-name}/fiqa/run.test.txt``` is a TREC-formatted run file. Each line contains

```qid Q0 pid pos score model```

- qid: the query identifier
- Q0: ignore
- pid: the passage identifier
- pos: the ranking position of pid for qid (in this case, each query will have the top 100)
- score: pid's score for qid
- model: model used for the run

```./retriever/data/qrel.fiqa.test.tsv```is a QREL file, containing ground-truth relevance judgements. Each line contains ```qid Q0 pid rel```. E.g:

- ```8 0 566392 1``` - 566392 is a relevant passage for query 8

```./data/embeddings/{model-name}/fiqa/trec_eval_results.txt``` contains multiple retrieval metrics to assess the quality of the model. 
It follows [trec-eval](https://github.com/usnistgov/trec_eval), but was computed using the python interface ([pytrec-eval](https://github.com/cvangysel/pytrec_eval)). For instance, the "Mean Reiprocal Rank" can be found next to "recip_rank".  