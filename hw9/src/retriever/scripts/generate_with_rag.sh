#!/bin/bash

### ENCODE THE CORPUS AND SEARCH
trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft

EMBEDDING_OUTPUT_DIR=./data/embeddings/$trained_model_name/rag

mkdir -p $EMBEDDING_OUTPUT_DIR

python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path ./data/model/$trained_model_name \
  --bf16 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_name jmvcoelho/toy-queries \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/query-test.pkl

python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path ./data/model/$trained_model_name \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_name jmvcoelho/toy-corpus \
  --dataset_number_of_shards 1 \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.pkl


python -m driver.search \
  --query_reps $EMBEDDING_OUTPUT_DIR/query-test.pkl \
  --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.pkl \
  --depth 10 \
  --batch_size 64 \
  --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.test.txt


### GENERATE WITH RAG

path_to_run=$EMBEDDING_OUTPUT_DIR/run.test.txt

python -m driver.rag \
  --prefixes ./data/prefixes.jsonl \
  --output_dir ./data/ \
  --augmentation_run $path_to_run