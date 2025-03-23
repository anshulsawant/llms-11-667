#!/bin/bash

test_data=fiqa
trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
EMBEDDING_OUTPUT_DIR=./data/embeddings/$trained_model_name/$test_data
trained_model_path=./data/model/$trained_model_name

mkdir -p $EMBEDDING_OUTPUT_DIR

python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path $trained_model_path \
  --bf16 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_name Tevatron/beir \
  --dataset_config $test_data \
  --dataset_split test \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/query-test.pkl

python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path $trained_model_path \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config $test_data \
  --dataset_number_of_shards 1 \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.pkl