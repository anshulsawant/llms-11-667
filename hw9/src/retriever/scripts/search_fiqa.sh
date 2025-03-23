#!/bin/bash

trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
EMBEDDING_OUTPUT_DIR=./data/embeddings/$trained_model_name/fiqa

python -m driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-test.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.test.txt \
    --save_metrics_to $EMBEDDING_OUTPUT_DIR/trec_eval_results.txt \
    --qrels ./data/qrel.fiqa.test.tsv

cat $EMBEDDING_OUTPUT_DIR/trec_eval_results.txt