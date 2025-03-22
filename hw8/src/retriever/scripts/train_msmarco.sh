#!/bin/bash


model_to_train=jmvcoelho/pythia-160m-1024-marco-docs-bow-contrastive-pretrain
trained_model_save_path=./data/model
mkdir -p $trained_model_save_path

trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
training_data=Tevatron/msmarco-passage-aug

python -m driver.train \
  --output_dir $trained_model_save_path/$trained_model_name \
  --model_name_or_path $model_to_train \
  --dataset_name $training_data \
  --save_steps 0 \
  --bf16 \
  --gradient_checkpointing \
  --temperature 0.01 \
  --per_device_train_batch_size 128 \
  --train_group_size 10 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name