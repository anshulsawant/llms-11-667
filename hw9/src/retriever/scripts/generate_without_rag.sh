#!/bin/bash

python -m driver.rag \
  --prefixes ./data/prefixes.jsonl \
  --output_dir ./data/