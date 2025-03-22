#!/bin/bash

rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x src/pytest_utils/__pycache__/**\* src/retriever/__pycache__/**\* src/retriever/logs/**\* src/retriever/wandb/**\* src/cmu_11967_hw6.egg-info/**\* src/retriever/driver/__pycache__/**\* src/retriever/modeling/__pycache__/**\*
