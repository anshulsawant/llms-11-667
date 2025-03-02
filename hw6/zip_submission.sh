#!/bin/bash

rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x src/calculator/__pycache__/**\* src/pytest_utils/__pycache__/**\* src/cmu_11967_hw6.egg-info/**\*
