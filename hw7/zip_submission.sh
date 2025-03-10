#!/bin/bash

rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x src/bias/__pycache__/**\* src/olmo/__pycache__/**\* src/pytest_utils/__pycache__/**\* src/cmu_11967_hw7.egg-info/**\*