#!/bin/bash

rm -rf submission submission.zip
mkdir submission

if [[ ! -d src/ ]] ; then
    echo 'Nothing found at src/, aborting'
    exit 1
fi

if [[ ! -f requirements.txt ]] ; then
    echo 'requirements.txt missing, aborting'
    exit 1
fi

if [[ ! -f setup.py ]] ; then
    echo 'setup.py missing, aborting'
    exit 1
fi

# if [ -f $MODEL_DIR/model.pt ]
# then
#     cp $MODEL_DIR/model.pt submission/
# else
#     echo "model file not found"
# fi

# if [ -f $MODEL_DIR/config.yaml ]
# then
#     cp $MODEL_DIR/config.yaml submission/
# else
#     echo "model configuration not found"
# fi

cp -r  src requirements.txt setup.py submission
cd submission
zip -qr ../submission.zip .

echo "done!"
