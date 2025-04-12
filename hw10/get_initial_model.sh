mkdir -p models/GPTNeoX-160m
cd models/GPTNeoX-160m
huggingface-cli download jmvcoelho/GPTNeoX-160m --local-dir .
cd ../..