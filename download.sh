#!/bin/bash

# Define the base URL for the downloads
BASE_URL="https://github.com/yakhyo/head-pose-estimation/releases/download/weights"

# Create the weights directory if it does not exist
mkdir -p weights

# Check if a model name was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "  model_name: resnet18, resnet34, resnet50, mobilenetv2, mobilenetv3_small, mobilenetv3_large"
    echo "Example: $0 resnet18"
    exit 1
fi

MODEL_NAME=$1

for EXT in pt onnx; do
    MODEL_FILE="${MODEL_NAME}.${EXT}"
    echo "Downloading $MODEL_FILE ..."
    wget -q -O weights/$MODEL_FILE $BASE_URL/$MODEL_FILE
    if [ $? -eq 0 ]; then
        echo "Downloaded $MODEL_FILE to weights/"
    else
        echo "Failed to download $MODEL_FILE"
    fi
done