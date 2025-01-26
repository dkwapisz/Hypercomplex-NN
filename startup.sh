#!/bin/bash

# Setup environment variables
if [ -f .env ]; then
  export $(cat .env | xargs -0)
fi

pip install tensorflow[and-cuda]
pip install -r requirements.txt

if [ ! -d datasets ]; then
  mkdir -p datasets
fi

if [ ! -f datasets/multi-cancer.zip ]; then
  curl -L -o datasets/multi-cancer.zip\
  https://www.kaggle.com/api/v1/datasets/download/obulisainaren/multi-cancer
else
  echo "Dataset already downloaded, skipping..."
fi

python3 Main.py