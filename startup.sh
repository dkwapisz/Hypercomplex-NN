#!/bin/bash

# Static variables
DATASET_UNPACKED_NAME="cifar-100-python"
DATASET_TAR_NAME="cifar-100-python.tar.gz"

DATASET_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
SEED=123

# Default dynamic values
num_gpus=1
download_data=false
prepare_data=false
test_run=false
run_training=false

if [ -f .env ]; then
  export "$(xargs -0 < .env)"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: ./startup.sh [options]"
  echo "Options:"
  echo "  --default: Equivalent to --prepare_data --download_data --run_training combined."
  echo "  --download_data: Downloads the dataset file."
  echo "  --prepare_data: Unzips the dataset file."
  echo "  --run_training: Starts the tuning & training process."
  echo "  --num_gpus <number>: Number of GPUs to use in parallel."
  echo "  Test split is calculated as 100 - TRAINING_SPLIT - VALIDATION_SPLIT."
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download_data)
      download_data=true
      ;;
    --prepare_data)
      prepare_data=true
      ;;
    --test_run)
      test_run=true
      ;;
    --run_training)
      run_training=true
      ;;
    --default)
      prepare_data=true
      download_data=true
      run_training=true
      ;;
    --num_gpus)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        num_gpus="$2"
        shift
      else
        echo "Error: --num_gpus flag requires a numerical argument."
        exit 1
      fi
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
  shift
done

sudo apt-get install -y zip htop nano
pip install -r requirements.txt


if [ "$download_data" = true ] && [ ! -d "$DATASET_TARGET_DIR" ]; then
  wget $DATASET_URL
  tar -xvzf $DATASET_TAR_NAME
  rm $DATASET_TAR_NAME

else
  echo "Dataset already downloaded or download skipped."
fi


if [ "$prepare_data" = true ] && [ ! -d "$DATASET_TARGET_DIR" ]; then
  echo "Prepare data requested"
  mkdir datasets
  python3 Cifar100Converter.py
  rm -rf $DATASET_UNPACKED_NAME
else
  echo "Dataset folder already exists or preparation skipped."
fi

if [ "$test_run" = true ]; then
  python3 Testing.py
fi

if [ "$run_training" = true ]; then
  rm -rf "results" "tuner_results" "final_results"
  mkdir "results" "tuner_results" "final_results"
  i=0
  while [ $i -lt "$num_gpus" ]; do
    python3 Main.py $i "$num_gpus" &
    pid=$!
    echo "Running process $i has PID: $pid"
    i=$((i + 1))
  done

  wait
  echo "All processes completed training. Packing results..."

  find tuner_results/ -type f -name 'checkpoint.weights.h5' -exec rm {} \; # Removing checkpoints to reduce zip size

  zip -r results.zip results
  zip -r tuner_results.zip tuner_results

  mv results.zip tuner_results.zip final_results/

  echo "Results packed. Processing completed. Thank you for your patience."
else
  echo "Training skipped."
fi

