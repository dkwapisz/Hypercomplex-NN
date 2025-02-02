#!/bin/bash

DATASET_MAIN_DIR="datasets"
DATASET_MAIN_SUBDIR="Lymphoma"
DATASET_TARGET_DIR="$DATASET_MAIN_DIR/$DATASET_MAIN_SUBDIR"
DATASET_ZIP_NAME="multi-cancer.zip"

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/obulisainaren/multi-cancer"
SEED=123

num_gpus=1
prepare_data=false
download_data=false
test_run=false
run_training=false
training_split=80
validation_split=10

if [ -f .env ]; then
  export "$(xargs -0 < .env)"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: ./startup.sh [options]"
  echo "Options:"
  echo "  --default: Equivalent to --prepare_data --download_data --run_training combined."
  echo "  --download_data: Downloads the dataset file."
  echo "  --prepare_data: Unzips the dataset file."
  echo "  --test_run: Run tests that checks dataset and data compatibility with models."
  echo "  --run_training: Starts the training process."
  echo "  --num_gpus <number>: Number of GPUs to use in parallel."
  echo "  --training_split <number>: Percentage of data to use for training."
  echo "  --validation_split <number>: Percentage of data to use for validation."
  echo "  Test split is calculated as 100 - training_split - validation_split."
  exit 0
fi

for arg in "$@"; do
  case $arg in
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
        num_gpus=$2
        shift
      else
        echo "Error: --num_processes flag requires a numerical argument."
        exit 1
      fi
      ;;
    --training_split)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        training_split=$2
        shift
      else
        echo "Error: --training_split flag requires a numerical argument."
        exit 1
      fi
      ;;
    --validation_split)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        validation_split=$2
        shift
      else
        echo "Error: --validation_split flag requires a numerical argument."
        exit 1
      fi
      ;;
    *)
      echo "Invalid option: $arg"
      exit 1
      ;;
  esac
done

pip install -r requirements.txt --quiet


if [ "$download_data" = true ] && [ ! -f $DATASET_MAIN_DIR/$DATASET_ZIP_NAME ]; then
  mkdir -p $DATASET_MAIN_DIR
  curl -L -o $DATASET_MAIN_DIR/$DATASET_ZIP_NAME $DATASET_URL
else
  echo "Dataset already downloaded or download skipped."
fi


if [ "$prepare_data" = true ] && [ ! -d "$DATASET_TARGET_DIR" ]; then
  echo "Prepare data requested"

  unzip $DATASET_MAIN_DIR/$DATASET_ZIP_NAME 'Multi Cancer/Multi Cancer/Lymphoma/*' -d $DATASET_MAIN_DIR/

  # Cleanup
  mv $DATASET_MAIN_DIR/Multi\ Cancer/Multi\ Cancer/$DATASET_MAIN_SUBDIR $DATASET_MAIN_DIR/
  rm -rf $DATASET_MAIN_DIR/Multi\ Cancer
  rm $DATASET_MAIN_DIR/$DATASET_ZIP_NAME

  for class in "$DATASET_TARGET_DIR"/*/; do
    class_name=$(basename "$class")

    mkdir -p "$DATASET_TARGET_DIR/train/$class_name"
    mkdir -p "$DATASET_TARGET_DIR/val/$class_name"
    mkdir -p "$DATASET_TARGET_DIR/test/$class_name"

    mapfile -t files < <(find "$class" -type f)
    total_files=${#files[@]}

    mapfile -t shuffled_files < <(printf "%s\n" "${files[@]}" | awk -v seed="$SEED" 'BEGIN { srand(seed) } {print rand(), $0}' | sort -n | cut -d' ' -f2-)

    train_count=$((total_files * training_split / 100))
    val_count=$((total_files * validation_split / 100))
    test_count=$((total_files - train_count - val_count))

    mv "${shuffled_files[@]:0:train_count}" "$DATASET_TARGET_DIR/train/$class_name/"
    mv "${shuffled_files[@]:train_count:val_count}" "$DATASET_TARGET_DIR/val/$class_name/"
    mv "${shuffled_files[@]:train_count+val_count:test_count}" "$DATASET_TARGET_DIR/test/$class_name/"

    rmdir "$class" 2>/dev/null
  done
else
  echo "Dataset folder already exists or preparation skipped."
fi

if [ "$test_run" = true ]; then
  python3 Testing.py
fi

if [ "$run_training" = true ]; then
  i=0
  while [ $i -lt "$num_gpus" ]; do
    python3 Main.py $i "$num_gpus" &
    pid=$!
    echo "Running process $i has PID: $pid"
    i=$((i + 1))
  done

  wait
  echo "All processes completed."
else
  echo "Training skipped."
fi

