#!/bin/bash

# Static variables
DATASET_MAIN_DIR="datasets"
DATASET_MAIN_SUBDIR="PBC_dataset_normal_DIB_224"
DATASET_TARGET_NAME_DIR="blood-cells"
DATASET_TARGET_DIR="$DATASET_MAIN_DIR/$DATASET_TARGET_NAME_DIR"
DATASET_ZIP_NAME="blood-cell.zip"
TRAINING_SPLIT=1
VALIDATION_SPLIT=1
MAX_IMAGES_PER_CLASS=1200

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/bzhbzh35/peripheral-blood-cell"
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
  mkdir -p $DATASET_MAIN_DIR
  curl -L -o $DATASET_MAIN_DIR/$DATASET_ZIP_NAME $DATASET_URL
else
  echo "Dataset already downloaded or download skipped."
fi


if [ "$prepare_data" = true ] && [ ! -d "$DATASET_TARGET_DIR" ]; then
  echo "Prepare data requested"

  unzip $DATASET_MAIN_DIR/$DATASET_ZIP_NAME $DATASET_MAIN_SUBDIR/$DATASET_MAIN_SUBDIR/'*' -d $DATASET_MAIN_DIR/

  # Cleanup
  mv $DATASET_MAIN_DIR/$DATASET_MAIN_SUBDIR/$DATASET_MAIN_SUBDIR $DATASET_MAIN_DIR/$DATASET_TARGET_NAME_DIR
  rm -rf $DATASET_MAIN_DIR/$DATASET_MAIN_SUBDIR
  rm $DATASET_MAIN_DIR/$DATASET_ZIP_NAME

  for class_dir in "$DATASET_MAIN_DIR"/"$DATASET_TARGET_NAME_DIR"/*/; do
      [ -d "$class_dir" ] || continue

      file_count=$(find "$class_dir" -type f | wc -l)

      if [ "$file_count" -gt "$MAX_IMAGES_PER_CLASS" ]; then
          files_to_remove=$((file_count - MAX_IMAGES_PER_CLASS))
          find "$class_dir" -type f | shuf | head -n "$files_to_remove" | xargs rm -f
          echo "Removed $files_to_remove files from $class_dir"
      else
          echo "Cannot remove from $class_dir. It has not enough files"
      fi
  done

  for class in "$DATASET_TARGET_DIR"/*/; do
    class_name=$(basename "$class")

    mkdir -p "$DATASET_TARGET_DIR/train/$class_name"
    mkdir -p "$DATASET_TARGET_DIR/val/$class_name"
    mkdir -p "$DATASET_TARGET_DIR/test/$class_name"

    mapfile -t files < <(find "$class" -type f)
    total_files=${#files[@]}

    mapfile -t shuffled_files < <(printf "%s\n" "${files[@]}" | awk -v seed="$SEED" 'BEGIN { srand(seed) } {print rand(), $0}' | sort -n | cut -d' ' -f2-)

    train_count=$((total_files * TRAINING_SPLIT / 100))
    val_count=$((total_files * VALIDATION_SPLIT / 100))
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

