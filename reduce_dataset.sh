#!/bin/bash

DATASET_MAIN_DIR="datasets"
DATASET_TARGET_NAME_DIR="blood-cells"
DATASET_TARGET_DIR="$DATASET_MAIN_DIR/$DATASET_TARGET_NAME_DIR"

SEED=42

NUM_FILES_TO_KEEP=400

for split in train val test; do
    for class_dir in "$DATASET_TARGET_DIR/$split/"*; do
        if [ -d "$class_dir" ]; then
            files=("$class_dir"/*)
            if [ "${#files[@]}" -gt "$NUM_FILES_TO_KEEP" ]; then
                ls "$class_dir"/* | shuf --random-source=<(echo "$SEED") | tail -n +$(($NUM_FILES_TO_KEEP + 1)) | xargs rm
            fi
        fi
    done
done