#!/bin/bash

ROOT_DIR="datasets/blood-cells"
SEED=42
TARGET_COUNT_TRAIN=100
TARGET_COUNT_VAL=50
TARGET_COUNT_TEST=700

prune_images() {
    local dir="$1"
    local target_count="$2"

    file_count=$(find "$dir" -type f | wc -l)

    if [ "$file_count" -gt "$target_count" ]; then
        delete_count=$((file_count - target_count))

        find "$dir" -type f | head -n "$delete_count" | xargs rm -f
        echo "Pruning $delete_count images in $dir"
    fi
}

for split in train val test; do
    case "$split" in
        train) target_count="$TARGET_COUNT_TRAIN" ;;
        val) target_count="$TARGET_COUNT_VAL" ;;
        test) target_count="$TARGET_COUNT_TEST" ;;
    esac

    for class_dir in "$ROOT_DIR/$split"/*; do
        if [ -d "$class_dir" ]; then
            prune_images "$class_dir" "$target_count"
        fi
    done

done

echo "Pruning finished"