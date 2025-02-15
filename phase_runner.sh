#!/bin/bash

NUM_GPUS=8

TRAINING_SPLITS=(1 3 5 10 15 20 40 50 60 80)
VALIDATION_SPLITS=(1 3 5 10 15 20 20 20 20 10)

for i in "${!TRAINING_SPLITS[@]}"; do
  echo "----------------- STARTING RUN $((i+1)) -----------------"
  rm -rf datasets results tuner_results

  ./startup.sh --default --num_gpus $NUM_GPUS --training_split "${TRAINING_SPLITS[$i]}" --validation_split "${VALIDATION_SPLITS[$i]}"

  if [ $? -ne 0 ]; then
    echo "Error for splits: ${TRAINING_SPLITS[$i]} i ${VALIDATION_SPLITS[$i]}"
    exit 1
  fi

  mv final_results/results.zip final_results/results_run$((i+1)).zip
done