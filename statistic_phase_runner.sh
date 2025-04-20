#!/bin/bash

for i in {1..5}; do
  echo "----------------- STARTING ITERATION $i -----------------"

  ./phase_runner.sh

  if [ $? -ne 0 ]; then
    echo "Error occurred during iteration $i. Exiting..."
    exit 1
  fi


  mv final_results final_results_$((i))
  rm -rf final_results/*
  echo "----------------- FINISHED ITERATION $i -----------------"
done

echo "All iterations completed successfully."