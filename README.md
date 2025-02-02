> Project is in progress. 


### Usage: `./startup.sh [options]`

#### Options:
- `--default`  
  Equivalent to `--prepare_data --download_data --run_training` combined.

- `--download_data`  
  Downloads the dataset file.

- `--prepare_data`  
  Unzips the dataset file and splits it into training, validation and test sets.

- `--test_run`  
  Run tests that checks dataset processing and data compatibility with models.

- `--run_training`  
  Starts the training process.

- `--num_gpus <number>`  
  Number of GPUs to use in parallel

- `--training_split <number>`  
  Percentage of data to use for training.

- `--validation_split <number>`  
  Percentage of data to use for validation.

> **Note:** Test split is calculated as `100 - training_split - validation_split`.