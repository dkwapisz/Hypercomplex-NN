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

> **Note:** Test split is calculated as `100 - training_split - validation_split`.\


### Model Training and Evaluation Schema
- **description**: `string`  
  - A description of the model and training progress.
- **model_index**: `integer`  
  - The index of the model in the current process.
- **total_model_num_in_process**: `integer`  
  - Total number of models in the current process.
- **process_index**: `integer`  
  - The index of the current process.
- **model_name**: `object`  
  - **type**: `string` (e.g., "CNN")
  - **color_space**: `string` (e.g., "RGB")
  - **algebra**: `null` or `string` (e.g., "Quaternions", "Cl11", etc.)
- **epochs**: `integer`  
  - Number of epochs used for training.
- **model_hyperparameters**: `object`  
  - **optimizer**: `object`  
    - **name**: `string` (e.g., "adam")
    - **learning_rate**: `float`
    - **weight_decay**: `null` or `float`
    - **clipnorm**: `null` or `float`
    - **global_clipnorm**: `null` or `float`
    - **clipvalue**: `null` or `float`
    - **use_ema**: `boolean`
    - **ema_momentum**: `float`
    - **loss_scale_factor**: `null` or `float`
    - **gradient_accumulation_steps**: `null` or `integer`
    - **beta_1**: `float`
    - **beta_2**: `float`
    - **epsilon**: `float`
    - **amsgrad**: `boolean`
  - **loss**: `string` (e.g., "categorical_crossentropy")
  - **metrics**: `array of strings` (e.g., `["loss", "accuracy", "precision"]`)
  - **learning_rate**: `float`
  
- **model_layers_details**: `array of objects`  
  Each object represents a layer in the model:
  - **layer_name**: `string` (e.g., "layer1", "layer2")
  - **type**: `string` (e.g., "Conv2D", "Dense", etc.)
  - **input**: `array of integers` (e.g., `[null, 128, 128, 3]`)
  - **output**: `array of integers` (e.g., `[null, 126, 126, 32]`)
  - **params**: `integer`  
  - **stride**: `array of integers` (e.g., `[1, 1]`)
  - **padding**: `string` (e.g., "valid", "same")
  - **filters**: `integer` (only for Conv2D layers)
  - **kernel_size**: `array of integers` (e.g., `[3, 3]`)

- **training_history**: `object`  
  - **AUC**: `array of floats`
  - **FalseNegatives**: `array of integers`
  - **FalsePositives**: `array of integers`
  - **Precision**: `array of floats`
  - **Recall**: `array of floats`
  - **TrueNegatives**: `array of integers`
  - **TruePositives**: `array of integers`
  - **accuracy**: `array of floats`
  - **categorical_accuracy**: `array of floats`
  - **loss**: `array of floats`
  - **val_AUC**: `array of floats`
  - **val_FalseNegatives**: `array of integers`
  - **val_FalsePositives**: `array of integers`
  - **val_Precision**: `array of floats`
  - **val_Recall**: `array of floats`
  - **val_TrueNegatives**: `array of integers`
  - **val_TruePositives**: `array of integers`
  - **val_accuracy**: `array of floats`
  - **val_categorical_accuracy**: `array of floats`
  - **val_loss**: `array of floats`

- **evaluation_result**: `object`  
  - **loss**: `float`
  - **accuracy**: `float`
  - **categorical_accuracy**: `float`
  - **AUC**: `float`
  - **Precision**: `float`
  - **Recall**: `float`
  - **TruePositives**: `integer`
  - **TrueNegatives**: `integer`
  - **FalsePositives**: `integer`
  - **FalseNegatives**: `integer`

- **training_time_seconds**: `float`  
  - Time taken for training in seconds.

- **evaluate_time_seconds**: `float`  
  - Time taken for evaluation in seconds.
