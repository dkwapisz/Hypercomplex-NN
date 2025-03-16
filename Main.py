import json
import os
import sys
import time

from keras.src.callbacks import EarlyStopping
from keras.src.utils import set_random_seed

from config.PropertiesResolver import PropertiesResolver
from data_processing.ColorSpaceConverter import create_dataset_tf
from models.CNN_Model import CNN_Model, objective_cnn, perform_model_tuning_cnn
from models.HyperComplexCNN_Model import HyperComplexCNN_Model, objective_hcnn, perform_model_tuning_hcnn
from models.ModelUtils import get_models_range, get_current_model_desc
from utils.GPU_Helper import check_gpu_health, set_gpu_device
from utils.LogsHelper import create_model_result_subdir, get_log_data, model_summary_to_dict, get_model_hyperparams

process_start_time = time.time()

# ------------------- Static parameters -------------------
dataset_path = "datasets/blood-cells"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")
img_size = (100, 100)
batch_size = 64
eval_batch_size = 32
epochs = 500
verbose = 0
num_classes = len(os.listdir(train_path))
metrics = ["accuracy", "categorical_accuracy", "AUC", "Precision", "Recall", "TruePositives", "TrueNegatives",
           "FalsePositives", "FalseNegatives"]
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)

# ------------------- Env setup -------------------
set_random_seed(123)

if len(sys.argv) != 3:
    print("Attributes are wrong or not provided. Default run, taking 1 GPU and 1 process.")
    gpu_index = 0
    num_processes = 1
else:
    gpu_index = int(sys.argv[1])
    num_processes = int(sys.argv[2])

check_gpu_health()
set_gpu_device(gpu_index)

# ------------------- Training mode setup -------------------
properties = PropertiesResolver("properties.json")

training_mode = properties.get("training_mode")

models_to_train = []

if training_mode["all_cnn"]:
    models_to_train += properties.get("cnn_models_preset", [])
if training_mode["all_hypercomplex_cnn"]:
    models_to_train += properties.get("hypercomplex_cnn_models_preset", [])
if training_mode["phase15+"]:
    models_to_train += properties.get("phase15+_preset", [])
if training_mode["custom"]:
    models_to_train += properties.get("custom_preset", [])

total_model_num = len(models_to_train)

models_range_to_run = get_models_range(total_model_num, num_processes, gpu_index)
tune_model = properties.get("tune_model")
if tune_model:
    log_filename = "tuning_log.json"
else:
    log_filename = "training.json"

# ------------------- Training -------------------
for model_index in range(models_range_to_run[0], models_range_to_run[1]):
    results_subdir_path = create_model_result_subdir(models_to_train[model_index])
    log_file_path = os.path.join(results_subdir_path, log_filename)
    log_data = get_log_data(model_index, total_model_num, gpu_index, models_to_train[model_index], epochs)

    model_training_start_time = time.time()
    model_name = models_to_train[model_index]

    hypercomplex = model_name["type"] == "HyperComplex"
    color_space = model_name["color_space"]

    print(get_current_model_desc(model_index, total_model_num, model_name))

    train_dataset = create_dataset_tf(train_path, img_size, batch_size, color_space, hypercomplex)
    val_dataset = create_dataset_tf(val_path, img_size, batch_size, color_space, hypercomplex)
    test_dataset = create_dataset_tf(test_path, img_size, batch_size, color_space, hypercomplex)

    num_channels = 0
    for image, _ in train_dataset.take(1):
        num_channels = image.shape[-1]
        break

    assert num_channels != 0, "Number of channels could not be determined."

    input_shape = img_size + (num_channels,) if (color_space == "CMYK" or hypercomplex) else img_size + (num_channels,)

    if tune_model:
        start_time = time.time()
        if hypercomplex:
            algebra_name = model_name["algebra"]
            best_params = perform_model_tuning_hcnn(train_dataset, val_dataset, input_shape, num_classes, algebra_name)
        else:
            best_params = perform_model_tuning_cnn(train_dataset, val_dataset, input_shape, num_classes)
        log_data["tuning_time"] = round(time.time() - start_time, 4)
        log_data["best_params"] = best_params
        with open(log_file_path, "w") as log_file:
            json.dump(log_data, log_file, indent=4)
        continue
    else:
        if hypercomplex:
            algebra_name = model_name["algebra"]
            model = HyperComplexCNN_Model(input_shape, num_classes, color_space, metrics, algebra_name)
        else:
            model = CNN_Model(input_shape, num_classes, color_space, metrics)

    log_data["model_hyperparameters"] = get_model_hyperparams(model.get_model())
    log_data["model_layers_details"] = model_summary_to_dict(model.get_model())

    # ------------------- Training -------------------
    print(f"Starting training for {model_name}")
    history = model.get_model().fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=verbose,
                                    callbacks=[early_stopping])
    print(f"Training complete for {model_name}")

    model_training_end_time = time.time()

    # ------------------- Evaluation -------------------
    print(f"Evaluating model {model_name}")
    eval_result = model.get_model().evaluate(test_dataset, batch_size=eval_batch_size, verbose=verbose)
    print(f"Evaluation complete for {model_name}")

    model_evaluate_end_time = time.time()

    log_data["training_history"] = history.history
    log_data["evaluation_result"] = {"loss": eval_result[0], **{metric: value for metric, value in zip(metrics, eval_result[1:])}}
    log_data["training_time_seconds"] = round(model_training_end_time - model_training_start_time, 4)
    log_data["evaluate_time_seconds"] = round(model_evaluate_end_time - model_training_end_time, 4)

    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)

process_end_time = time.time()

print(f"\033[32mProcess: {gpu_index} FINISHED. Total time: {(process_end_time - process_start_time):.4f} seconds | Models: {models_range_to_run}\033[0m")
