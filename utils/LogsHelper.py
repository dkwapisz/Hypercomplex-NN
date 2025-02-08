import os

from models.ModelUtils import get_current_model_desc


def get_log_data(model_index, total_model_num, process_index, model_name, epochs):
    return {
        "description": get_current_model_desc(model_index, total_model_num, model_name),
        "model_index": model_index,
        "total_model_num_in_process": total_model_num,
        "process_index": process_index,
        "model_name": model_name,
        "epochs": epochs,
        "model_hyperparameters": None,
        "model_layers_details": None,
        "training_history": None,
        "evaluation_result": None,
        "tuning_time_seconds": None,
        "training_time_seconds": None,
        "evaluate_time_seconds": None
    }

def create_model_result_subdir(model_name):
    subdir_name = model_name["type"] + "_" + model_name["color_space"]
    if model_name["type"] == "HyperComplex":
        subdir_name += "_" + model_name["algebra"]

    result_dir = os.path.join("results", subdir_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def model_summary_to_dict(model):
    summary_list = []
    for i, layer in enumerate(model.layers, start=1):
        layer_info = {
            "layer_name": f"layer{i}",
            "type": layer.__class__.__name__,
            "input": layer.input.shape if hasattr(layer, "input") else None,
            "output": layer.output.shape if hasattr(layer, "output") else None,
            "params": layer.count_params()
        }

        if hasattr(layer, "strides"):
            layer_info["stride"] = layer.strides
        if hasattr(layer, "padding"):
            layer_info["padding"] = layer.padding
        if hasattr(layer, "filters"):
            layer_info["filters"] = layer.filters
        if hasattr(layer, "kernel_size"):
            layer_info["kernel_size"] = layer.kernel_size

        summary_list.append(layer_info)
    return summary_list

def get_model_hyperparams(model):
    return {
        "optimizer": model.optimizer.get_config(),
        "loss": model.loss,
        "metrics": model.metrics_names,
        "learning_rate": float(model.optimizer.learning_rate.numpy()) if hasattr(model.optimizer, "learning_rate") else None,
    }