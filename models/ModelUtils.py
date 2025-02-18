from HypercomplexKeras import Algebra

algebras = {
    "Quaternions": Algebra.Quaternions,
    "Klein4": Algebra.Klein4,
    "Cl20": Algebra.Cl20,
    "Coquaternions": Algebra.Coquaternions,
    "Cl11": Algebra.Cl11,
    "Bicomplex": Algebra.Bicomplex,
    "Tessarines": Algebra.Tessarines
}

def get_models_range(total_model_num, num_processes, gpu_index):
    models_offset = total_model_num // num_processes
    start_model = models_offset * gpu_index
    end_model = start_model + models_offset

    remainder_models = total_model_num % num_processes
    if remainder_models != 0:
        if gpu_index < remainder_models:
            start_model += gpu_index
            end_model += gpu_index + 1
        else:
            start_model += remainder_models
            end_model += remainder_models

    return start_model, end_model  # (inclusive, exclusive)

def get_current_model_desc(model_index, total_model_num, model_name):
    yellow_start = "\033[93m"
    yellow_end = "\033[0m"
    if model_name["type"] == "HyperComplex":
        return (f"{yellow_start}Training model {model_index + 1}/{total_model_num}: "
              f"Type: {model_name['type']} | "
              f"Color space: {model_name['color_space']} | "
              f"Algebra: {model_name['algebra']}{yellow_end}")
    else:
        return (f"{yellow_start}Training model {model_index + 1}/{total_model_num}: "
              f"Type: {model_name['type']} | "
              f"Color space: {model_name['color_space']}{yellow_end}")