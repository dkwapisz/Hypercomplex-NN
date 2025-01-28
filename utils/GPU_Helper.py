import sys

import tensorflow as tf


def check_gpu_health():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Error: No GPU detected! Please ensure your environment is properly configured.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"GPU is available! Number of GPUs detected: {len(gpus)}")
        for gpu in gpus:
            print(f"- GPU Name: {gpu.name}")


def set_gpu_device(gpu_index: int):
    gpus = tf.config.list_physical_devices('GPU')

    if gpu_index >= len(gpus):
        raise ValueError(f"Given GPU index: {gpu_index}, but only {len(gpus)} GPUs are available.")

    try:
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPU assigned: {gpu_index}: {gpus[gpu_index]}")
    except RuntimeError as e:
        raise RuntimeError(f"Error during GPU assignment: {e}")