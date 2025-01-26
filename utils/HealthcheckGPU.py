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
