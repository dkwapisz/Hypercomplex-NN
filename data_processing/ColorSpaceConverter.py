import tensorflow as tf
import numpy as np
import cv2

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def convert_to_hsv(image):
    image_np = image.numpy()
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    return tf.convert_to_tensor(hsv_image, dtype=tf.float32)


def convert_to_cielab(image):
    image_np = image.numpy()
    cielab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    return tf.convert_to_tensor(cielab_image, dtype=tf.float32)


def convert_to_ycbcr(image):
    image_np = image.numpy()
    ycbcr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    return tf.convert_to_tensor(ycbcr_image, dtype=tf.float32)


def convert_to_cmyk(image):
    image_np = image.numpy()
    K = 1 - np.max(image_np, axis=2)
    C = (1 - image_np[..., 0] - K) / (1 - K + 1e-10)
    M = (1 - image_np[..., 1] - K) / (1 - K + 1e-10)
    Y = (1 - image_np[..., 2] - K) / (1 - K + 1e-10)
    C = np.nan_to_num(C)
    M = np.nan_to_num(M)
    Y = np.nan_to_num(Y)
    cmyk_image = np.stack((C, M, Y, K), axis=2)
    return tf.convert_to_tensor(cmyk_image, dtype=tf.float32)


def preprocess_image(image_path, color_space="HSV"):
    image = load_and_preprocess_image(image_path)
    if color_space == "HSV":
        image = tf.py_function(convert_to_hsv, [image], Tout=tf.float32)
    elif color_space == "CIELab":
        image = tf.py_function(convert_to_cielab, [image], Tout=tf.float32)
    elif color_space == "YCbCr":
        image = tf.py_function(convert_to_ycbcr, [image], Tout=tf.float32)
    elif color_space == "CMYK":
        image = tf.py_function(convert_to_cmyk, [image], Tout=tf.float32)
    return image


def create_dataset(image_paths, batch_size, color_space="HSV"):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: preprocess_image(x, color_space=color_space),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset