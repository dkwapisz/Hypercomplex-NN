import cv2
import numpy as np
import tensorflow as tf
from keras.src.utils import image_dataset_from_directory


def convert_to_hsv(image):
    image_np = tf.image.rgb_to_hsv(image)
    return image_np

# TODO Fix
def convert_to_cielab(image):
    image_ndarray = tf.make_ndarray(image)
    cielab_image = cv2.cvtColor(image_ndarray, cv2.COLOR_RGB2LAB)
    return tf.convert_to_tensor(cielab_image, dtype=tf.float32)

# TODO Fix
def convert_to_ycbcr(image):
    image_ndarray = tf.make_ndarray(image)
    ycbcr_image = cv2.cvtColor(image_ndarray, cv2.COLOR_RGB2YCrCb)
    return tf.convert_to_tensor(ycbcr_image, dtype=tf.float32)

# TODO Fix
def convert_to_cmyk(image):
    image_ndarray = tf.make_ndarray(image)
    K = 1 - np.max(image_ndarray, axis=2)
    C = (1 - image_ndarray[..., 0] - K) / (1 - K + 1e-10)
    M = (1 - image_ndarray[..., 1] - K) / (1 - K + 1e-10)
    Y = (1 - image_ndarray[..., 2] - K) / (1 - K + 1e-10)
    C = np.nan_to_num(C)
    M = np.nan_to_num(M)
    Y = np.nan_to_num(Y)
    cmyk_image = np.stack((C, M, Y, K), axis=2)
    return tf.convert_to_tensor(cmyk_image, dtype=tf.float32)


def preprocess_image(image, color_space="HSV"):
    image = tf.image.convert_image_dtype(image / 255.0, tf.float32)

    match color_space:
        case "RGB":
            return image
        case "HSV":
            return convert_to_hsv(image)
        case "CIELab":
            return convert_to_cielab(image)
        case "YCbCr":
            return convert_to_ycbcr(image)
        case "CMYK":
            return convert_to_cmyk(image)
        case _:
            return image


def create_dataset(dataset_path, image_size, color_space="RGB"):
    dataset = image_dataset_from_directory(
        dataset_path,
        label_mode='categorical',
        image_size=image_size,
        batch_size=512
    )

    def preprocess(image, label):
        return preprocess_image(image, color_space), label

    return dataset.map(preprocess)