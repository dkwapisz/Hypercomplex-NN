import numpy as np
import tensorflow as tf
from keras.src.utils import image_dataset_from_directory

def normalize_zscore(image):
  mean = tf.reduce_mean(image)
  std = tf.math.reduce_std(image)
  return (image - mean) / std

def perform_basic_transformation(image, color_space):
    if "RGB" in color_space:
        return image
    elif "HSV" in color_space:
        return tf.image.rgb_to_hsv(image)
    elif "YUV" in color_space:
        return tf.image.rgb_to_yuv(image)
    elif "YIQ" in color_space:
        return tf.image.rgb_to_yiq(image)
    elif "CMYK" in color_space:
        cmy = 1 - image
        k = tf.reduce_min(cmy, axis=-1, keepdims=True)
        denominator = 1 - k + 1e-8
        c = (cmy[..., 0:1] - k) / denominator
        m = (cmy[..., 1:2] - k) / denominator
        y = (cmy[..., 2:3] - k) / denominator
        return tf.concat([c, m, y, k], axis=-1)

def perform_hypercomplex_transformation(image, color_space):
    if "RGB" in color_space :
        r, g, b = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(r ** 2 + g ** 2 + b ** 2)
        phase_rg = tf.atan2(g, r)
        phase_rb = tf.atan2(b, r)
        return tf.concat([tf.math.log1p(magnitude), phase_rg, phase_rb, magnitude * tf.cos(phase_rg + phase_rb)], axis=-1)
    elif "HSV" in color_space:
        image = tf.image.rgb_to_hsv(image)
        h, s, v = tf.split(image, 3, axis=-1)
        theta = h * 2 * np.pi
        return tf.concat([v, s * tf.cos(theta), s * tf.sin(theta), v * tf.cos(theta)], axis=-1)
    elif "YUV" in color_space:
        image = tf.image.rgb_to_yuv(image)
        y, u, v = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(u ** 2 + v ** 2)
        theta = tf.atan2(v, u)
        return tf.concat([y, magnitude * tf.cos(2 * theta), magnitude * tf.sin(2 * theta), tf.exp(-magnitude)], axis=-1)
    elif "YIQ" in color_space:
        image = tf.image.rgb_to_yiq(image)
        y, i, q = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(i ** 2 + q ** 2)
        theta = tf.atan2(q, i)
        return tf.concat([y, magnitude * tf.cos(2 * theta), magnitude * tf.sin(2 * theta), tf.exp(-magnitude)], axis=-1)
    elif "CMYK" in color_space:
        cmy = 1 - image
        k = tf.reduce_min(cmy, axis=-1, keepdims=True)
        denominator = 1 - k + 1e-8
        c = (cmy[..., 0:1] - k) / denominator
        m = (cmy[..., 1:2] - k) / denominator
        y = (cmy[..., 2:3] - k) / denominator
        return tf.concat([c, m, y, k], axis=-1)

def convert_color_tf(image, color_space, hypercomplex=False):
    image = tf.image.convert_image_dtype(image, tf.float32)

    if hypercomplex or "transformed" in color_space:
        image = perform_hypercomplex_transformation(image, color_space)
    else:
        image = perform_basic_transformation(image, color_space)

    image = normalize_zscore(image)

    return image


def create_dataset_tf(dir_path, img_size, batch_size=64, color_space="RGB", hypercomplex=False):
    dataset = image_dataset_from_directory(dir_path, image_size=img_size, batch_size=batch_size,
                                           label_mode="categorical")

    def process_image(image, label):
        image = convert_color_tf(image, color_space, hypercomplex)
        return image, label

    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
