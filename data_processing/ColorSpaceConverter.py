import tensorflow as tf
from keras.src.utils import image_dataset_from_directory

def normalize_zscore(image):
  mean = tf.reduce_mean(image)
  std = tf.math.reduce_std(image)
  return (image - mean) / std


def convert_color_tf(image, color_space):
    image = tf.image.convert_image_dtype(image, tf.float32)

    if color_space == "RGB":
        pass
    elif color_space == "HSV":
        image = tf.image.rgb_to_hsv(image)
    elif color_space == "YUV":
        image = tf.image.rgb_to_yuv(image)
    elif color_space == "YIQ":
        image = tf.image.rgb_to_yiq(image)
    elif color_space == "CMYK":
        image = 1 - image
        k = tf.reduce_min(image, axis=-1, keepdims=True)
        cmy = (image - k) / (1 - k + 1e-8)
        image = tf.concat([cmy, k], axis=-1)
    else:
        raise ValueError(f"{color_space} color space is not supported.")

    return normalize_zscore(image)


def create_dataset_tf(dir_path, img_size, batch_size=32, color_space="RGB"):
    dataset = image_dataset_from_directory(dir_path, image_size=img_size, batch_size=batch_size,
                                           label_mode="categorical")

    def process_image(image, label):
        image = convert_color_tf(image, color_space)
        return image, label

    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
