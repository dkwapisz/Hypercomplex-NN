from keras_tuner.src.backend.io import tf


def yuv_permutation(color_space, image):
    image = tf.image.rgb_to_yuv(image)
    y, u, v = tf.split(image, 3, axis=-1)
    magnitude = tf.sqrt(u ** 2 + v ** 2)
    theta = tf.atan2(v, u)

    u = magnitude * tf.cos(2 * theta)
    v = magnitude * tf.sin(2 * theta)
    fourth_channel = tf.exp(-magnitude)

    if color_space == "YUV-1":
        return tf.concat([y, u, v, fourth_channel], axis=-1)
    elif color_space == "YUV-2":
        return tf.concat([y, u, fourth_channel, v], axis=-1)
    elif color_space == "YUV-3":
        return tf.concat([y, v, u, fourth_channel], axis=-1)
    elif color_space == "YUV-4":
        return tf.concat([y, v, fourth_channel, u], axis=-1)
    elif color_space == "YUV-5":
        return tf.concat([y, fourth_channel, u, v], axis=-1)
    elif color_space == "YUV-6":
        return tf.concat([y, fourth_channel, v, u], axis=-1)
    elif color_space == "YUV-7":
        return tf.concat([u, y, v, fourth_channel], axis=-1)
    elif color_space == "YUV-8":
        return tf.concat([u, y, fourth_channel, v], axis=-1)
    elif color_space == "YUV-9":
        return tf.concat([u, v, y, fourth_channel], axis=-1)
    elif color_space == "YUV-10":
        return tf.concat([u, v, fourth_channel, y], axis=-1)
    elif color_space == "YUV-11":
        return tf.concat([u, fourth_channel, y, v], axis=-1)
    elif color_space == "YUV-12":
        return tf.concat([u, fourth_channel, v, y], axis=-1)
    elif color_space == "YUV-13":
        return tf.concat([v, y, u, fourth_channel], axis=-1)
    elif color_space == "YUV-14":
        return tf.concat([v, y, fourth_channel, u], axis=-1)
    elif color_space == "YUV-15":
        return tf.concat([v, u, y, fourth_channel], axis=-1)
    elif color_space == "YUV-16":
        return tf.concat([v, u, fourth_channel, y], axis=-1)
    elif color_space == "YUV-17":
        return tf.concat([v, fourth_channel, y, u], axis=-1)
    elif color_space == "YUV-18":
        return tf.concat([v, fourth_channel, u, y], axis=-1)
    elif color_space == "YUV-19":
        return tf.concat([fourth_channel, y, u, v], axis=-1)
    elif color_space == "YUV-20":
        return tf.concat([fourth_channel, y, v, u], axis=-1)
    elif color_space == "YUV-21":
        return tf.concat([fourth_channel, u, y, v], axis=-1)
    elif color_space == "YUV-22":
        return tf.concat([fourth_channel, u, v, y], axis=-1)
    elif color_space == "YUV-23":
        return tf.concat([fourth_channel, v, y, u], axis=-1)
    elif color_space == "YUV-24":
        return tf.concat([fourth_channel, v, u, y], axis=-1)