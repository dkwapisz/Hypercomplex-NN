# TBD

> TODO Convert it to a proper description
```python
def perform_hypercomplex_transformation(image, color_space):
    if color_space == "RGB":
        r, g, b = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(r ** 2 + g ** 2 + b ** 2)
        phase_rg = tf.atan2(g, r)
        phase_rb = tf.atan2(b, r)
        return tf.concat([tf.math.log1p(magnitude), phase_rg, phase_rb, magnitude * tf.cos(phase_rg + phase_rb)], axis=-1)
    elif color_space == "HSV":
        image = tf.image.rgb_to_hsv(image)
        h, s, v = tf.split(image, 3, axis=-1)
        theta = h * 2 * np.pi
        return tf.concat([v, tf.exp(s) * tf.cos(theta), tf.exp(s) * tf.sin(theta), tf.sin(v * np.pi)], axis=-1)
    elif color_space == "YUV":
        image = tf.image.rgb_to_yuv(image)
        y, u, v = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(u ** 2 + v ** 2)
        theta = tf.atan2(v, u)
        return tf.concat([y, magnitude * tf.cos(2 * theta), magnitude * tf.sin(2 * theta), tf.exp(-magnitude)], axis=-1)
    elif color_space == "YIQ":
        image = tf.image.rgb_to_yiq(image)
        y, i, q = tf.split(image, 3, axis=-1)
        magnitude = tf.sqrt(i ** 2 + q ** 2)
        theta = tf.atan2(q, i)
        return tf.concat([y, magnitude * tf.cos(2 * theta), magnitude * tf.sin(2 * theta), tf.exp(-magnitude)], axis=-1)
    elif color_space == "CMYK":
        cmy = 1 - image
        k = tf.reduce_min(cmy, axis=-1, keepdims=True)
        denominator = 1 - k + 1e-8
        c = (cmy[..., 0:1] - k) / denominator
        m = (cmy[..., 1:2] - k) / denominator
        y = (cmy[..., 2:3] - k) / denominator
        return tf.concat([c, m, y, k], axis=-1)
```