import tensorflow as tf


def h_flip(x, apply):
    return tf.image.flip_left_right(x) if apply else x


def v_flip(x, apply):
    return tf.image.flip_up_down(x) if apply else x


def rotate(x, angle):
    k = angle // 90 if angle >= 0 else (angle + 360) // 90
    return tf.image.rot90(x, k)


def h_shift(x, distance):
    return tf.manip.roll(x, distance, axis=0)


def v_shift(x, distance):
    return tf.manip.roll(x, distance, axis=1)


def gmean(x):
    g_pow = 1 / x.get_shape().as_list()[0]
    x = tf.reduce_prod(x, axis=0, keepdims=True)
    x = tf.pow(x, g_pow)
    return x


def mean(x):
    return tf.reduce_mean(x, axis=0, keepdims=True)


def max(x):
    return tf.reduce_max(x, axis=0, keepdims=True)
