# import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
# import tensorflow.keras.backend as K

from . import functional as F


class Repeat(Layer):
    """
    Layer for cloning input information
    input_shape = (1, H, W, C)
    output_shape = (N, H, W, C)
    """
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, x):
        return tf.stack([x[0]] * self.n, axis=0)

    def compute_output_shape(self, input_shape):
        return (self.n, *input_shape[1:])


class TTA(Layer):

    def __init__(self, functions, params):
        super().__init__()
        self.functions = functions
        self.params = params

    def apply_transforms(self, images):
        transformed_images = []
        for i, args in enumerate(self.params):
            image = images[i]
            for f, arg in zip(self.functions, args):
                image = f(image, arg)
            transformed_images.append(image)
        return tf.stack(transformed_images, 0)

    def call(self, images):
        return self.apply_transforms(images)


class Merge(Layer):

    def __init__(self, type):
        super().__init__()
        self.type = type

    def build(self, input_shape):
        w_init = tf.ones_initializer()
        if self.type is None:
            self.theta = tf.Variable(
                initial_value=w_init(input_shape),
                dtype='float32',
                trainable=False
            )
        elif self.type == 'ClassTTA':
            self.theta = tf.Variable(
                initial_value=w_init(input_shape),
                dtype='float32',
                trainable=True
            )
        elif self.type == 'AugTTA':
            self.theta = tf.Variable(
                initial_value=w_init((1, *input_shape[:-1])),
                dtype='float32',
                trainable=True
            )

    def merge(self, x):
        if self.type is None:
            return x
        elif self.type == 'mean':
            return F.mean(x)
        elif self.type == 'gmean':
            return F.gmean(x)
        elif self.type == 'max':
            return F.max(x)
        elif self.type == 'ClassTTA':
            return F.mean(tf.multiply(self.theta, x))
        elif self.type == 'AugTTA':
            return tf.matmul(self.theta, x)
        else:
            raise ValueError(f'Wrong merge type {type}')

    def call(self, x):
        return self.merge(x)

    def compute_output_shape(self, input_shape):
        return (1, *input_shape[1:])
