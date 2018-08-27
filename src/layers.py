import tensorflow as tf
import keras.backend as K
from keras.layers import Layer


class Augmentation(Layer):
    def __init__(self, hflip=True, vflip=True, rotation_angles=(0, 90, 180, 270), **kwargs):

        super().__init__(**kwargs)

        self.flips = ['i']
        if hflip:
            self.flips.append('h')
        if vflip:
            self.flips.append('v')
        if hflip and vflip:
            self.flips.append('hv')

        self.rotation_angles = rotation_angles
        self.n_transforms = len(self.flips) * len(self.rotation_angles)

    def _flip(self, x, flip_type):
        if flip_type == 'h':
            return tf.image.flip_left_right(x)
        elif flip_type == 'v':
            return tf.image.flip_up_down(x)
        elif flip_type == 'hv':
            return tf.image.flip_left_right(tf.image.flip_up_down(x))
        else:
            return x

    def _rotate(self, x, angle):
        k = angle // 90
        x = tf.image.rot90(x, k)
        return x


class TTA(Augmentation):
    def call(self, x):

        outputs = []

        for flip in self.flips:
            for angle in self.rotation_angles:
                outputs.append(self._flip(self._rotate(x, angle), flip))

        x = K.concatenate(outputs, axis=0)

        return x

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0] * self.n_transforms if input_shape[0] else None
        return (batch_size, *input_shape[1:])


class TTAMerge(Augmentation):
    def __init__(self, merge_type='mean', **kwargs):

        super().__init__(**kwargs)

        if merge_type == 'mean':
            self.merge_fn = self.mean_merge
        elif merge_type == 'gmean':
            self.merge_fn = self.gmean_merge
        elif merge_type == 'max':
            self.merge_fn = self.max_merge

    def gmean_merge(self, x):
        g_pow = 1 / self.n_transforms
        x = tf.reduce_prod(x, axis=0, keepdims=True)
        x = tf.pow(x, g_pow)
        return x

    def mean_merge(self, x):
        return K.mean(x, axis=0, keepdims=True)

    def max_merge(self, x):
        return tf.reduce_max(x, axis=0, keepdims=True)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0] // self.n_transforms if input_shape[0] else None
        return (batch_size, *input_shape[1:])


class TTAMergeSegmenatation(TTAMerge):
    def call(self, x):

        outputs = []

        i = 0
        for flip in self.flips:
            for angle in self.rotation_angles:
                outputs.append(self._rotate(self._flip(x[i], flip), (360 - angle)))
                i += 1

        x = K.stack(outputs, axis=0)
        x = self.merge_fn(x)
        return x


class TTAMergeClassification(TTAMerge):
    def call(self, x):
        return self.merge_fn(x)
