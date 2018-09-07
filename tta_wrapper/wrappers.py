from keras.models import Model
from keras.layers import Input

from .layers import Repeat, TTA, Merge
from .augmentation import Augmentation


doc = """
    IMPORTANT constraints:
        1) model has to have 1 input and 1 output
        2) inference batch_size = 1
        3) image height == width if rotate augmentation is used

    Args:
        model: instance of Keras model
        h_flip: (bool) horizontal flip
        v_flip: (bool) vertical flip
        h_shifts: (list of int) list of horizontal shifts (e.g. [10, -10])
        v_shifts: (list of int) list of vertical shifts (e.g. [10, -10])
        rotation_angles: (list of int) list of angles (deg) for rotation in range [0, 360),
            should be divisible by 90 deg (e.g. [90, 180, 270])
        merge: one of 'mean', 'gmean' and 'max' - mode of merging augmented
            predictions together.

    Returns:
        Keras Model instance

"""

def tta_segmentation(model,
                     h_flip=False,
                     v_flip=False,
                     h_shifts=None,
                     v_shifts=None,
                     rotation_angles=None,
                     merge='mean'):

    """
    Segmentation model test time augmentation wrapper.
    """
    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shifts=h_shifts,
                       v_shifts=v_shifts,
                       rotation_angles=rotation_angles)

    input_shape = (1, *model.input.shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = TTA(*tta.backward)(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


def tta_classification(model,
                       h_flip=False,
                       v_flip=False,
                       h_shifts=None,
                       v_shifts=None,
                       rotation_angles=None,
                       merge='mean'):
    """
    Classification model test time augmentation wrapper.
    """

    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shifts=h_shifts,
                       v_shifts=v_shifts,
                       rotation_angles=rotation_angles)

    input_shape = (1, *model.input.shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


tta_classification.__doc__ += doc
tta_segmentation.__doc__ += doc
