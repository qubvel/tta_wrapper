from .layers import TTA, TTAMergeClassification, TTAMergeSegmenatation
from keras.models import Model
from keras.layers import Input

def tta_classifier(model, horizontal_flip=False, vertical_flip=False,
                   rotation_angles=(0,), merge_type='mean'):
    batch_shape = (1, model.input.shape[1:])
    inp = Input(batch_shape=batch_shape)
    x = TTA(hflip=horizontal_flip, vflip=vertical_flip, rotation_angles=rotation_angles)(inp)
    x = model(x)
    x = TTAMergeClassification(hflip=horizontal_flip, vflip=vertical_flip,
                               rotation_angles=rotation_angles, merge_type=merge_type)(x)
    tta_model = Model(inp, x)
    return tta_model


def tta_segmentator(model, horizontal_flip=False, vertical_flip=False,
                    rotation_angles=(0,), merge_type='mean'):

    batch_shape = (1, model.input.shape[1:])
    inp = Input(batch_shape=batch_shape)
    x = TTA(hflip=horizontal_flip, vflip=vertical_flip, rotation_angles=rotation_angles)(inp)
    x = model(x)
    x = TTAMergeSegmenatation(hflip=horizontal_flip, vflip=vertical_flip,
                              rotation_angles=rotation_angles, merge_type=merge_type)(x)
    tta_model = Model(inp, x)
    return tta_model

