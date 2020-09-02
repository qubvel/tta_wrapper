import sys
import numpy as np
import pytest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

from tta_wrapper import tta_classification
from tta_wrapper import tta_segmentation


def identity_model(input_shape):
    inp = Input(input_shape)
    x = Lambda(lambda a: a)(inp)
    identity_model = Model(inp, x)
    return identity_model

# inputs
input_sample = np.arange(9).reshape((1, 3, 3, 1))

# outputs
tta_segmentation_output = input_sample
tta_classification_output = np.ones((1, 3, 3, 1)) * 4

# model
seg_identity_model = identity_model(input_sample.shape[1:])
cls_identity_model = identity_model(input_sample.shape[1:])

@pytest.mark.parametrize("wrapper, base_model, inputs, outputs", 
                         [(tta_segmentation, 
                           seg_identity_model, 
                           input_sample,
                           tta_segmentation_output),
                          (tta_classification,
                          cls_identity_model,
                          input_sample,
                          tta_classification_output)
                          ])
def test_wrapper(wrapper, base_model, inputs, outputs):
    print('[TEST] wrapping model with {} ... '.format(wrapper.__name__))

    params = dict(
        h_flip=True,
        v_flip=True,
        h_shift=(10, -10),
        v_shift=(10, -10),
        rotation=(90, 180, 270),
        merge='mean'
    )

    print('[TEST] parameters: ', params)

    model = wrapper(base_model, **params)
    prediction = model.predict(inputs)

    assert np.allclose(prediction, outputs), f"\nprediction: \n{prediction}\n\nground_truth: \n{outputs}"
    print('[TEST] {} - test passed. '.format(wrapper.__name__))