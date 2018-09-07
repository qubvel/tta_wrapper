import itertools
from ..tta_wrapper import functional as F


def add_identity(params):
    """Add identical parameter (0) for all manipulations"""
    if params is None:
        res = [0]
    elif isinstance(params, bool):
        if params:
            res = [0] + [int(params)]
        else:
            res = [0]
    elif isinstance(params, tuple):
        res = [0] + list(params)
    elif isinstance(params, list):
        res = [0] + params
    else:
        raise ValueError('Wrong param type')
    return res


def invert(params):
    """Invert order of parameters for manipulations"""
    return list(map(lambda x: -x, params))


class Augmentation(object):

    def __init__(self,
                 h_flip=True,
                 v_flip=True,
                 h_shifts=(10, -10),
                 v_shifts=(10, -10),
                 rotation_angles=(90, 180, 270),):

        super().__init__()

        self.h_flip = add_identity(h_flip)
        self.v_flip = add_identity(v_flip)
        self.rotation_angles = add_identity(rotation_angles)
        self.h_shifts = add_identity(h_shifts)
        self.v_shifts = add_identity(v_shifts)

        self.n_transforms = len(self.h_flip) * \
                            len(self.v_flip)* \
                            len(self.rotation_angles) * \
                            len(self.h_shifts) * \
                            len(self.v_shifts)

        self.forward_aug = [F.h_flip, F.v_flip, F.rotate, F.h_shift, F.v_shift]
        self.forward_transform_params = list(itertools.product(
            self.h_flip,
            self.v_flip,
            self.rotation_angles,
            self.h_shifts,
            self.v_shifts,
        ))

        self.backward_aug = self.forward_aug[::-1]

        backward_transform_params = list(itertools.product(
            invert(self.h_flip),
            invert(self.v_flip),
            invert(self.rotation_angles),
            invert(self.h_shifts),
            invert(self.v_shifts),
        ))
        self.backward_transform_params = list(map(lambda x: x[::-1],
                                        backward_transform_params))

    @property
    def forward(self):
        return self.forward_aug, self.forward_transform_params

    @property
    def backward(self):
        return self.backward_aug, self.backward_transform_params