# TTA wrapper
Test time augmnentation wrapper for keras image segmentation and classification models.

## Description

### How it works?

Wrapper add augmentation layers to your Keras model like this:

```
          Input
            |           # input image; shape 1, H, W, C
       / / / \ \ \      # duplicate image for augmentation; shape N, H, W, C
      | | |   | | |     # apply augmentations (flips, rotation, shifts)
     your Keras model
      | | |   | | |     # reverse transformations
       \ \ \ / / /      # merge predictions (mean, max, gmean)
            |           # output mask; shape 1, H, W, C
          Output
```

### Arguments

  - `h_flip` - bool, horizontal flip augmentation
  - `v_flip` - bool, vertical flip augmentation
  - `rotataion` - list, allowable angles - 90, 180, 270
  - `h_shift` - list of int, horizontal shift augmentation in pixels
  - `v_shift` - list of int, vertical shift augmentation in pixels
  - `add` - list of int/float, additive factor (aug_image = image + factor)
  - `mul` - list of int/float, additive factor (aug_image = image * factor)
  - `contrast` - list of int/float, contrast adjustment factor (aug_image = (image - mean) * factor + mean)
  - `merge` - one of 'mean', 'gmean' and 'max' - mode of merging augmented predictions together
  
### Constraints
  1) model has to have 1 `input` and 1 `output`
  2) inference `batch_size == 1`
  3) image `height == width` if `rotation_angles` augmentation is used


## Example
```python
from keras.models import load_model
from tta_wrapper import tta_segmentation

model = load_model('path/to/model.h5')
tta_model = tta_segmentation(model, h_flip=True, rotation_angles=(90, 270), 
                             h_shifts=(-5, 5), merge='mean')
y = tta_model.predict(x)
```
