import numpy as np

# Augmentations that can be applied to a batch of 2D images and inverted.
# Structure is the augmentation name, the augmentation, and the inverse
# of the augmentation. Intended for test-time augmentation for segmentation.
INVERTIBLE_2D_BATCH_AUGMENTATIONS = [
    ('identity',
     lambda x: x,
     lambda x: x),
    ('vflip',
     lambda x: x[:, ::-1, ...],
     lambda x: x[:, ::-1, ...]),
    ('hflip',
     lambda x: x[:, :, ::-1],
     lambda x: x[:, :, ::-1]),
    ('rot90',
     lambda x: np.rot90(x, 1, axes=(1, 2)),
     lambda x: np.rot90(x, -1, axes=(1, 2))),
    ('rot180',
     lambda x: np.rot90(x, 2, axes=(1, 2)),
     lambda x: np.rot90(x, -2, axes=(1, 2))),
    ('rot270',
     lambda x: np.rot90(x, 3, axes=(1, 2)),
     lambda x: np.rot90(x, -3, axes=(1, 2))),
    ('rot90vflip',
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...],
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...]),
    ('rot90hflip',
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1],
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1])
]
