# Example of using UNet for frame-by-frame segmentation on the 3DEM
# dataset: http://cvlab.epfl.ch/data/em.
# Trained for 30 epochs (~31 minutes) with random cropping, flips, rotations
# using binary_crossentropy objective yields the following evaluation metrics:
# Training data with no test-time augmentation:
# F1       0.95314318
# prec     0.95377219
# reca     0.95251507
# jacc     0.91048104
# Training data with 8x test-time augmentation averaged:
# F1       0.95245916
# prec     0.95441848
# reca     0.95050794
# jacc     0.90923351
# Testing data with no test-time augmentation:
# F1       0.92137849
# prec     0.91044456
# reca     0.93257827
# jacc     0.85421860
# Testing data with 8x test-time augmentation averaged:
# F1       0.92709804
# prec     0.91988462
# reca     0.93442547
# jacc     0.86410320
# Training with the Jaccard loss results in a slightly worse Jaccard score (0.853),
# but this could also be attributed to a non-optimal learning rate.
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
import argparse
import numpy as np
import tensorflow as tf
import tifffile as tif

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import unet_builder
from deepcalcium.utils.keras_helpers import F1, prec, reca, jacc, dice, dicesq, load_model_with_new_input_shape, MetricsPlotCallback
from deepcalcium.utils.data_utils import INVERTIBLE_2D_AUGMENTATIONS

np.random.seed(865)
tf.set_random_seed(7535)


def batch_gen(imgs, msks, window_shape, batch_size, augment=False):
    """Generator for randomly sampling training and validation data
    with minimal augmentations."""

    rng = np.random
    batch_shape = (batch_size,) + window_shape
    h, w = window_shape

    # Identity, flipping, rotation.
    augment_funcs = [lambda x: x, lambda x: np.fliplr(x), lambda x: np.flipud(x)] + \
        [lambda x: np.rot90(x, n) for n in range(4)]

    # Compute the locations of cells in masks.
    cell_locs = [zip(*np.where(m == 1)) for m in msks]

    while True:
        ib, mb = np.zeros(batch_shape), np.zeros(batch_shape)
        for bidx in range(batch_size):

            # Window boundaries with a random offset and extra care to stay in bounds.
            iidx = rng.randint(0, imgs.shape[0])
            ymax = imgs[iidx].shape[0]
            xmax = imgs[iidx].shape[1]
            cidx = rng.randint(0, len(cell_locs[iidx]))
            cy, cx = cell_locs[iidx][cidx]
            cy = max((h / 2), min(cy, ymax - (h / 2)))
            cx = max((w / 2), min(cx, xmax - (w / 2)))
            y0, x0 = cy - (h / 2), cx - (w / 2)
            y1, x1 = y0 + h, x0 + w

            # Grab windows.
            ib[bidx] = imgs[iidx, y0:y1, x0:x1]
            mb[bidx] = msks[iidx, y0:y1, x0:x1]

            # Apply random selection of augmentations to both image and mask.
            for f in rng.choice(augment_funcs, rng.randint(0, 5)):
                ib[bidx], mb[bidx] = f(ib[bidx]), f(mb[bidx])

            # Sanity check.
            assert np.min(ib) >= 0 and np.max(ib) <= 1
            assert np.min(mb) == 0 and np.max(mb) == 1

        yield ib, mb


def train_3DEM(model_path, weights_path):
    """Use the UNet architecture to train a model from scratch on 3DEM data.
    Use the stored weights and the unet2ds_nf script to evaluate directly on the
    Neurofinder data."""

    cpdir = 'checkpoints/unet2d_3dem'
    model_save_path = '%s/model_tst_jacc.hdf5' % cpdir
    path_imgs_trn = '%s/training.tif' % cpdir
    path_msks_trn = '%s/training_groundtruth.tif' % cpdir
    path_imgs_tst = '%s/testing.tif' % cpdir
    path_msks_tst = '%s/testing_groundtruth.tif' % cpdir
    nb_epochs = 100
    nb_steps_trn = 150
    nb_steps_tst = 12
    batch_size = 16
    window_shape = (256, 256)

    # Import data and define generators.
    imgs_trn = tif.imread(path_imgs_trn) / 255.
    msks_trn = tif.imread(path_msks_trn) / 255.
    imgs_tst = tif.imread(path_imgs_tst) / 255.
    msks_tst = tif.imread(path_msks_tst) / 255.
    gen_trn = batch_gen(imgs_trn, msks_trn, window_shape, batch_size, augment=True)
    gen_tst = batch_gen(imgs_tst, msks_tst, window_shape, batch_size, augment=False)

    # Define UNet Keras model.
    if model_path:
        cobj = {'F1': F1, 'prec': prec, 'reca': reca, 'jacc': jacc,
                'dice': dice, 'dicesq': dicesq, 'tf': tf}
        net = load_model_with_new_input_shape(model_path, window_shape,
                                              custom_objects=cobj)
    else:
        net = unet_builder(window_shape, prop_dropout_base=0.3)

    if weights_path:
        net.load_weights(weights_path)

    # Training callbacks.
    cb = [
        ModelCheckpoint(model_save_path, monitor='val_jacc', mode='max',
                        save_best_only=True, verbose=1),
        CSVLogger('%s/history.csv' % cpdir),
        MetricsPlotCallback('%s/metrics.png' % cpdir)
    ]

    # Network compilation with loss and metrics.
    net.compile(optimizer=Adam(0.0007), loss='binary_crossentropy',
                metrics=[F1, prec, reca, jacc, dice, dicesq])

    # Train using generators for training and validation data.
    net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                      callbacks=cb, verbose=1, validation_data=gen_tst,
                      validation_steps=nb_steps_tst)


def evaluate_3DEM(yt, yp, metrics=[F1, prec, reca, jacc]):
    import keras.backend as K
    kyt = K.variable(yt, dtype='float32')
    kyp = K.variable(yp, dtype='float32')
    for metric in metrics:
        mval = K.get_value(metric(kyt, kyp))
        print('%-8s %.8lf' % (metric.__name__, mval))


def predict_3DEM(model_path):

    cpdir = 'checkpoints/unet2d_3dem'
    path_imgs_trn = '%s/training.tif' % cpdir
    path_msks_trn = '%s/training_groundtruth.tif' % cpdir
    path_imgs_tst = '%s/testing.tif' % cpdir
    path_msks_tst = '%s/testing_groundtruth.tif' % cpdir
    window_shape = (1024, 1024)

    # Load net and set it to use window_shape input size.
    cobj = {'F1': F1, 'prec': prec, 'reca': reca, 'jacc': jacc,
            'dice': dice, 'dicesq': dicesq, 'tf': tf}
    net = load_model_with_new_input_shape(model_path, window_shape, custom_objects=cobj)

    # Load tiffs.
    print('Loading tiffs..')
    imgs_trn = tif.imread(path_imgs_trn) / 255.
    msks_trn = tif.imread(path_msks_trn) / 255.
    imgs_tst = tif.imread(path_imgs_tst) / 255.
    msks_tst = tif.imread(path_msks_tst) / 255.

    # Pad images to fit window shape. Masks keep same shape.
    print('Padding tiffs..')
    h0, w0 = msks_trn.shape[1:]
    h1, w1 = window_shape
    crop = lambda x: x[:, :h0, :w0]
    pad_shape = lambda x: ((0, 0), (0, h1 - x.shape[1]), (0, w1 - x.shape[2]))
    imgs_trn = np.pad(imgs_trn, pad_shape(imgs_trn), mode='reflect')
    imgs_tst = np.pad(imgs_tst, pad_shape(imgs_tst), mode='reflect')

    print('Training evaluation, no test-time augmentation...')
    prds_trn = net.predict(imgs_trn, batch_size=4, verbose=1)
    evaluate_3DEM(msks_trn, crop(prds_trn))

    print('Training evaluation with averaged test-time augmentation...')
    prds_trn = np.zeros(msks_trn.shape)
    for name, aug, inv in INVERTIBLE_2D_AUGMENTATIONS:
        print('Augmentation: %s..' % name)
        assert np.all(imgs_trn == inv(aug(imgs_trn)))
        p = net.predict(aug(imgs_trn), batch_size=4, verbose=1)
        prds_trn += crop(inv(p)) / len(INVERTIBLE_2D_AUGMENTATIONS)
    evaluate_3DEM(msks_trn, prds_trn)

    print('Testing evaluation, no test-time augmentation...')
    prds_tst = net.predict(imgs_tst, batch_size=4, verbose=1)
    evaluate_3DEM(msks_tst, crop(prds_tst))

    print('Testing evaluation with averaged test-time augmentation...')
    prds_tst = np.zeros(msks_tst.shape)
    for name, aug, inv in INVERTIBLE_2D_AUGMENTATIONS:
        print('Augmentation: %s..' % name)
        assert np.all(imgs_tst == inv(aug(imgs_tst)))
        p = net.predict(aug(imgs_tst), batch_size=4, verbose=1)
        prds_tst += crop(inv(p)) / len(INVERTIBLE_2D_AUGMENTATIONS)
    evaluate_3DEM(msks_tst, prds_tst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model runner.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    p = sub.add_parser('train', help='training')
    p.set_defaults(which='train')
    p.add_argument('-m', '--model', help='path to model hdf5 file.')
    p.add_argument('-w', '--weights', help='path to load weights from hdf5 file.')

    p = sub.add_parser('predict', help='training')
    p.set_defaults(which='predict')
    p.add_argument('-m', '--model', help='path to model hdf5 file.', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in {'train', 'predict'}

    if args['which'] == 'train':
        train_3DEM(args['model'], args['weights'])

    if args['which'] == 'predict':
        predict_3DEM(args['model'])
