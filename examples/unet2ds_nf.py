# Neurofinder training and prediction using UNet 2D Summary model.
from time import time
import argparse
import logging
import numpy as np
import tensorflow as tf

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary
from deepcalcium.datasets.nf import nf_load_hdf5, nf_submit
from deepcalcium.utils.runtime import funcname

np.random.seed(865)
tf.set_random_seed(7535)
logging.basicConfig(level=logging.INFO)


def training(dataset_name, model_path):
    '''Train on neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    ds_trn = nf_load_hdf5(dataset_name)

    # Remove low-quality datasets.
    bad_names = ['neurofinder.04.00']
    ds_trn = [ds for ds in ds_trn if ds.attrs['name'] not in bad_names]

    # Setup model.
    model = UNet2DSummary(cpdir='checkpoints/unet2ds_128x128_nf')

    # Training.
    model.fit(
        ds_trn,                     # hdf5 series and masks.
        model_path=model_path,      # Keras architecture and weights.
        shape_trn=(128, 128),       # Input/output windows to the network.
        shape_val=(512, 512),
        batch_size_trn=20,          # Batch size.
        nb_steps_trn=250,           # Training batches / epoch.
        nb_epochs=10,               # Epochs.
        keras_callbacks=[],         # Custom keras callbacks.
        prop_trn=0.70,              # Proportion of height for training, validation.
        prop_val=0.25,
    )


def evaluation(dataset_name, model_path):
    """Evaluate datasets - once without test-time augmentation and once with."""

    logger = logging.getLogger(funcname())
    ds_trn = nf_load_hdf5(dataset_name)
    model = UNet2DSummary(cpdir='checkpoints/unet2ds_128x128_nf')

    for aug in [True, False]:
        logger.info('Evaluation with%s.' % ' TTA' if aug else 'out TTA')
        # Evaluate training data performance using neurofinder metrics.
        model.predict(
            ds_trn,
            model_path=model_path,
            window_shape=(512, 512),
            save=False,
            print_scores=True,
            augmentation=aug,          # Test-time augmentation.
        )


def prediction(dataset_name, model_path):
    """Predictions on given datasets with and without test-time augmentation."""

    logger = logging.getLogger(funcname())
    ds_tst = nf_load_hdf5(dataset_name)
    model = UNet2DSummary(cpdir='checkpoints/unet2ds_128x128_nf')

    for aug in [True, False]:
        logger.info('Prediction with%s.' % ' TTA' if aug else 'out TTA')

        # Returns predictions as list of numpy arrays.
        Mp = model.predict(
            ds_tst,                      # hdf5 sequences (no masks).
            model_path=model_path,       # Pre-trained Keras architecture and weights.
            window_shape=(512, 512),     # Input/output windows to the network.
            save=False,
            augmentation=aug
        )

        # Make a submission from the predicted masks.
        json_path = '%s/submission_%d.json' % (model.cpdir, time())
        names = [ds.attrs['name'] for ds in ds_tst]
        nf_submit(Mp, names, json_path)
        json_path = '%s/submission_latest.json' % model.cpdir
        nf_submit(Mp, names, json_path)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='CLI for UNet 2D Summary example.')
    sp = ap.add_subparsers(title='actions', description='Choose an action.')

    # Training cli.
    sp_trn = sp.add_parser('train', help='CLI for training.')
    sp_trn.set_defaults(which='train')
    sp_trn.add_argument('dataset', help='dataset name', default='all_train')
    sp_trn.add_argument('-m', '--model', help='path to model')

    # Training cli.
    sp_eva = sp.add_parser('evaluate', help='CLI for training.')
    sp_eva.set_defaults(which='evaluate')
    sp_eva.add_argument('dataset', help='dataset name', default='all_train')
    sp_eva.add_argument('-m', '--model', help='path to model', required=True)

    # Prediction cli.
    sp_prd = sp.add_parser('predict', help='CLI for prediction.')
    sp_prd.set_defaults(which='predict')
    sp_prd.add_argument('dataset', help='dataset name', default='all')
    sp_prd.add_argument('-m', '--model', help='path to model', required=True)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['model'])

    if args['which'] == 'evaluate':
        evaluation(args['dataset'], args['model'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['model'])
