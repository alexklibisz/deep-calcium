# Neurofinder training and prediction using UNet 2D Summary model.
from time import time
from os import mkdir, path, getpid
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary
from deepcalcium.datasets.nf import nf_load_hdf5, nf_submit, nf_mask_metrics
from deepcalcium.utils.runtime import funcname

np.random.seed(865)
tf.set_random_seed(7535)
logging.basicConfig(level=logging.INFO)


def training(dataset_name, model_path, cpdir):
    '''Train on neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    ds_trn = nf_load_hdf5(dataset_name)

    # # Remove low-quality datasets.
    # bad_names = ['neurofinder.04.00']
    # ds_trn = [ds for ds in ds_trn if ds.attrs['name'] not in bad_names]

    # Setup model.
    model = UNet2DSummary(cpdir=cpdir)

    # Training.
    return model.fit(
        ds_trn,                     # hdf5 series and masks.
        model_path=model_path,      # Keras architecture and weights.
        shape_trn=(128, 128),       # Input/output windows to the network.
        shape_val=(512, 512),
        batch_size_trn=20,          # Batch size.
        nb_steps_trn=100,           # Training batches / epoch.
        nb_epochs=10,               # Epochs.
        keras_callbacks=[],         # Custom keras callbacks.
        prop_trn=0.75,              # Proportion of height for training, validation.
        prop_val=0.25,
    )


def ensemble_training(dataset_name, model_path, nb_members, cpdir):
    """Trains a model nb_members times. Saves a CSV file that can be used to
    make predictions with the ensemble."""

    logger = logging.getLogger(funcname())
    tic = int(time())
    rows, cols = [], ['model_path', 'val_nf_f1_mean']

    if not path.exists(cpdir):
        mkdir(cpdir)

    for i in range(nb_members):

        np.random.seed(int(time()))
        tf.set_random_seed(7535)

        logger.info('Training ensemble member %d.' % i)

        # Member gets its own checkpoint directory.
        member_dir = '%s/member_%d_%03d' % (cpdir, tic, i)
        mkdir(member_dir)

        # Train according to the training function setup.
        history, trained_model_path = training(dataset_name, model_path, cpdir=member_dir)

        # Track the serialized model paths and validation scores.
        rows.append([trained_model_path, np.max(history['val_nf_f1_mean'])])
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv('%s/ensemble_%d.csv' % (cpdir, tic), index=False)


def ensemble_prediction(dataset_name, ensemble_path, cpdir):
    """Uses an ensemble of networks to make predictions."""

    logger = logging.getLogger(funcname())
    model = UNet2DSummary(cpdir=cpdir)
    datasets = nf_load_hdf5(dataset_name)
    datasets_have_masks = len([ds for ds in datasets if 'masks' in ds]) == len(datasets)
    df = pd.read_csv(ensemble_path)

    Mp = []
    paths, val_scores = df['model_path'].values, df['val_nf_f1_mean'].values
    for i, (model_path, val_score) in enumerate(zip(paths, val_scores)):
        logger.info('Member %d path==%s, val=%.4lf' % (i, model_path, val_score))
        _Mp = model.predict(datasets, model_path=model_path, window_shape=(512, 512),
                            save=False, print_scores=datasets_have_masks, augmentation=True)
        # Masks kept as running mean.
        _Mp = [mp / len(paths) for mp in _Mp]
        Mp = _Mp if i == 0 else [Mp[mi] + _Mp[mi] for mi in range(len(_Mp))]

    # Round the activation values.
    Mp = [m.round() for m in Mp]

    # Compute and print scores if masks are available.
    if datasets_have_masks:
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for ds, mp in zip(datasets, Mp):
            if 'masks/max' not in ds:
                continue
            m = ds.get('masks/max')[...]
            prec, reca, incl, excl, comb = nf_mask_metrics(m, mp)
            mean_prec += prec / len(datasets)
            mean_reca += reca / len(datasets)
            mean_comb += comb / len(datasets)
            logger.info('%s: prec=%.3lf, reca=%.3lf, incl=%.3lf, excl=%.3lf, comb=%.3lf' %
                        (ds.attrs['name'], prec, reca, incl, excl, comb))
        logger.info('Mean prec=%.3lf, reca=%.3lf, comb=%.3lf' %
                    (mean_prec, mean_reca, mean_comb))

    # Make a submission from the predicted masks.
    json_path = '%s/submission_%d.json' % (cpdir, time())
    names = [ds.attrs['name'] for ds in datasets]
    nf_submit(Mp, names, json_path)
    json_path = '%s/submission_latest.json' % cpdir
    nf_submit(Mp, names, json_path)


def evaluation(dataset_name, model_path, cpdir):
    """Evaluate datasets - once without test-time augmentation and once with."""

    logger = logging.getLogger(funcname())
    ds_trn = nf_load_hdf5(dataset_name)
    model = UNet2DSummary(cpdir=cpdir)

    for aug in [True, False]:
        logger.info('Evaluation with%s.' % (' TTA' if aug else 'out TTA'))
        # Evaluate training data performance using neurofinder metrics.
        model.predict(
            ds_trn,
            model_path=model_path,
            window_shape=(512, 512),
            save=True,
            print_scores=True,
            augmentation=aug,          # Test-time augmentation.
        )


def prediction(dataset_name, model_path, cpdir):
    """Predictions on given datasets with and without test-time augmentation."""

    logger = logging.getLogger(funcname())
    ds_tst = nf_load_hdf5(dataset_name)
    model = UNet2DSummary(cpdir=cpdir)
    tic = int(time())

    for aug in [True, False]:
        logger.info('Prediction with%s.' % (' TTA' if aug else 'out TTA'))

        # Returns predictions as list of numpy arrays.
        Mp = model.predict(
            ds_tst,                      # hdf5 sequences (no masks).
            model_path=model_path,       # Pre-trained Keras architecture and weights.
            window_shape=(512, 512),     # Input/output windows to the network.
            save=False,
            augmentation=aug
        )

        # Round the activations.
        Mp = [m.round() for m in Mp]

        # Make a submission from the predicted masks.
        json_path = '%s/submission_%d%s.json' % (model.cpdir, tic, ('_TTA' if aug else ''))
        names = [ds.attrs['name'] for ds in ds_tst]
        nf_submit(Mp, names, json_path)
        json_path = '%s/submission_latest%s.json' % (model.cpdir, ('_TTA' if aug else ''))
        nf_submit(Mp, names, json_path)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='CLI for UNet 2D Summary example.')
    sp = ap.add_subparsers(title='actions', description='Choose an action.')

    cpdir = 'checkpoints/unet2ds_nf'

    # Training cli.
    sp_trn = sp.add_parser('train', help='CLI for training.')
    sp_trn.set_defaults(which='train')
    sp_trn.add_argument('dataset', help='dataset name', default='all_train')
    sp_trn.add_argument('-m', '--model', help='path to model')
    sp_trn.add_argument('-c', '--cpdir', help='checkpoint directory', default=cpdir)

    # Training cli.
    sp_eva = sp.add_parser('evaluate', help='CLI for training.')
    sp_eva.set_defaults(which='evaluate')
    sp_eva.add_argument('dataset', help='dataset name', default='all_train')
    sp_eva.add_argument('-m', '--model', help='path to model', required=True)
    sp_eva.add_argument('-c', '--cpdir', help='checkpoint directory', default=cpdir)

    # Prediction cli.
    sp_prd = sp.add_parser('predict', help='CLI for prediction.')
    sp_prd.set_defaults(which='predict')
    sp_prd.add_argument('dataset', help='dataset name', default='all')
    sp_prd.add_argument('-m', '--model', help='path to model', required=True)
    sp_prd.add_argument('-c', '--cpdir', help='checkpoint directory', default=cpdir)

    cpdir = 'checkpoints/unet2ds_nf_ensemble'

    # Ensemble training cli.
    sp_entrn = sp.add_parser('train-ensemble', help='CLI for training an ensemble.')
    sp_entrn.set_defaults(which='train-ensemble')
    sp_entrn.add_argument('dataset', help='dataset name', default='all_train')
    sp_entrn.add_argument('-n', '--members', help='no. members', default=5, type=int)
    sp_entrn.add_argument('-m', '--model', help='path to model')
    sp_entrn.add_argument('-c', '--cpdir', help='checkpoint directory', default=cpdir)

    # Ensemble prediction cli.
    sp_enprd = sp.add_parser('predict-ensemble', help='CLI for ensemble prediction.')
    sp_enprd.set_defaults(which='predict-ensemble')
    sp_enprd.add_argument('dataset', help='dataset name', default='all_train')
    sp_enprd.add_argument('-m', '--model', help='path to model')
    sp_enprd.add_argument('-c', '--cpdir', help='checkpoint directory', default=cpdir)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['model'], args['cpdir'])

    if args['which'] == 'evaluate':
        evaluation(args['dataset'], args['model'], args['cpdir'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['model'], args['cpdir'])

    if args['which'] == 'train-ensemble':
        ensemble_training(args['dataset'], args['model'], args['members'], args['cpdir'])

    if args['which'] == 'predict-ensemble':
        ensemble_prediction(args['dataset'], args['model'], args['cpdir'])
