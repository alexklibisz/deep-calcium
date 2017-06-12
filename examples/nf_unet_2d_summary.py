# Neurofinder training and prediction using UNet 2D Summary model.
from time import time
import argparse
import logging
import numpy as np

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary
from deepcalcium.datasets.nf import nf_load_hdf5, nf_submit

np.random.seed(865)
logging.basicConfig(level=logging.INFO)


def training(dataset_name, weights_path):
    '''Train on all neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    S_trn, M_trn = nf_load_hdf5(dataset_name)

    # Remove low-quality datasets.
    bad_names = ['neurofinder.04.00']
    S_trn = [s for s in S_trn if s.attrs['name'] not in bad_names]
    M_trn = [m for m in M_trn if m.attrs['name'] not in bad_names]

    # Setup model.
    model = UNet2DSummary(cpdir='checkpoints/unet_2d_summary_96x96_nf')

    # Training.
    model.fit(
        S_trn, M_trn,               # hdf5 sequences and masks.
        weights_path=weights_path,  # Pre-trained weights.
        window_shape=(96, 96),      # Input/output windows to the network.
        batch_size=100,             # Batch size - adjust based on GPU.
        nb_steps_trn=150,           # Training batches / epoch.
        nb_steps_val=100,           # Validation batches / epoch.
        nb_epochs=50,               # Epochs.
        keras_callbacks=[],         # Custom keras callbacks.
        prop_trn=0.75,              # Proportion of height for training, validation.
        prop_val=0.25,
    )


def evaluation(dataset_name, weights_path):
    '''Evaluate datasets.'''

    S_trn, M_trn = nf_load_hdf5(dataset_name)

    model = UNet2DSummary(cpdir='checkpoints/unet_2d_summary_96x96_nf')

    # Evaluate training data performance using neurofinder metrics.
    model.evaluate(
        S_trn, M_trn,
        weights_path=weights_path,
        window_shape=(512, 512),
        save=True
    )


def prediction(dataset_name, weights_path):
    '''Predictions on all neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    S_tst, _ = nf_load_hdf5(dataset_name)

    model = UNet2DSummary(cpdir='checkpoints/unet_2d_summary_96x96_nf')

    # Prediction. Saves predictions to checkpoint directory and returns them
    # as numpy arrays.
    Mp = model.predict(
        S_tst,                       # hdf5 sequences (no masks).
        weights_path=weights_path,   # Pre-trained weights.
        window_shape=(512, 512),     # Input/output windows to the network.
        save=True
    )

    # Make a submission from the predicted masks.
    json_path = '%s/submission_%d.json' % (model.cpdir, time())
    names = [s.attrs['name'] for s in S_tst]
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
    sp_trn.add_argument('-w', '--weights', help='path to weights')

    # Training cli.
    sp_eva = sp.add_parser('evaluate', help='CLI for training.')
    sp_eva.set_defaults(which='evaluate')
    sp_eva.add_argument('dataset', help='dataset name', default='all_train')
    sp_eva.add_argument('-w', '--weights', help='path to weights', required=True)

    # Prediction cli.
    sp_prd = sp.add_parser('predict', help='CLI for prediction.')
    sp_prd.set_defaults(which='predict')
    sp_prd.add_argument('dataset', help='dataset name', default='all')
    sp_prd.add_argument('-w', '--weights', help='path to weights', required=True)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['weights'])

    if args['which'] == 'evaluate':
        evaluation(args['dataset'], args['weights'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['weights'])
