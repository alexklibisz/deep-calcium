# Neurofinder training and prediction using UNet 2D Summary model.
import argparse
import logging
import numpy as np

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary
from deepcalcium.datasets.neurofinder import load_neurofinder

np.random.seed(865)
logging.basicConfig(level=logging.INFO)


def training(dataset_name, weights_path):
    '''Train on all neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    S_trn, M_trn = load_neurofinder(dataset_name)

    # Setup model.
    model = UNet2DSummary(
        checkpoint_dir='checkpoints/unet_2d_summary_96x96_%s' % dataset_name
    )

    # Training.
    model.fit(
        S_trn, M_trn,               # hdf5 sequences and masks.
        weights_path=weights_path,  # Pre-trained weights.
        window_shape=(96, 96),      # Input/output windows to the network.
        nb_epochs=125,              # Epochs.
        batch_size=32,              # Batch size - adjust based on GPU.
        keras_callbacks=[],         # Custom keras callbacks.
        val_prop=0.2,              # Proportion of each sequence for validation.
    )

    # Evaluate training data performance using neurofinder metrics.
    model.evaluate(
        S_trn, M_trn,
        weights_path=weights_path,
        window_shape=(512, 512),
        batch_size=10,
        random_mean=True
    )


def prediction(dataset_name, weights_path):
    '''Predictions on all neurofinder datasets.'''

    # Load all sequences and masks as hdf5 File objects.
    S, _ = load_neurofinder(dataset_name)

    model = UNet2DSummary(
        checkpoint_dir='checkpoints/unet_2d_summary_96x96_%s' % dataset_name
    )

    # Prediction. Saves predictions to checkpoint directory and returns them
    # as numpy arrays.
    M_prd = model.predict(
        S_tst,                       # hdf5 sequences (no masks).
        weights_path=weights_path,   # Pre-trained weights.
        window_shape=(512, 512),     # Input/output windows to the network.
        batch_size=10,
        random_mean=True,
        save_to_checkpoint_dir=True
    )

    # Make a submission from the predicted masks.


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='CLI for UNet 2D Summary example.')
    sp = ap.add_subparsers(title='actions', description='Choose an action.')

    # Training cli.
    sp_trn = sp.add_parser('train', help='CLI for training.')
    sp_trn.set_defaults(which='train')
    sp_trn.add_argument('dataset', help='dataset name', default='all_train', type=str)
    sp_trn.add_argument('-w', '--weights', help='path to weights', type=str)

    # Prediction cli.
    sp_prd = sp.add_parser('predict', help='CLI for prediction.')
    sp_prd.set_defaults(which='predict')
    sp_prd.add_argument('dataset', help='dataset name', default='all', type=str)
    sp_prd.add_argument('-w', '--weights', help='path to weights',
                        type=str, required=True)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['weights'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['weights'])
