# Neurofinder training and prediction using UNet 2D Summary model.
from sklearn.model_selection import KFold
from time import time
import argparse
import logging
import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('.')

from deepcalcium.models.neurons.neuron_segmentation import NeuronSegmentation, \
    summary_imgs_kmeans, summary_msks_max_erosion, augmentation_kmeans
from deepcalcium.datasets.neurofinder_helpers import neurofinder_load_hdf5, \
    NEUROFINDER_NAMES_ALL, NEUROFINDER_NAMES_TRAIN, NEUROFINDER_NAMES_TEST
from deepcalcium.utils.runtime import funcname
from deepcalcium.models.networks import unet2d
from deepcalcium.utils.config import CHECKPOINTS_DIR, DATASETS_DIR

np.random.seed(865)
tf.set_random_seed(7535)
logging.basicConfig(level=logging.INFO)

def cross_validate(dataset_names, checkpoints_dir):
    """Leave-one-out cross-validation on neurofinder datasets."""

    # HDF5 paths to all series and masks.
    hdf5_paths = neurofinder_load_hdf5(dataset_names)

    # Split datasets and run cross-validation.
    histories = []
    splits = KFold(len(hdf5_paths)).split(hdf5_paths)
    for idxs_train, idxs_validate in splits:
        hdf5_paths_train = [hdf5_paths[i] for i in idxs_train]
        hdf5_paths_validate = [hdf5_paths[i] for i in idxs_validate]

        # Model set up with kmeans clustered summaries.
        k, shape = 10, (256, 256)
        model = NeuronSegmentation(
            checkpoints_dir=checkpoints_dir,
            network_func=unet2d,
            imgs_summary_func=lambda p: summary_imgs_kmeans(p, k, shape),
            msks_summary_func=summary_msks_max_erosion,
            fit_shape=(128, 128, k),
            fit_augmentation_func=augmentation_kmeans
        )

        # Fit and store history.
        history = model.fit(hdf5_paths_train, hdf5_paths_validate)
        histories.append(history)


def train(dataset_names, checkpoints_dir):
    """Train and validate on neurofinder datasets."""

    # HDF5 paths to all series and masks.
    hdf5_paths = neurofinder_load_hdf5(dataset_names)

    # Split data 80/20.
    np.random.shuffle(hdf5_paths)
    split = int(0.8 * len(hdf5_paths))
    hdf5_paths_train = hdf5_paths[:split]
    hdf5_paths_validate = hdf5_paths[split:]

    # Model set up with kmeans clustered summaries.
    k, shape = 10, (256, 256)
    model = NeuronSegmentation(
        checkpoints_dir=checkpoints_dir,
        network_func=unet2d,
        imgs_summary_func=lambda p: summary_imgs_kmeans(p, k, shape),
        msks_summary_func=summary_msks_max_erosion,
        fit_shape=(128, 128, k),
        fit_augmentation_func=augmentation_kmeans
    )

    # Train.
    model.fit(hdf5_paths_train, hdf5_paths_validate)

def predict(dataset_names, checkpoints_dir, model_path=None):
    """Predictions on given datasets."""

    hdf5_paths = neurofinder_load_hdf5(dataset_names)
    k, shape = 10, (256, 256)
    model = NeuronSegmentation(
        checkpoints_dir=checkpoints_dir,
        model_path=model_path,
        network_func=unet2d,
        imgs_summary_func=lambda p: summary_imgs_kmeans(p, k, shape),
        msks_summary_func=summary_msks_max_erosion,
        predict_shape=(512, 512, k),
        fit_augmentation_func=augmentation_kmeans
    )
    msks_predicted = model.predict(hdf5_paths)

if __name__ == "__main__":

    # Make checkpoints directory for this particular model.
    checkpoints_dir = '%s/neurons/neurofinder' % CHECKPOINTS_DIR
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Command line argument parser.
    ap = argparse.ArgumentParser(description='CLI for neurofinder segmentation')
    sp = ap.add_subparsers(title='actions', description='Choose an action')

    # Training CLI.
    x = sp.add_parser('train', help='training')
    x.set_defaults(which='train')
    x.add_argument('-d', '--dataset_names', help='dataset name(s)', type=str,
                   default=','.join(NEUROFINDER_NAMES_TRAIN))
    x.add_argument('-c', '--checkpoints_dir', help='checkpoint directory',
                   default=checkpoints_dir)

    # Cross-validation CLI.
    x = sp.add_parser('crossval', help='cross validation')
    x.set_defaults(which='crossval')
    x.add_argument('-d', '--dataset_names', help='dataset name(s)', type=str,
                   default=','.join(NEUROFINDER_NAMES_TRAIN))
    x.add_argument('-c', '--checkpoints_dir', help='checkpoint directory',
                   default=checkpoints_dir)

    # Prediction CLI.
    x = sp.add_parser('predict', help='prediction')
    x.set_defaults(which='predict')
    x.add_argument('-d', '--dataset_names', help='dataset name(s)', type=str,
                   default=','.join(NEUROFINDER_NAMES_ALL))
    x.add_argument('-m', '--model_path', help='path to model', default=None)
    x.add_argument('-c', '--checkpoints_dir', help='checkpoint directory',
                   default=checkpoints_dir)

    # Map args['which'] -> function
    which_func = { 'train': train, 'predict': predict,
                   'crossval': cross_validate }

    # Parse and run appropriate function.
    args = vars(ap.parse_args())
    f = which_func[args['which']]
    del args['which']
    f(**args)
