# Neurofinder training and prediction using UNet 2D Summary model.
from time import time
import argparse
import logging
import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('.')

from deepcalcium.models.neurons import NeuronSegmentation
from deepcalcium.datasets.neurofinder_helpers import neurofinder_load_hdf5, NEUROFINDER_NAMES
from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.config import CHECKPOINTS_DIR, DATASETS_DIR

np.random.seed(865)
tf.set_random_seed(7535)
logging.basicConfig(level=logging.INFO)

def scaffold():
    """Make checkpoints and datasets directories for this particular model."""

    checkpoints_dir = '%s/neurons/unet_neurofinder' % CHECKPOINTS_DIR
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    datasets_dir = '%s/neurons/unet_neurofinder' % DATASETS_DIR
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    return checkpoints_dir, datasets_dir

def train(dataset_names, checkpoints_dir):
    """Cross-validate on all neurofinder datasets"""

    # Default model.
    model = NeuronSegmentation(
        checkpoints_dir=checkpoints_dir,
        model='unet',
        series_summary='kmeans',
        masks_summary='max_erosion',
        fit_shape=(8, 128, 128),
        fit_iters=10e3,
        fit_batch=20,
        fit_valdiation_metrics='neurofinder',
        fit_validation_interval=1000,
        fit_validation_folds=5,
        fit_objective='binary_crossentropy',
        fit_optimizer='adam',
        fit_optimizer_parameters={'lr': 0.002},
        fit_parallelize=True
    )

    # HDF5 paths to all series and masks.
    dataset_hdf5_paths = neurofinder_load_hdf5(dataset_names, n_jobs=-2)

    # Cross-validate.
    model.cross_validate(dataset_hdf5_paths)

def predict(dataset_names, checkpoints_dir, model_path=None):
    """Predictions on given datasets."""

    logger = logging.getLogger(funcname())
    model = NeuronSegmentation(
        checkpoints_dir=checkpoints_dir,
        series_summary='kmeans',
        predict_shape=(8, 512, 512),
        predict_augmentation=True,
        predict_threshold=0.5,
        predict_print_metrics=True,
        predict_save=True,
        model_path=model_path,
    )
    dataset_hdf5_paths = neurofinder_load_hdf5(dataset_names)
    masks_predicted = model.predict(dataset_hdf5_paths)

if __name__ == "__main__":

    checkpoints_dir, datasets_dir = scaffold()

    # Command line argument parser.
    ap = argparse.ArgumentParser(description='CLI for neurofinder segmentation')
    sp = ap.add_subparsers(title='actions', description='Choose an action')

    # Training CLI.
    sp_trn = sp.add_parser('crossval', help='cross-validation')
    sp_trn.set_defaults(which='crossval')
    sp_trn.add_argument('dataset_name', help='dataset name(s)', type=str,
                        default=','.join(NEUROFINDER_NAMES))
    sp_trn.add_argument('-m', '--model_path', help='path to model')
    sp_trn.add_argument('-c', '--checkpoints_dir', help='checkpoint directory',
                        default=checkpoints_dir)

    # Prediction CLI.
    sp_prd = sp.add_parser('predict', help='prediction')
    sp_prd.set_defaults(which='predict')
    sp_prd.add_argument('dataset_name', help='dataset name(s)', type=str,
                        default=','.join(NEUROFINDER_NAMES))
    sp_prd.add_argument('-m', '--model_path', help='path to model',
                        default=model_path)
    sp_prd.add_argument('-c', '--checkpoints_dir', help='checkpoint directory',
                        default=checkpoints_dir)

    # Map args['which'] -> function
    which_func = {
        'train': train,
        'predict': predict
    }

    # Parse and run appropriate function.
    args = vars(ap.parse_args())
    f = which_func[args['which']]
    del args['which']
    f(**args)
