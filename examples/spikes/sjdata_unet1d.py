# Example script for calcium trace segmentation using a labeled dataset.
# - Converts datasets from matlab to HDF5 format.
# - Extracts a calcium trace from all ROIS across multiple imaging series.
# - Stores the calcium trace and corresponding spike labels as arrays in HDF5.
# - Trains and predicts on new data.
# Execution starts at starts at if __name__ == "__main__":
# Optional modifications around the lines with comment "CONFIG".
import argparse
import logging
import numpy as np
import os

np.random.seed(865)

import sys
sys.path.append('.')
from examples.spikes.sjdata_preprocess import preprocess
from deepcalcium.models.spikes.unet_1d_segmentation import UNet1DSegmentation


def training(dataset_name, model_path, cpdir, dsdir):
    dataset_paths = preprocess(dataset_name, cpdir, dsdir)
    model = UNet1DSegmentation(cpdir=cpdir)
    return model.fit(
        dataset_paths,
        model_path=model_path,
        val_type='random_split',
        prop_trn=0.8,
        prop_val=0.2,
        nb_epochs=75,
        error_margin=2
    )


def crossval(dataset_name, model_path, cpdir, dsdir):
    dataset_paths = preprocess(dataset_name, cpdir, dsdir)
    model = UNet1DSegmentation(cpdir=cpdir)
    return model.fit(
        dataset_paths,
        model_path=model_path,
        val_type='cross_validate',
        nb_folds=5,
        nb_epochs=75,
        error_margin=2
    )


def prediction(dataset_name, model_path, cpdir, dsdir):
    # TODO: implement this.
    pass


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    DSDIR = '%s/.deep-calcium-datasets/stjude' % os.path.expanduser('~')
    CPDIR = 'checkpoints/sjspikes_unet1d'

    if not os.path.exists(DSDIR):
        os.mkdir(DSDIR)

    if not os.path.exists(CPDIR):
        os.mkdir(CPDIR)

    ap = argparse.ArgumentParser(description='UNet1D trace segmentations.')
    sp = ap.add_subparsers(title='actions', description='Choose an action.')

    # Training cli.
    sp_trn = sp.add_parser('train', help='CLI for training.')
    sp_trn.set_defaults(which='train')
    sp_trn.add_argument('dataset', help='dataset name', default='all')
    sp_trn.add_argument('-m', '--model', help='path to model')
    sp_trn.add_argument('-c', '--cpdir', help='checkpoint dir', default=CPDIR)
    sp_trn.add_argument('-d', '--dsdir', help='datasets dir', default=DSDIR)

    # Cross-validation cli.
    sp_cvl = sp.add_parser('crossval', help='CLI for crossvalidation.')
    sp_cvl.set_defaults(which='crossval')
    sp_cvl.add_argument('dataset', help='dataset name', default='all')
    sp_cvl.add_argument('-m', '--model', help='path to model')
    sp_cvl.add_argument('-c', '--cpdir', help='checkpoint dir', default=CPDIR)
    sp_cvl.add_argument('-d', '--dsdir', help='datasets dir', default=DSDIR)

    # Prediction cli.
    sp_eva = sp.add_parser('predict', help='CLI for training.')
    sp_eva.set_defaults(which='predict')
    sp_eva.add_argument('dataset', help='dataset name', default='all')
    sp_eva.add_argument('-m', '--model', help='path to model', required=True)
    sp_eva.add_argument('-c', '--cpdir', help='checkpoint dir', default=CPDIR)
    sp_eva.add_argument('-d', '--dsdir', help='datasets dir', default=DSDIR)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['model'], args['cpdir'], args['dsdir'])

    if args['which'] == 'crossval':
        crossval(args['dataset'], args['model'], args['cpdir'], args['dsdir'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['model'], args['cpdir'], args['dsdir'])
