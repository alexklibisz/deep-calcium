# Trace segmentation using C2S library.
import argparse
import logging
import numpy as np
import os

import sys
sys.path.append('.')
from examples.spikes.sjdata_preprocess import preprocess
from deepcalcium.models.spikes.c2s_segmentation import C2SSegmentation


def training(dataset_name, model_path, cpdir, dsdir):
    np.random.seed(int(os.getpid()))
    dataset_paths = preprocess(dataset_name, cpdir, dsdir)
    model = C2SSegmentation(cpdir)
    return model.fit(
        dataset_paths,
        prop_trn=0.75,
        prop_val=0.25,
        error_margin=2
    )


def prediction(dataset_name, model_path, cpdir, dsdir):

    # TODO: would be good to define the networks such that any pre-processing is
    # built-in as a layer to make prediction simpler less tightly-coupled to the
    # code that was used when the network was defined.

    pass


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    DSDIR = '%s/.deep-calcium-datasets/stjude' % os.path.expanduser('~')
    CPDIR = 'checkpoints/sjspikes_c2s'

    if not os.path.exists(DSDIR):
        os.mkdir(DSDIR)

    if not os.path.exists(CPDIR):
        os.mkdir(CPDIR)

    ap = argparse.ArgumentParser(description='C2S trace segmentations.')
    sp = ap.add_subparsers(title='actions', description='Choose an action.')

    # Training cli.
    sp_trn = sp.add_parser('train', help='CLI for training.')
    sp_trn.set_defaults(which='train')
    sp_trn.add_argument('dataset', help='dataset name', default='all')
    sp_trn.add_argument('-m', '--model', help='path to model')
    sp_trn.add_argument('-c', '--cpdir', help='checkpoint directory',
                        default=CPDIR)
    sp_trn.add_argument('-d', '--dsdir', help='datasets directory',
                        default=DSDIR)

    # Training cli.
    sp_eva = sp.add_parser('predict', help='CLI for training.')
    sp_eva.set_defaults(which='predict')
    sp_eva.add_argument('dataset', help='dataset name', default='all')
    sp_eva.add_argument('-m', '--model', help='path to model', required=False)
    sp_eva.add_argument('-c', '--cpdir', help='checkpoint directory',
                        default=CPDIR)
    sp_eva.add_argument('-d', '--dsdir', help='datasets directory',
                        default=DSDIR)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['model'], args['cpdir'], args['dsdir'])

    if args['which'] == 'predict':
        prediction(args['dataset'], args['model'], args['cpdir'], args['dsdir'])
