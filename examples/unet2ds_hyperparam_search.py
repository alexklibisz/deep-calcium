# Random hyper-parameter search over multiple GPUs.
# Hashtag deep learning. Run on a box with 4 titan cards.
# Run with this command:
# while true; do CUDA_VISIBLE_DEVICES="1" python
# examples/unet2ds_hyperparam_search.py; sleep 1; done
from keras.optimizers import Adam
from shutil import rmtree
from time import time, sleep
import argparse
import json
import logging
import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary, _build_compile_unet, _summarize_sequence
from deepcalcium.datasets.nf import nf_load_hdf5, nf_submit
from pprint import pprint, pformat

logging.basicConfig(level=logging.INFO)


def single_run(dataset_trn, dataset_tst, CUDA_VISIBLE_DEVICES, nb_samples_per_epoch, nb_epochs):
    '''
    1. Randomly select hyperparameter combinations.
    2. Compile and train the network with this parameter configuration.
    3. Make a submisssion with the network.
    Function can be launched as standalone process but makes more sense to just
    run it in multiple TMUX sessions.
    '''

    # Setup.
    seed = int(time()) + int(os.getpid())
    rng = np.random
    rng.seed(seed)
    tf.set_random_seed(seed)
    ID = '%d_%d' % (int(time()), os.getpid())
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    # Load all sequences and masks as hdf5 File objects.
    ds_trn = nf_load_hdf5(dataset_trn)
    ds_tst = nf_load_hdf5(dataset_tst)

    # Remove low-quality datasets.
    bad_names = ['neurofinder.04.00']
    ds_trn = [ds for ds in ds_trn if ds.attrs['name'] not in bad_names]

    # Choose params.
    p = {
        'seed': seed,
        'wdw':  2**rng.randint(5, 9),
        'batch_size': rng.choice(np.arange(10, 150, 2)),
        'preprocess': rng.choice(['zero_one', 'negative_one_one']),

        # Layer settings.
        'nb_filters_base': rng.choice(np.arange(10, 64, 2)),
        'conv_kernel_init': rng.choice(['he_normal', 'he_uniform']),
        'prop_dropout_base': rng.choice(np.arange(0.05, 0.4, 0.05)),
        'upsampling_or_transpose': rng.choice(['transpose', 'upsampling']),
        'conv_l2_lambda': rng.choice([0, 10**rng.uniform(-5, 5)]),

        # Optimization.
        'learning_rate': 10**rng.uniform(-2, -6),
        'loss': rng.choice(['binary_crossentropy', 'dice_sqaured'])
    }

    # By default scales to [0,1], alternatively to [-1,1].
    series_summary_func = _summarize_sequence
    if p['preprocess'] == 'negative_one_one':
        series_summary_func = lambda x: _summarize_sequence(x) * 2 - 1

    pprint(p)
    sleep(3)

    # Wrap the net builder function to use these parameters.
    def build_unet(window_shape):
        return _build_compile_unet(window_shape=window_shape,
                                   nb_filters_base=p['nb_filters_base'],
                                   conv_kernel_init=p['conv_kernel_init'],
                                   conv_l2_lambda=p['conv_l2_lambda'],
                                   prop_dropout_base=p['prop_dropout_base'],
                                   upsampling_or_transpose=p['upsampling_or_transpose'])

    # Checkpoint directory and model for this run.
    cpdir = 'checkpoints/unet2ds_rs_%s' % (ID)
    model = UNet2DSummary(cpdir=cpdir, net_builder=build_unet,
                          series_summary_func=series_summary_func)

    # Train.
    nb_steps_trn = int(np.ceil(nb_samples_per_epoch / p['batch_size']))
    history = model.fit(ds_trn, shape_trn=(p['wdw'], p['wdw']),
                        batch_size_trn=p['batch_size'],
                        nb_steps_trn=nb_steps_trn,
                        optimizer=Adam(p['learning_rate']),
                        nb_epochs=nb_epochs,
                        prop_trn=0.7, prop_val=0.25)

    # Rename the checkpoint directory with the best validation score.
    f1_mean = np.max(history['val_nf_f1_mean'])
    _cpdir = cpdir
    cpdir = cpdir.replace('rs_', 'rs_%lf_' % f1_mean)
    os.rename(_cpdir, cpdir)

    # Save the parameters and results.
    with open('%s/report.json' % cpdir, 'w') as f:
        payload = {
            'params': p,
            'results': {k: np.max(history[k]) for k in history.keys()}
        }
        json.dump(payload, f, indent=4)

    # Remove any saved weights for bad runs to save disk space.
    # Don't delete the directory itself because it might be useful to analyze.
    if f1_mean < 0.5:
        os.remove('%s/weights_val_nf_f1_mean.hdf5' % cpdir)
        return

    # Make a submission using the trained network.
    Mp = model.predict(ds_tst, '%s/weights_val_nf_f1_mean.hdf5' % cpdir)
    names = [ds.attrs['name'] for ds in ds_tst]
    json_path = '%s/submission_latest.json' % cpdir
    nf_submit(Mp, names, json_path)


if __name__ == "__main__":

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, "Define the CUDA devices."
    assert len(os.environ['CUDA_VISIBLE_DEVICES']) == 1
    single_run('all_train', 'all_test', os.environ['CUDA_VISIBLE_DEVICES'],
               nb_samples_per_epoch=5000, nb_epochs=20)
