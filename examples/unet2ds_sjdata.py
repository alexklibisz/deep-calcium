# Example script showing how to convert a custom dataset into the correct HDF5 format.
from __future__ import division
from glob import glob
from os import path, mkdir, listdir
from pprint import pprint
from scipy.misc import imread
from tqdm import tqdm
import argparse
import h5py
import logging
import numpy as np
import requests

import sys
sys.path.append('.')
from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary


def make_dataset_series_only(name, tiffglob, datasets_dir):

    dspath = '%s/%s/dataset.hdf5' % (datasets_dir, name)
    if path.exists(dspath):
        logger.info('%s already exists.' % dspath)
        return h5py.File(dspath, 'r')

    dsdir = '%s/%s' % (datasets_dir, name)
    if not path.exists(dsdir):
        mkdir(dsdir)

    logger.info('Populating %s.' % dspath)
    dsf = h5py.File(dspath, 'w')
    dsf.attrs['name'] = name
    s_paths = sorted(glob(tiffglob))
    i_shape = imread(s_paths[0]).shape
    s_shape = (len(s_paths),) + i_shape
    dset_sraw = dsf.create_dataset('series/raw', s_shape, dtype='int16')
    dset_smean = dsf.create_dataset('series/mean', i_shape, dtype='float16')
    dset_smax = dsf.create_dataset('series/max', i_shape, dtype='int16')

    for idx, p in tqdm(enumerate(s_paths)):
        img = imread(p)
        dset_sraw[idx, :, :] = img
        dset_smax[...] = np.maximum(img, dset_smax[...])
        dset_smean[...] += img / len(s_paths)

    dsf.flush()
    dsf.close()
    return h5py.File(dspath, 'r')


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define, make checkpoint directory.
    cpdir = 'checkpoints/sjdata'
    if not path.exists(cpdir):
        mkdir(cpdir)

    # Names and TIFF paths for datasets. The given TIFF paths should list all TIFFs
    # for this dataset. i.e. "ls tiffpath" command will list all of the TIFFs.
    datasets_dir = '%s/.deep-calcium-datasets' % path.expanduser('~')
    base = '/data/stjude/Data/AuditoryCortex/'
    dataset_args = [
        ('sj.010517', base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/frame*.tif'),
        ('sj.010617', base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/frame*.tif'),
        ('sj.111216', base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/frame*.tif')
    ]

    # Create hdf5 datasets from the raw TIFFs.
    datasets = [make_dataset_series_only(name, tiffglob, datasets_dir)
                for name, tiffglob in dataset_args]

    # Download weights - or you can do it manually.
    weights_path = '%s/weights.hdf5' % cpdir
    weights_url = 'https://www.dropbox.com/sh/tqbclt7muuvqfw4/AACqVVA8oJlZNIYvfc6x6gO2a/weights_val_nf_f1_mean.hdf5?dl=1'
    weights_download = requests.get(weights_url)
    with open(weights_path, 'wb') as weights_file:
        weights_file.write(weights_download.content)

    # Setup the model.
    model = UNet2DSummary(cpdir=cpdir)

    # save=True saves the mean summary with outlines on it.
    Mp = model.predict(datasets, weights_path, window_shape=(512, 512), save=True)

    # Returns a list of mask predictions as numpy arrays.
    pprint([mp.shape for mp in Mp])

    # Show images in checkpoint directory.
    pprint(sorted(glob('%s/*.png' % cpdir)))
