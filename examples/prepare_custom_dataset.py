# Example script showing how to convert a custom dataset into the correct HDF5 format.
from __future__ import division
from glob import glob
from os import path, mkdir
from scipy.misc import imread
from tqdm import tqdm
import h5py
import logging
import numpy as np


def make_dataset_series_only(name, tiffglob, datasets_dir):

    dspath = '%s/%s/dataset.hdf5' % (datasets_dir, name)
    if path.exists(dspath):
        logger.info('Skipping %s.' % dspath)
        return

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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    datasets_dir = '%s/.deep-calcium-datasets' % path.expanduser('~')
    base = '/data/stjude/Data/AuditoryCortex/'

    make_dataset_series_only(
        'sj.010517',
        base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/frame*.tif',
        datasets_dir)

    make_dataset_series_only(
        'sj.010617',
        base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/frame*.tif',
        datasets_dir)

    make_dataset_series_only(
        'sj.111216',
        base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/frame*.tif',
        datasets_dir)
