from glob import glob
from os import path, mkdir, remove
from scipy.misc import imread
from urllib.request import urlretrieve
from zipfile import ZipFile
import h5py
import json
import logging
import numpy as np

from deepcalcium.utils.runtime import funcname

alias_to_URL = {
    '00.00': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.00.zip',
    '00.01': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.01.zip',
    '00.02': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.02.zip',
    '00.03': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.03.zip',
    '00.04': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.04.zip',
    '00.05': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.05.zip',
    '00.06': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.06.zip',
    '00.07': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.07.zip',
    '00.08': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.08.zip',
    '00.09': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.09.zip',
    '00.10': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.10.zip',
    '00.11': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.11.zip',
    '01.00': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.01.00.zip',
    '01.01': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.01.01.zip',
    '02.00': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.02.00.zip',
    '02.01': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.02.01.zip',
    '03.00': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.03.00.zip',
    '04.00': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.04.00.zip',
    '04.01': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.04.01.zip',
    '00.00.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.00.test.zip',
    '00.01.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.01.test.zip',
    '01.00.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.01.00.test.zip',
    '01.01.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.01.01.test.zip',
    '02.00.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.02.00.test.zip',
    '02.01.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.02.01.test.zip',
    '03.00.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.03.00.test.zip',
    '04.00.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.04.00.test.zip',
    '04.01.test': 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.04.01.test.zip'
}


def load_neurofinder(dataset_aliases, datasets_dir='/home/kzh/.deep-calcium-datasets'):

    logger = logging.getLogger(funcname())

    # Convert special aliases 'all', 'all_train', and 'all_test'.
    if dataset_aliases.lower() == 'all':
        dataset_aliases = sorted(list(alias_to_URL.keys()))
    elif dataset_aliases.lower() == 'all_train':
        _ = [a for a in list(alias_to_URL.keys()) if '.test' not in a]
        dataset_aliases = sorted(_)
    elif dataset_aliases.lower() == 'all_test':
        _ = [a for a in list(alias_to_URL.keys()) if '.test' in a]
        dataset_aliases = sorted(_)
    else:
        dataset_aliases = dataset_aliases.split(',')

    if not path.exists(datasets_dir):
        mkdir(datasets_dir)

    # Download datasets, unzip them, delete zip files.
    for alias in dataset_aliases:
        url = alias_to_URL[alias]
        zip_name = url.split('/')[-1]
        zip_path = '%s/%s' % (datasets_dir, zip_name)
        unzip_name = zip_name[:-4]
        unzip_path = zip_path[:-4]

        if path.exists(unzip_path):
            logger.info('%s already downloaded.' % unzip_name)
            continue

        logger.info('Downloading %s to %s.' % (zip_name, zip_path))
        urlretrieve(url, zip_path)

        logger.info('Unzipping %s to %s.' % (zip_name, unzip_path))
        zip_ref = ZipFile(zip_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

        logger.info('Deleting %s.' % zip_name)
        remove(zip_path)

    NP_DT_S, NP_DT_M = np.float32, np.uint8
    HDF5_DT_S, HDF5_DT_M = 'float', 'i8'
    MAX_VAL_S = 2**16

    def tomask(coords, shape):
        yy, xx = [c[0] for c in coords], [c[1] for c in coords]
        m = np.zeros(shape)
        m[yy, xx] = 1
        return m

    # Read the sequences and masks and store them as hdf5 files. Return the hdf5 objects.
    S, M = [], []
    for alias in dataset_aliases:

        logger.info('Preparing hdf5 files for %s.' % alias)

        # Create and populate the hdf5 sequence.
        path_s = '%s/neurofinder.%s/sequence.hdf5' % (datasets_dir, alias)
        if not path.exists(path_s):
            sf = h5py.File(path_s, 'w')
            dir_s_i = '%s/neurofinder.%s/images' % (datasets_dir, alias)
            paths_s_i = sorted(glob('%s/*.tiff' % (dir_s_i)))
            s = np.array([imread(p) * 1. / MAX_VAL_S for p in paths_s_i], dtype=NP_DT_S)
            dset_s = sf.create_dataset('s', s.shape, HDF5_DT_S)
            dset_s[...] = s
            dset_mean = sf.create_dataset('summary_mean', s.shape[1:], dtype=NP_DT_S)
            dset_mean[...] = np.mean(s, axis=0)
            dset_max = sf.create_dataset('summary_max', s.shape[1:], dtype=NP_DT_S)
            dset_max[...] = np.max(s, axis=0)
            sf.flush()
            sf.close()

        # Store the read-only sequence.
        S.append(h5py.File(path_s, 'r'))

        # No mask for test sequences.
        if '.test' in alias:
            M.append(None)
            continue

        # Create and populate the hdf5 mask.
        path_m = '%s/neurofinder.%s/mask.hdf5' % (datasets_dir, alias)
        if not path.exists(path_m):
            mf = h5py.File(path_m, 'w')
            r = '%s/neurofinder.%s/regions/regions.json' % (datasets_dir, alias)
            regions = json.load(open(r))
            m_shape = S[-1].get('s').shape[1:]
            m = [tomask(r['coordinates'], m_shape) for r in regions]
            m = np.array(m, dtype=NP_DT_M)
            dset_m = mf.create_dataset('m', m.shape, HDF5_DT_M)
            dset_m[...] = m
            mf.flush()
            mf.close()

        # Store the read-only mask.
        M.append(h5py.File(path_m, 'r'))

    return S, M
