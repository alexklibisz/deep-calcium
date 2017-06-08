from glob import glob
from os import path, mkdir, remove
from scipy.misc import imread
from urllib2 import urlopen
from zipfile import ZipFile
import h5py
import json
import logging
import numpy as np

from deepcalcium.utils.runtime import funcname

neurofinder_names = ['neurofinder.00.00', 'neurofinder.00.01', 'neurofinder.00.02',
                     'neurofinder.00.03', 'neurofinder.00.04', 'neurofinder.00.05',
                     'neurofinder.00.06', 'neurofinder.00.07', 'neurofinder.00.08',
                     'neurofinder.00.09', 'neurofinder.00.10', 'neurofinder.00.11',
                     'neurofinder.01.00', 'neurofinder.01.01', 'neurofinder.02.00',
                     'neurofinder.02.01', 'neurofinder.03.00', 'neurofinder.04.00',
                     'neurofinder.04.01', 'neurofinder.00.00.test', 'neurofinder.00.01.test',
                     'neurofinder.01.00.test', 'neurofinder.01.01.test', 'neurofinder.02.00.test',
                     'neurofinder.02.01.test', 'neurofinder.03.00.test', 'neurofinder.04.00.test',
                     'neurofinder.04.01.test']

name_to_URL = {name: 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/%s.zip' % name
               for name in neurofinder_names}


def load_neurofinder(names, datasets_dir='/home/kzh/.deep-calcium-datasets'):
    '''Downloads neurofinder datasets and pre-processes them into hdf5 files.'''

    logger = logging.getLogger(funcname())

    # Convert special names 'all', 'all_train', and 'all_test'.
    if names.lower() == 'all':
        names = neurofinder_names
    elif names.lower() == 'all_train':
        _ = [n for n in neurofinder_names if '.test' not in n]
        names = sorted(_)
    elif names.lower() == 'all_test':
        _ = [n for n in neurofinder_names if '.test' in n]
        names = sorted(_)
    else:
        names = names.split(',')

    if not path.exists(datasets_dir):
        mkdir(datasets_dir)

    # Download datasets, unzip them, delete zip files.
    for name in names:
        url = name_to_URL[name]
        zip_name = '%s.zip' % name
        zip_path = '%s/%s' % (datasets_dir, zip_name)
        unzip_name = zip_name[:-4]
        unzip_path = zip_path[:-4]

        if path.exists(unzip_path):
            logger.info('%s already downloaded.' % unzip_name)
            continue

        logger.info('Downloading %s to %s.' % (zip_name, zip_path))
        urlopen(url, zip_path)

        logger.info('Unzipping %s to %s.' % (zip_name, unzip_path))
        zip_ref = ZipFile(zip_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

        logger.info('Deleting %s.' % zip_name)
        remove(zip_path)

    NP_DT_S, NP_DT_M = np.float32, np.uint8
    HDF5_DT_S, HDF5_DT_M = 'int8', 'int8'
    MAX_VAL_S = 2**16

    def tomask(coords, shape):
        yy, xx = [c[0] for c in coords], [c[1] for c in coords]
        m = np.zeros(shape)
        m[yy, xx] = 1
        return m

    # Read the sequences and masks and store them as hdf5 files. Return the hdf5 objects.
    S, M = [], []
    for name in names:

        logger.info('Preparing hdf5 files for %s.' % name)

        # Create and populate the hdf5 sequence.
        path_s = '%s/%s/sequence.hdf5' % (datasets_dir, name)
        if not path.exists(path_s):
            logger.info('Populating %s.' % path_s)
            sf = h5py.File(path_s, 'w')
            sf.attrs['name'] = name
            dir_s_i = '%s/%s/images' % (datasets_dir, name)
            paths_s_i = sorted(glob('%s/*.tiff' % (dir_s_i)))
            s = np.array([imread(p) * 255. / MAX_VAL_S for p in paths_s_i], dtype=NP_DT_S)
            dset_s = sf.create_dataset('s', s.shape, dtype=HDF5_DT_S)
            dset_s[...] = s
            dset_mean = sf.create_dataset('summary_mean', s.shape[1:], dtype=NP_DT_S)
            dset_mean[...] = np.mean(s, axis=0)
            dset_max = sf.create_dataset('summary_max', s.shape[1:], dtype=NP_DT_S)
            dset_max[...] = np.max(s, axis=0)
            sf.flush()
            sf.close()

        # Store the read-only sequence.
        S.append(h5py.File(path_s, 'r'))

        # Create the hdf5 mask either way for consistency.
        path_m = '%s/%s/mask.hdf5' % (datasets_dir, name)
        if not path.exists(path_m):
            mf = h5py.File(path_m, 'w')
            mf.close()

            # Populate the mask for training datasets.
            if '.test' not in name:
                logger.info('Populating %s.' % path_m)
                mf = h5py.File(path_m, 'w')
                mf.attrs['name'] = name
                r = '%s/%s/regions/regions.json' % (datasets_dir, name)
                regions = json.load(open(r))
                m_shape = S[-1].get('s').shape[1:]
                m = [tomask(r['coordinates'], m_shape) for r in regions]
                m = np.array(m, dtype=NP_DT_M)
                dset_m = mf.create_dataset('m', m.shape, dtype=HDF5_DT_M)
                dset_m[...] = m
                mf.flush()
                mf.close()

        # Store the read-only mask.
        M.append(h5py.File(path_m, 'r'))

    return S, M
