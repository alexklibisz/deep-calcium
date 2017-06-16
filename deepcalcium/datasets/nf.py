from __future__ import division
from glob import glob
from hashlib import md5
from neurofinder import centers, shapes
from os import path, mkdir, remove, rename
from scipy.misc import imread
from skimage import measure
from sys import getsizeof
from tqdm import tqdm
from regional import many
from zipfile import ZipFile
import h5py
import json
import logging
import numpy as np
import requests

from deepcalcium.utils.runtime import funcname

neurofinder_names = sorted([
    'neurofinder.00.00', 'neurofinder.00.01', 'neurofinder.00.02',
    'neurofinder.00.03', 'neurofinder.00.04', 'neurofinder.00.05',
    'neurofinder.00.06', 'neurofinder.00.07', 'neurofinder.00.08',
    'neurofinder.00.09', 'neurofinder.00.10', 'neurofinder.00.11',
    'neurofinder.01.00', 'neurofinder.01.01', 'neurofinder.02.00',
    'neurofinder.02.01', 'neurofinder.03.00', 'neurofinder.04.00',
    'neurofinder.04.01', 'neurofinder.00.00.test', 'neurofinder.00.01.test',
    'neurofinder.01.00.test', 'neurofinder.01.01.test', 'neurofinder.02.00.test',
    'neurofinder.02.01.test', 'neurofinder.03.00.test', 'neurofinder.04.00.test',
    'neurofinder.04.01.test'])

name_to_URL = {name: 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/%s.zip' % name
               for name in neurofinder_names}


def nf_load_hdf5(names, datasets_dir='%s/.deep-calcium-datasets' % path.expanduser('~')):
    '''Downloads neurofinder datasets and pre-processes them into hdf5 files.'''

    logger = logging.getLogger(funcname())

    # Convert special names 'all', 'all_train', and 'all_test'.
    if type(names) == str and names.lower() == 'all':
        names = neurofinder_names
    elif type(names) == str and names.lower() == 'all_train':
        _ = [n for n in neurofinder_names if '.test' not in n]
        names = sorted(_)
    elif type(names) == str and names.lower() == 'all_test':
        _ = [n for n in neurofinder_names if '.test' in n]
        names = sorted(_)
    elif type(names) == str:
        names = names.split(',')
    assert type(names) == list

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

        logger.info('Downloading %s.' % zip_name)

        download = requests.get(url)
        logger.info('Download complete, writing to %s.' % zip_path)
        with open(zip_path, 'wb') as zip_file:
            zip_file.write(download.content)

        logger.info('Unzipping %s to %s.' % (zip_name, unzip_path))
        zip_ref = ZipFile(zip_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

        logger.info('Deleting %s.' % zip_name)
        remove(zip_path)

    def tomask(coords, shape):
        yy, xx = [c[0] for c in coords], [c[1] for c in coords]
        m = np.zeros(shape)
        m[yy, xx] = 1
        return m

    # Read the sequences and masks and store them as hdf5 files. Return the hdf5 objects.
    datasets = []
    S, M = [], []
    for name in names:

        logger.info('Preparing hdf5 files for %s.' % name)
        ds_path = '%s/%s/dataset.hdf5' % (datasets_dir, name)
        if not path.exists(ds_path):
            logger.info('Populating %s.' % ds_path)
            dsf = h5py.File(ds_path, 'w')
            dsf.attrs['name'] = name
            # Populate series, mean summary, max summary in such a way
            # that shouldn't eat up all memory.
            s_dir = '%s/%s/images' % (datasets_dir, name)
            s_paths = sorted(glob('%s/*.tiff' % (s_dir)))
            i_shape = imread(s_paths[0]).shape
            s_shape = (len(s_paths),) + i_shape
            dset_s = dsf.create_dataset('series/raw', s_shape, dtype='int16')
            dset_smean = dsf.create_dataset('series/mean', i_shape, dtype='float16')
            dset_smax = dsf.create_dataset('series/max', i_shape, dtype='int16')
            dset_smax[...] = np.zeros(i_shape)
            for idx, p in tqdm(enumerate(s_paths)):
                img = imread(p)
                dset_s[idx, :, :] = img
                dset_smean[...] += (img * 1. / len(s_paths))
                dset_smax[...] = np.maximum(dset_smax[...], img)

            # Populate mask, max summary.
            if '.test' not in name:
                r = '%s/%s/regions/regions.json' % (datasets_dir, name)
                regions = json.load(open(r))
                i_shape = s_shape[1:]
                m_shape = (len(regions),) + i_shape
                dset_m = dsf.create_dataset('masks/raw', m_shape, dtype='int8')
                dset_mmax = dsf.create_dataset('masks/max', i_shape, dtype='int8')
                for idx, r in tqdm(enumerate(regions)):
                    msk = tomask(r['coordinates'], m_shape[1:])
                    dset_m[idx, :, :] = msk
                    dset_mmax[...] = np.maximum(dset_mmax[...], msk)
                dset_mmax = np.max(dset_m, axis=0)

            dsf.flush()
            dsf.close()

        dsf = h5py.File(ds_path, 'r')
        datasets.append(dsf)

    return datasets


def nf_mask_metrics(m, mp):
    '''Computes precision, recall, inclusion, exclusion, and combined (F1) score for the given mask (m) and predicted mask (mp).
    Note that this does assumes single 2D masks and does not aaccount for overlapping neurons.'''

    logger = logging.getLogger(funcname())

    def mask_to_regional(m):
        '''Convert a mask to a regional many object so it can be measured
        using the neurofinder library.'''
        mlbl = measure.label(m)
        coords = []
        for lbl in range(1, np.max(mlbl) + 1):
            yy, xx = np.where(mlbl == lbl)
            coords.append([[y, x] for y, x in zip(yy, xx)])
        return many(coords)

    # Things can get buggy if the predicted mask is empty,
    # so just return all zeros.
    if np.sum(mp.round()) == 0:
        return 0., 0., 0., 0., 0.

    m_reg = mask_to_regional(m)
    mp_reg = mask_to_regional(mp)
    r, p = centers(m_reg, mp_reg)
    i, e = shapes(m_reg, mp_reg)
    c = 2. * (r * p) / (r + p)
    return (p, r, i, e, c)


def nf_submit(Mp, names, json_path):

    logger = logging.getLogger(funcname())

    submission = []
    for mp, name in zip(Mp, names):
        # Label each of the distinct components and break the labeled
        # image into its regions.
        if name.startswith('neurofinder.'):
            name = '.'.join(name.split('.')[1:])

        mp_labeled = measure.label(mp)
        if np.max(mp_labeled) == 0:
            regions = [{'coordinates': [[[0, 0]]]}]
        else:
            regions = []
            for lbl in range(1, np.max(mp_labeled)):
                xx, yy = np.where(mp_labeled == lbl)
                coords = [[x, y] for x, y in zip(xx, yy)]
                regions.append({'coordinates': coords})

        submission.append({
            "dataset": name,
            "regions": regions
        })

    logger.info('md5: %s' % md5(str(submission)).hexdigest())
    logger.info('size (bytes): %d' % getsizeof(submission))

    fp = open(json_path, 'w')
    json.dump(submission, fp)
    fp.close()
    logger.info('Saved submission to %s.' % json_path)
