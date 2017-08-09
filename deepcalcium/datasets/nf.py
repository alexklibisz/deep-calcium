from __future__ import division
from glob import glob
from hashlib import md5
from neurofinder import centers, shapes
from os import path, mkdir, remove
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
from deepcalcium.utils.config import DATASETS_DIR

NEUROFINDER_NAMES = sorted([
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

NAME_TO_URL = {name: 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/%s.zip' % name
               for name in NEUROFINDER_NAMES}


def nf_load_hdf5(names, datasets_dir='%s/neurons_nf' % DATASETS_DIR):
    """Downloads neurofinder datasets and pre-processes them into HDF5 datasets.
    Each HDF5 file will consist of the following datasets:
    1. series/raw:  (no. images x height x width) array of the raw TIFF values.
    2. series/mean: (height x width) mean summary of series/raw.
    3. series/max:  (height x width) max summary of series/raw.
    4. masks/raw: (no. neurons x height x width) array of the masks for every individual neuron.
    5. masks/max: (height x width) max summary of the masks/raw.

    # Arguments
        names: single name (string) or list of names from the NEUROFINDER_NAMES list.
        datasets_dir: directory where created datasets should be stored.

    # Returns
        dataset_paths: list of paths to the HDF5 dataset files that were created.
    """

    logger = logging.getLogger(funcname())

    # Convert special names 'all', 'all_train', and 'all_test'.
    if type(names) == str and names.lower() == 'all':
        dataset_names = NEUROFINDER_NAMES
    elif type(names) == str and names.lower() == 'all_train':
        dataset_names = sorted(
            [n for n in NEUROFINDER_NAMES if '.test' not in n])
    elif type(names) == str and names.lower() == 'all_test':
        dataset_names = sorted([n for n in NEUROFINDER_NAMES if '.test' in n])
    elif type(names) == str:
        dataset_names = names.split(',')
    else:
        dataset_names = names

    if not path.exists(datasets_dir):
        mkdir(datasets_dir)

    # Download datasets, unzip them, delete zip files.
    for name in dataset_names:
        url = NAME_TO_URL[name]
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
    dataset_paths = []
    for name in dataset_names:

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
            ds_ = dsf.create_dataset('series/raw', s_shape, dtype='int16')
            ds_mean = dsf.create_dataset(
                'series/mean', i_shape, dtype='float16')
            ds_max = dsf.create_dataset('series/max', i_shape, dtype='int16')
            ds_max[...] = np.zeros(i_shape)
            for idx, p in tqdm(enumerate(s_paths)):
                img = imread(p)
                ds_[idx, :, :] = img
                ds_mean[...] += (img * 1. / len(s_paths))
                ds_max[...] = np.maximum(ds_max[...], img)

            # Populate mask, max summary.
            if '.test' not in name:
                r = '%s/%s/regions/regions.json' % (datasets_dir, name)
                regions = json.load(open(r))
                i_shape = s_shape[1:]
                m_shape = (len(regions),) + i_shape
                ds_raw = dsf.create_dataset('masks/raw', m_shape, dtype='int8')
                ds_max = dsf.create_dataset('masks/max', i_shape, dtype='int8')
                for idx, r in tqdm(enumerate(regions)):
                    msk = tomask(r['coordinates'], m_shape[1:])
                    ds_raw[idx, :, :] = msk
                    ds_max[...] = np.maximum(ds_max[...], msk)
                ds_max = np.max(ds_raw, axis=0)

            dsf.close()

        dataset_paths.append(ds_path)

    return dataset_paths


def nf_mask_metrics(m, mp):
    """Computes precision, recall, inclusion, exclusion, and combined (F1) score for the given mask (m) and predicted mask (mp). Note that this does assumes single 2D masks and does not aaccount for overlapping neurons.

    # Arguments
        m: ground-truth (height x width) binary numpy mask.
        mp: predicted (height x width) binary numpy mask.

    # Returns
        p,r,i,e,f1: precision, recall, inclusion, exclusion, and F1 scores.

    """
    # Return all zeros if the predicted mask is empty.
    if np.sum(mp.round()) == 0:
        return 0., 0., 0., 0., 0.

    # Convert masks to regional format and compute their metrics.
    m = _mask_to_regional(m)
    mp = _mask_to_regional(mp)
    r, p = centers(m, mp)
    i, e = shapes(m, mp)
    f1 = 2. * (r * p) / (r + p)
    return (p, r, i, e, f1)


def nf_submit(Mp, names, json_path):
    """Given masks and dataset names, create a Neurofinder submission according to the
    format outlined on http://neurofinder.codeneuro.org/.

    # Arguments
        Mp: list of predicted (height x width) binary masks.
        names: list of dataset names.
        json_path: path where the JSON submission should be saved.

    # Returns
        Nothing

    """

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
                coords = [[int(x), int(y)] for x, y in zip(xx, yy)]
                regions.append({'coordinates': coords})

        submission.append({
            "dataset": name,
            "regions": regions
        })

    fp = open(json_path, 'w')
    json.dump(submission, fp)
    fp.close()
    logger.info('Saved submission to %s.' % json_path)


def _mask_to_regional(m):
    """Convert a 2D numpy mask to a regional many object so it can be measured
    using the neurofinder library."""
    mlbl = measure.label(m)
    coords = []
    for lbl in range(1, np.max(mlbl) + 1):
        yy, xx = np.where(mlbl == lbl)
        coords.append([[y, x] for y, x in zip(yy, xx)])
    return many(coords)
