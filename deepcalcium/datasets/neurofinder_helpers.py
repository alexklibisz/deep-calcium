from glob import glob
from zipfile import ZipFile
import h5py
import json
import logging
import numpy as np
import os
import requests

from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.config import DATASETS_DIR
from deepcalcium.models.neurons.neuron_segmentation import make_neurons_hdf5

NEUROFINDER_NAMES_ALL = [
    'neurofinder.00.00', 'neurofinder.00.00.test',
    'neurofinder.00.01', 'neurofinder.00.01.test',
    'neurofinder.00.02',
    'neurofinder.00.03',
    'neurofinder.00.04',
    'neurofinder.00.05',
    'neurofinder.00.06',
    'neurofinder.00.07',
    'neurofinder.00.08',
    'neurofinder.00.09',
    'neurofinder.00.10',
    'neurofinder.00.11',
    'neurofinder.01.00', 'neurofinder.01.00.test',
    'neurofinder.01.01', 'neurofinder.01.01.test',
    'neurofinder.02.00', 'neurofinder.02.00.test',
    'neurofinder.02.01', 'neurofinder.02.01.test',
    'neurofinder.03.00', 'neurofinder.03.00.test',
    'neurofinder.04.00', 'neurofinder.04.00.test',
    'neurofinder.04.01', 'neurofinder.04.01.test'
]
NEUROFINDER_NAMES_TRAIN = [s for s in NEUROFINDER_NAMES_ALL if '.test' not in s]
NEUROFINDER_NAMES_TEST = [s for s in NEUROFINDER_NAMES_ALL if '.test' in s]
S3URL = 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder'
NEUROFINDER_NAMES_TO_URLS = {name: '%s/%s.zip' % (S3URL, name)
                             for name in NEUROFINDER_NAMES_ALL}


def neurofinder_load_hdf5(dataset_names=','.join(NEUROFINDER_NAMES_ALL),
                          datasets_dir='%s/neurons/neurofinder' % DATASETS_DIR):
    """Downloads and populates the hdf5 files containing neurofinder datasets. A file
    ".complete" is created in the dataset directory after it has been downloaded
    and unzipped to prevent downloading and unzipping again in the future. After the
    dataset is unzipped, the images paths and masks are aggregated and passed to the
    make_neurons_hdf5 function to create the hdf5 file.

    # Arguments
        dataset_names: comma-separated string of dataset names.
        datasets_dir: directory where downloaded datasets and created hdf5
            files are stored.
    # Returns
        hdf5_paths: paths to the created hdf5 files.
    """

    logger = logging.getLogger(funcname())

    if type(dataset_names) == str:
        dataset_names = dataset_names.split(',')

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    # Helper functions.
    def download_unzip(name):
        url = NEUROFINDER_NAMES_TO_URLS[name]
        zip_name = '%s.zip' % name
        zip_path = '%s/%s' % (datasets_dir, zip_name)
        unzip_dir = '%s/%s' % (datasets_dir, name)
        complete_path = '%s/%s/.complete' % (datasets_dir, name)

        # Check if the directory exists and the complete file has been created.
        if os.path.exists(complete_path):
            logger.info('Already downloaded %s' % name)
            return

        logger.info('Downloading %s' % zip_name)
        download = requests.get(url)
        logger.info('Download complete, writing to %s' % zip_path)
        with open(zip_path, 'wb') as zip_file:
            zip_file.write(download.content)

        logger.info('Unzipping %s to %s' % (zip_name, unzip_dir))
        zip_ref = ZipFile(zip_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

        logger.info('Deleting %s' % zip_name)
        os.remove(zip_path)

        # Create the complete file.
        fp = open(complete_path, 'w')
        fp.close()

    def get_image_paths(name):
        return sorted(glob('%s/%s/images/*.tif*' % (datasets_dir, name)))

    def tomask(coords, shape):
        m = np.zeros(shape)
        yy, xx = [c[0] for c in coords], [c[1] for c in coords]
        m[yy, xx] = 1
        return m

    def get_masks(name, shape):
        fp = open('%s/%s/regions/regions.json' % (datasets_dir, name))
        regions = json.load(fp)
        fp.close()
        return np.array([tomask(r['coordinates'], shape) for r in regions])

    def get_shape(name):
        fp = open('%s/%s/info.json' % (datasets_dir, name))
        shape = tuple(json.load(fp)['dimensions'][:2])
        fp.close()
        return shape

    hdf5_paths = ['%s/%s/neurons.hdf5' % (datasets_dir, name)
                  for name in dataset_names]

    for name, hdf5_path in zip(dataset_names, hdf5_paths):
        download_unzip(name)
        image_paths = get_image_paths(name)
        shape = get_shape(name)
        masks = get_masks(name, shape)
        make_neurons_hdf5(image_paths, masks, hdf5_path)

    return hdf5_paths
