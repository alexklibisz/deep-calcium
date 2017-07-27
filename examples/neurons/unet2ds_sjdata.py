# Example script for working with a new dataset:
# - Converts TIFFs in a directory into HDF5 format for use with deep-calcium interface.
# - Downloads pre-trained model and weights for UNet2DS.
# - Makes predictions on the TIFFs.
# - Saves the predictions as PNGs.
# To run: python examples/unet2ds_sjdata.py
# Execution starts at starts at if __name__ == "__main__":
# Optional modifications around the lines with comment "CONFIG".
from __future__ import division
from glob import glob
from scipy.io import loadmat
from scipy.misc import imread
from tqdm import tqdm
import h5py
import logging
import numpy as np
import os
import requests

import sys
sys.path.append('.')
from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary


def make_stjude_dataset(name, tiffglob, mat_path, dataset_path):
    """Converts the St. Jude datasets into an HDF5 format compatible with UNet2DS.
    The HDF5 file contains the following datasets:
    1. series/raw:  (no. images x height x width) array of the raw TIFF values.
    2. series/mean: (height x width) mean summary of series/raw.
    3. series/max:  (height x width) max summary of series/raw.
    4. masks/raw: (no. neurons x height x width) array of the masks for every individual neuron.
    5. masks/max: (height x width) max summary of the masks/raw.
    The same data setup should be followed for new datasets.

    # Arguments
        name: a unique name for the dataset.
        tiffglob: glob-style pattern for accessing TIFF images (e.g. /path/to/images/frame*.tif)
        mat_path: path to custom matlab file that defines annotated masks for each dataset.
        dataset_path: path where the created dataset is saved.

    # Returns
        dataset_path: path to the HDF5 dataset.
    """

    if os.path.exists(dataset_path):
        logger.info('%s already exists.' % dataset_path)
        return dataset_path

    # Create/open hdf5 file that will contain the dataset.
    # Set the dataset name as an attribute on the file.
    logger.info('Creating %s.' % dataset_path)
    fp = h5py.File(dataset_path, 'w')
    fp.attrs['name'] = name

    # Populate the series from the TIFF files. The series is stored as three datasets:
    # 1. series/raw:  (time x height x width) array of the raw TIFF values.
    # 2. series/mean: (height x width) mean summary of series/raw.
    # 3. series/max:  (height x width) max summary of series/raw.
    paths = sorted(glob(tiffglob))
    t, h, w = (len(paths), ) + imread(paths[0]).shape
    ds_raw = fp.create_dataset('series/raw', (t, h, w), dtype='int16')
    ds_mean = fp.create_dataset('series/mean', (h, w), dtype='float16')
    ds_max = fp.create_dataset('series/max', (h, w), dtype='int16')

    for idx, p in tqdm(enumerate(paths)):

        # Read and gracefully handle corrupted or missing TIFFs.
        try:
            ds_raw[idx, :, :] = imread(p)
        except IOError as e:
            logger.warning('Error on file %s.' % p)
            logger.warning(str(e))
            ds_raw[idx, :, :] = np.zeros((h, w))
            pass

        ds_max[...] = np.maximum(ds_raw[idx, :, :], ds_max[...])
        ds_mean[...] += ds_raw[idx, :, :] / len(paths)

    # Populate the masks from custom matlab exports. Masks are stored as two datasets:
    # 1. masks/raw: (no. neurons x height x width) array of the masks for every individual neuron.
    # 2. masks/max: (height x width) max summary of the masks/raw.
    # The file structure was inferred via painful trial and error. Maybe writing directly to a
    # standardized format (e.g. hdf5) would be a feature.
    mat_data = loadmat(mat_path)
    mat_main_app_vars = mat_data['appStateData']['mainAppVars']
    pcx = mat_main_app_vars[0][0][0][0][0][0]
    pcy = mat_main_app_vars[0][0][0][0][0][1]
    bbox_coords = [(int(round(x)), int(round(y))) for x, y in zip(pcx, pcy)]
    bbox_radius = mat_main_app_vars[0][0][0][0][2][0][0]

    n, h, w = len(bbox_coords), h, w
    ds_raw = fp.create_dataset('masks/raw', (n, h, w), dtype='int8')
    ds_max = fp.create_dataset('masks/max', (h, w), dtype='int8')
    for idx, (x, y) in tqdm(enumerate(bbox_coords)):
        y0, y1 = max(0, y - bbox_radius), min(h, y + bbox_radius)
        x0, x1 = max(0, x - bbox_radius), min(w, x + bbox_radius)
        ds_raw[idx, y0:y1, x0:x1] = 1.
        ds_max[...] = np.maximum(ds_raw[idx, :, :], ds_max[...])
        assert np.sum(ds_raw[idx, :, :]) == (2 * bbox_radius)**2

    fp.close()
    logger.info('Done. File is %.2lf GB on disk.' %
                (os.path.getsize(dataset_path) / 1024**3))
    return dataset_path


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # CONFIG: URL to the serialized deep neural network model.
    MURL = 'https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AAAteOMVC45Ovf6g2iu10c_Ya/1499980441_model_07_0.843.hdf5?dl=1'

    # CONFIG: define the checkpoint directory where predictions will be stored.
    cpdir = 'checkpoints/sjdata'
    if not os.path.exists(cpdir):
        os.mkdir(cpdir)

    # CONFIG: define the directory where the converted datasets will be stored.
    dsdir = '%s/.deep-calcium-datasets/stjude' % os.path.expanduser('~')
    if not os.path.exists(dsdir):
        os.mkdir(dsdir)

    # CONFIG: parameters for converting each dataset. See the first item below for precise explanation.
    # These datasets are current as of 7/19/17.
    # Common directory where all datasets were stored.
    base = '/data/stjude/Data/AuditoryCortex/'
    dataset_args = [
        (
            # 1. Unique name for the dataset.
            'sj.010517',
            # 2. Path pattern for TIFF file names.
            base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/frame*.tif',
            # 3. Path to Matlab export created with the custom GUI.
            base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            # 4. Path where this dataset should be saved.
            dsdir + '/sj.neurons.010517.hdf5'
        ), (
            'sj.010617',
            base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/frame*.tif',
            base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/Exported_Matlab_Data_200.mat',
            dsdir + '/sj.neurons.010617.hdf5'
        ), (
            'sj.022616.01',
            base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/frame*.tif',
            base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.neurons.022616.01.hdf5'
        ), (
            'sj.022616.02',
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/frame*.tif',
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.neurons.022616.02.hdf5'
        ), (
            'sj.022616.03',
            base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/frame*.tif',
            base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.neurons.022616.03.hdf5'
        ), (
            'sj.100716',
            base + '100716/TSeries-10072016-1007-003/512_pruned/frame*.tif',
            base + '100716/TSeries-10072016-1007-003/512_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.neurons.100716.hdf5'
        ), (
            'sj.111216',
            base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/frame*.tif',
            base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.neurons.111216.hdf5'
        ), (
            'sj.120116',
            base + '120116/TSeries-12012016-1201-002_stabilized/512_pruned/frame*.tif',
            base + '120116/TSeries-12012016-1201-002_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            dsdir + '/sj.neurons.120116.hdf5'
        ), (
            'sj.120216',
            base + '120216/TSeries-12022016-1202-001_stabilized/512_pruned/frame*.tif',
            base + '120216/TSeries-12022016-1202-001_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            dsdir + '/sj.neurons.120216.hdf5'
        )
    ]

    # Create hdf5 datasets from the raw TIFFs.
    ds_paths = sorted([make_stjude_dataset(name, tg, rp, dsp)
                       for name, tg, rp, dsp in dataset_args])

    # Download weights - or you can do it manually.
    model_path = '%s/weights.hdf5' % cpdir
    if not os.path.exists(model_path):
        model_dnld = requests.get(MURL)
        with open(model_path, 'wb') as model_file:
            model_file.write(model_dnld.content)

    # Model setup and predictions.
    model = UNet2DSummary(cpdir=cpdir)
    Mp, names = model.predict(ds_paths, model_path,
                              window_shape=(512, 512), save=True)

    # Print name, shape, path to saved image for each dataset.
    for name, mp, path in zip(names, Mp, sorted(glob('%s/*.png' % cpdir))):
        print('%-15s %-10s %s' % (name, str(mp.shape), path))
