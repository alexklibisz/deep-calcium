# Example script for calcium trace segmentation using a labeled dataset.
# - Converts datasets from matlab to HDF5 format.
# - Extracts a calcium trace from all ROIS across multiple imaging series.
# - Stores the calcium trace and corresponding spike labels as arrays in HDF5.
# - Trains and predicts on new data.
# Execution starts at starts at if __name__ == "__main__":
# Optional modifications around the lines with comment "CONFIG".
from __future__ import division
from glob import glob
from scipy.io import loadmat
from scipy.misc import imread
from skimage.filters import gaussian
from tqdm import tqdm
import argparse
import h5py
import logging
import numpy as np
import os
import requests

import sys
sys.path.append('.')
from deepcalcium.utils.runtime import funcname
from deepcalcium.models.spikes.trace_segmentation import TraceSegmentation


def sj_ROI_trace(tiff_paths, exist_mask, h, w, cyy, cxx, radius):
    """Given an image, compute the mean trace of ROIs within bounding boxes.

    # Arguments
        tiff_paths: list of paths to tiff files containg ROIs.
        exist_mask: binary array with an element corresponding to each tiff
            path. If 1, the tiff exists, if 0 the tiff doesn't exist. This
            is a side-effect of having removed some unstable/blurred tiffs.
            The missing tiffs are mean-interpolated.
        h: height in pixels of all tiff files.
        w: width in pixels of all tiff files.
        cyy: list of y-coordinates marking the center of each ROI.
        cxx: list of x-coordinates marking the center of each ROI.
        radius: ROI radius.

    # Returns
        traces: a numpy array with shape (no. ROIs x no. tiff images) i.e.
            each row is the trace for a single ROI across all images.
    """

    traces = np.zeros((len(cyy), len(tiff_paths)), dtype=np.float16)
    volume = np.zeros((len(tiff_paths), h, w))

    # Read full images once.
    for i in tqdm(range(len(tiff_paths))):
        try:
            volume[i] = imread(tiff_paths[i]) if exist_mask[i] else 0.
            volume[i] = gaussian(volume[i], preserve_range=True)
        except IOError as e:
            print(str(e))
            exist_mask[i] = 0.
            pass

    # Mean interpolation.
    mean = np.mean(volume[np.where(exist_mask == 1.)], axis=0)
    volume[np.where(exist_mask == 0.)] = mean

    for i, (cy, cx) in tqdm(enumerate(zip(cyy, cxx))):

        # Get window from volume.
        y0, x0 = max(0, cy - radius), max(0, cx - radius)
        y1, x1 = min(h - 1, cy + radius), min(w - 1, cx + radius)
        wdw = volume[:, y0:y1, x0:x1]

        # Simple segmentation based on standard deviation.
        stdv = np.std(wdw, axis=0)
        mask = 1. * (stdv > np.median(stdv))

        # Mean trace from masked pixels.
        traces[i] = np.sum(wdw * mask, axis=(1, 2)) / np.sum(mask)

    return traces


def make_stjude_dataset(name, tiff_glob, n_to_path, mat_path, dataset_path, trace_func=sj_ROI_trace):
    """Converts the St. Jude datasets from a custom matlab format into an HDF5
        format with the following datasets:

        traces: matrix with shape (no. ROIs x no. frames) containing the
            real-valued trace extracted at each neuron.
        spikes: matrix with shape (no. ROIs x no. frames) containing the
            binary label (spike vs. no spike) at each neuron.

    # Arguments
        name: a unique name for the dataset.
        tiff_glob: glob-style pattern for accessing TIFF images (e.g. /path/to/images/frame*.tif)
        n_to_path: function that takes a number and returns the corresponding tiff path.
        mat_path: path to custom matlab file that defines annotated masks for each dataset.
        dataset_path: path where the created dataset is saved.

    # Returns
        dataset_path: path to the HDF5 dataset with structure described above.
    """
    logger = logging.getLogger(funcname())
    if os.path.exists(dataset_path):
        logger.info('%s already exists.' % dataset_path)
        return dataset_path

    # Create/open hdf5 file that will contain the dataset.
    # Set the dataset name as an attribute on the file.
    logger.info('Creating %s.' % dataset_path)
    fp = h5py.File(dataset_path, 'w')
    fp.attrs['name'] = name

    # TODO: add the sampling rate as an attribute.
    fp.attrs['sample_rate'] = -1

    # Load file and extract application state.
    mat_data = loadmat(mat_path)
    app_vars = mat_data['appStateData']['mainAppVars']

    # Uniform bounding box radius used when labeling.
    radius = app_vars[0][0][0][0][2][0][0]

    # ROI centers.
    roi_to_pcx = app_vars[0][0][0][0][0][0].astype(np.uint16)
    roi_to_pcy = app_vars[0][0][0][0][0][1].astype(np.uint16)

    # Frame indexes at which each neuron spiked. These are extracted once manually
    # and once automatically with corrections.
    roi_to_spks_aut = mat_data['appStateData']['peak_inds_list_auto'][0][0][0]
    roi_to_spks_man = mat_data['appStateData']['peak_inds_list_manual'][0][0][0]

    # Merge the manual and automatic spike indexes.
    def mrg(a, b): return sorted(list(np.union1d(a.ravel(), b.ravel())))
    roi_to_spks = [mrg(a, m)for a, m in zip(roi_to_spks_aut, roi_to_spks_man)]

    assert len(roi_to_pcy) == len(roi_to_pcx) == len(roi_to_spks)

    # Populate the trace and spike vectors.
    nb_rois = len(roi_to_spks)
    nb_imgs = max([max(l if len(l) else [-1]) for l in roi_to_spks])
    paths = [n_to_path(n) for n in range(1, nb_imgs + 1)]
    exist = np.array([int(os.path.exists(p)) for p in paths])
    h, w = imread([paths[i] for i in range(nb_imgs) if exist[i]][0]).shape
    traces = fp.create_dataset('traces', (nb_rois, nb_imgs), dtype='float16')
    spikes = fp.create_dataset('spikes', (nb_rois, nb_imgs), dtype='int8')
    traces[...] = trace_func(paths, exist, h, w, roi_to_pcy, roi_to_pcx, radius)

    # Subtract one to account for matlab indexing.
    for i in range(nb_rois):
        if len(roi_to_spks[i]):
            spikes[i, np.array(roi_to_spks[i]) - 1.] = 1.

    return dataset_path


def preprocess(dataset_name, cpdir, dsdir):
    """Defines the arguments to convert one or more St. Jude formatted
    datasets into HDF5 format and calls make_stjude_dataset to convert them.

    # Returns
        dataset_paths: paths to the converted datasets.
    """

    logger = logging.getLogger(funcname())
    base = '/data/stjude/Data/AuditoryCortex/'
    dataset_args = [
        (
            'sj.010517',
            base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/frame*.tif',
            lambda n: base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/frame%05d.tif' % n,
            base + '010517/TSeries-01052017-0105-008_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            dsdir + '/sj.spikes.010517.hdf5'
        ),
        (
            'sj.010617',
            base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/frame*.tif',
            lambda n: base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/frame%05d.tif' % n,
            base + '010617/TSeries-01062017-0106-002_stabilized/512_pruned/Exported_Matlab_Data_200.mat',
            dsdir + '/sj.spikes.010617.hdf5'
        ),
        # (
        #     'sj.022616.01',
        #     base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/frame*.tif',
        #     lambda n: base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/frame%05d.tif' % n,
        #     base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/Exported_Matlab_Data.mat',
        #     dsdir + '/sj.spikes.022616.01.hdf5'
        # ),
        (
            'sj.022616.02',
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/frame*.tif',
            lambda n: base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/frame%05d.tif' % n,
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.022616.02.hdf5'
        ),
        # (
        #     'sj.022616.03',
        #     base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/frame*.tif',
        #     lambda n: base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/frame%05d.tif' % n,
        #     base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/Exported_Matlab_Data.mat',
        #     dsdir + '/sj.spikes.022616.03.hdf5'
        # ),
        (
            'sj.100716',
            '100716/TSeries-10072016-1007-003/512_pruned/frame*.tif',
            lambda n: base + '100716/TSeries-10072016-1007-003/512_pruned/frame%05d.tif' % n,
            base + '100716/TSeries-10072016-1007-003/512_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.100716.hdf5'
        ),
        (
            'sj.111216',
            '111216/TSeries-11122016-1112-003_stabilized/512_pruned/frame*.tif',
            lambda n: base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/frame%05d.tif' % n,
            base + '111216/TSeries-11122016-1112-003_stabilized/512_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.111216.hdf5'
        ),
        (
            'sj.120116',
            '120116/TSeries-12012016-1201-002_stabilized/512_pruned/frame*.tif',
            lambda n: base + '120116/TSeries-12012016-1201-002_stabilized/512_pruned/frame%05d.tif' % n,
            base + '120116/TSeries-12012016-1201-002_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            dsdir + '/sj.spikes.120116.hdf5'
        ),
        (
            'sj.120216',
            '120216/TSeries-12022016-1202-001_stabilized/512_pruned/frame*.tif',
            lambda n: base + '120216/TSeries-12022016-1202-001_stabilized/512_pruned/frame%05d.tif' % n,
            base + '120216/TSeries-12022016-1202-001_stabilized/512_pruned/Exported_Matlab_Data_200ms.mat',
            dsdir + '/sj.spikes.120216.hdf5'
        )
    ]

    if dataset_name != 'all':
        dataset_args = [dsa for dsa in dataset_args if dsa[0] == dataset_name]

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # Convert and plot each dataset.
    dataset_paths = [make_stjude_dataset(*args) for args in dataset_args]
    for path in dataset_paths:
        fp = h5py.File(path)
        traces = fp.get('traces')[:10]
        spikes = fp.get('spikes')[:10]
        fig, axes = plt.subplots(6, 1, figsize=(20, 20))
        for i in range(len(axes)):
            t = traces[i, 0:800]
            s = spikes[i, 0:800]
            axes[i].plot(t, 'k')
            axes[i].plot(t * (s * 2 - 1), 'ro')
            axes[i].set_ylim(0,  1.2 * np.max(t))
        plt.suptitle('%s\n%d ROIs, %d images' %
                     (path, fp.get('traces').shape[0], fp.get('traces').shape[1]))
        plot_path = '%s/data-%s.png' % (cpdir, path.split('/')[-1])
        plt.savefig(plot_path, dpi=90)
        plt.close()
        logger.info('Saved dataset samples to %s' % plot_path)

    return dataset_paths


def training(dataset_name, model_path, cpdir, dsdir):
    np.random.seed(int(os.getpid()))
    dataset_paths = preprocess(dataset_name, cpdir, dsdir)
    model = TraceSegmentation(cpdir=cpdir)
    return model.fit(
        dataset_paths,
        model_path=model_path,
        val_type='random_split',
        prop_trn=0.5,
        prop_val=0.5
    )


def evaluation(dataset_name, model_path, cpdir, dsdir):
    pass


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    DSDIR = '%s/.deep-calcium-datasets/stjude' % os.path.expanduser('~')
    CPDIR = 'checkpoints/traceseg'

    if not os.path.exists(DSDIR):
        os.mkdir(DSDIR)

    if not os.path.exists(CPDIR):
        os.mkdir(CPDIR)

    ap = argparse.ArgumentParser(description='CLI for trace segmentations.')
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
    sp_eva = sp.add_parser('evaluate', help='CLI for training.')
    sp_eva.set_defaults(which='evaluate')
    sp_eva.add_argument('dataset', help='dataset name', default='all')
    sp_eva.add_argument(
        '-m', '--model', help='path to model', required=True)
    sp_eva.add_argument('-c', '--cpdir', help='checkpoint directory',
                        default=CPDIR)
    sp_eva.add_argument('-d', '--dsdir', help='datasets directory',
                        default=DSDIR)

    # Parse and run appropriate function.
    args = vars(ap.parse_args())

    if args['which'] == 'train':
        training(args['dataset'], args['model'], args['cpdir'], args['dsdir'])

    if args['which'] == 'evaluate':
        evaluation(args['dataset'], args['model'], args['cpdir'], args['dsdir'])
