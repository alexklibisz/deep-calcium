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

import sys
sys.path.append('.')
from deepcalcium.utils.runtime import funcname
from deepcalcium.models.spikes.utils import plot_traces_spikes


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

    # Volume to populate.
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

    # Traces to populate and helpers for segmentation and interpolation.
    traces = np.zeros((len(cyy), len(tiff_paths)), dtype=np.float32)
    zzok, = np.where(exist_mask == 1.)
    rng = np.random.RandomState(sum(exist_mask))

    for i, (cy, cx) in tqdm(enumerate(zip(cyy, cxx))):

        # Get window from volume.
        y0, x0 = max(0, cy - radius), max(0, cx - radius)
        y1, x1 = min(h - 1, cy + radius), min(w - 1, cx + radius)
        wdw = volume[:, y0:y1, x0:x1]

        # Segmentation based on standard deviation of non-missing frames.
        stdv = np.std(wdw[zzok], axis=0)
        mask = 1. * (stdv > np.median(stdv))

        # Mean trace from masked pixels.
        traces[i] = np.sum(wdw * mask, axis=(1, 2)) / np.sum(mask)

        # Interpolate trace at missing images by computing a straight line
        # between the surrounding trace values and adding scaled unit Gaussian
        # noise. Code below follows y = mx+b notation.
        gaps = [(a, b) for a, b in zip(zzok[:-1], zzok[1:]) if b - a > 1]
        for x0, x1 in gaps:
            y0, y1 = traces[i, x0], traces[i, x1]
            m = (y1 - y0) / (x1 - x0)
            noise = rng.normal(0, max(1, 0.1 * abs(y1 - y0)), x1 - x0)
            traces[i, x0:x1] = np.arange(x1 - x0) * m + y0 + noise

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
        trace_func: function used to extract traces from a set of TIFFs.

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

    # Load file and extract application state.
    mat_data = loadmat(mat_path)
    app_vars = mat_data['appStateData']['mainAppVars']

    # Sample rate (FPS) computed as 1 / time delta bw first two samples.
    xscale = mat_data['appStateData']['xscale'][0][0][0]
    fp.attrs['sample_rate'] = 1. / (xscale[1] - xscale[0])

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
    nb_imgs = len(xscale)
    paths = [n_to_path(n) for n in range(1, nb_imgs + 1)]
    exist = np.array([int(os.path.exists(p)) for p in paths])
    h, w = imread([paths[i] for i in range(nb_imgs) if exist[i]][0]).shape
    traces = fp.create_dataset('traces', (nb_rois, nb_imgs), dtype='float64')
    spikes = fp.create_dataset('spikes', (nb_rois, nb_imgs), dtype='int8')
    traces[...] = trace_func(paths, exist, h, w, roi_to_pcy, roi_to_pcx, radius)

    # Subtract one to account for matlab indexing.
    for i in range(nb_rois):
        if len(roi_to_spks[i]):
            spikes[i, np.array(roi_to_spks[i]) - 1.] = 1.

    return dataset_path


def preprocess(dataset_name, cpdir, dsdir, plots=False):
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
        (
            'sj.022616.01',
            base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/frame*.tif',
            lambda n: base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/frame%05d.tif' % n,
            base + '022616/TSeries-02262016-0226-001_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.022616.01.hdf5'
        ),
        (
            'sj.022616.02',
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/frame*.tif',
            lambda n: base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/frame%05d.tif' % n,
            base + '022616/TSeries-02262016-0226-002_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.022616.02.hdf5'
        ),
        (
            'sj.022616.03',
            base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/frame*.tif',
            lambda n: base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/frame%05d.tif' % n,
            base + '022616/TSeries-02262016-0226-003_stabilized/400_pruned/Exported_Matlab_Data.mat',
            dsdir + '/sj.spikes.022616.03.hdf5'
        ),
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
    dataset_paths = []
    for args in dataset_args:
        path = make_stjude_dataset(*args)
        dataset_paths.append(path)
        if not plots:
            continue
        fp = h5py.File(path)
        t, s = fp.get('traces')[:10], fp.get('spikes')[:10]
        save_path = '%s/data-%s.png' % (cpdir, path.split('/')[-1])
        title = '%s\n%d ROIs, %d images' % (path, t.shape[0], t.shape[1])
        plot_traces_spikes(traces=t, spikes_true=s, dpi=250,
                           title=title, save_path=save_path)
        logger.info('Saved dataset samples to %s' % save_path)

    return dataset_paths


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    DSDIR = '%s/.deep-calcium-datasets/stjude' % os.path.expanduser('~')
    CPDIR = 'checkpoints/sjspikes_unet1d'

    if not os.path.exists(DSDIR):
        os.mkdir(DSDIR)

    if not os.path.exists(CPDIR):
        os.mkdir(CPDIR)

    ap = argparse.ArgumentParser(description='CLI for preprocessing.')
    ap.add_argument('dataset', help='dataset name', default='all')
    ap.add_argument('-c', '--cpdir', help='checkpoint directory', default=CPDIR)
    ap.add_argument('-d', '--dsdir', help='datasets directory', default=DSDIR)
    args = vars(ap.parse_args())

    preprocess(args['dataset'], args['cpdir'], args['dsdir'], plots=False)
