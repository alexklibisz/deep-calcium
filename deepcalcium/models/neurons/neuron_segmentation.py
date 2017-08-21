from hashlib import md5
from scipy.misc import imresize
from sklearn.cluster import KMeans
from tifffile import imread
from time import time
from tqdm import tqdm
import h5py
import logging
import numpy as np
import types
import os

from deepcalcium.models.networks import unet2d
from deepcalcium.utils.config import CHECKPOINTS_DIR
from deepcalcium.utils.runtime import funcname

def make_neurons_hdf5(img_paths, msks, hdf5_path):
    """Create an HDF5 file for neuron segmentation. The file is considered to
    be created if it contains an attribute 'img_paths' which is a comma-separated
    string of the img_paths list. This attribute is added after populating
    the images and masks.

    The HDF5 format simply contains two datasets:
    - 'imgs': an int16 array with shape (no. images, height, width) containing
        the images read from the given img_paths list.
    - 'msks': a uint8 array with shape (no. masks, height, width) containing
        the masks passed as an argument.

    # Arguments
        img_paths: list of paths to the images in the series.
        msks: numpy array of individual neuron masks with shape
            (no. masks, height width).
        hdf5_path: path where the hdf5 file is stored.

    # Returns
        hdf5_path: the path where the hdf5 file was stored.

    """

    logger = logging.getLogger(funcname())
    img_paths = sorted(img_paths)
    imgs_hash = md5(str(img_paths).encode()).hexdigest()
    msks_hash = md5(msks).hexdigest()

    # If the file already exists and has an attribute containing the img paths,
    # then it has already been populated and the path is returned.
    if os.path.exists(hdf5_path):
        fp = h5py.File(hdf5_path)
        a = dict(fp.attrs)
        fp.close()
        # print(imgs_hash, a['imgs_hash'] if 'imgs_hash' in a else '--')
        # print(msks_hash, a['msks_hash'] if 'msks_hash' in a else '--')
        if 'imgs_hash' in a and a['imgs_hash'] == imgs_hash and 'msks_hash' in a and a['msks_hash'] == msks_hash:
            logger.info('File already exists and is populated: %s' % (hdf5_path))
            return hdf5_path

    # Create the file.
    fp = h5py.File(hdf5_path, 'w')

    # Populate images.
    logger.info('Populating images')
    shape = (len(img_paths), *msks.shape[1:])
    imgs_ds = fp.create_dataset('imgs/raw', shape, dtype='int16')
    for i in tqdm(range(len(img_paths))):
        imgs_ds[i,:,:] = imread(img_paths[i])

    # Populate the masks.
    msks_ds = fp.create_dataset('msks/raw', msks.shape, dtype='int8')
    msks_ds[...] = msks

    # Add hash attributes.
    fp.attrs['imgs_hash'] = imgs_hash
    fp.attrs['msks_hash'] = msks_hash
    fp.close()

    return hdf5_path

def summary_imgs_mean(hdf5_path):
    return 0

def summary_imgs_kmeans(hdf5_path, k, shape=None):
    """Computes series s

    """

    logger = logging.getLogger(funcname())
    fp = h5py.File(hdf5_path)
    assert 'imgs/raw' in fp
    imgs = fp.get('imgs/raw')[...]
    summary_key = 'imgs/kmeans_%d_%d_%d_%s' % (k, shape[0], shape[1], md5(imgs).hexdigest())

    # Check if summary has already been computed and stored.
    if summary_key in fp:
        summary = fp.get(summary_key)[...]
        fp.close()
        return summary
    else:
        fp.close()

    # Resize the images for more efficient clustering.
    if shape is not None:
        x = np.array([imresize(img, shape) for img in imgs])
    else:
        x = imgs

    # Cluster the images which may have been resized.
    n, h, w = x.shape
    km = KMeans(k, n_jobs=1)
    km.fit(x.reshape(n, h * w))

    # Use the cluster labels to compute centroids from the full-size images.
    summary = np.zeros((k, *imgs.shape[1:]))
    for i in range(np.max(km.labels_)):
        xx = np.where(km.labels_ == i)
        summary[i] = np.mean(imgs[xx], axis=0)

    # import matplotlib.pyplot as plt
    # fig, _ = plt.subplots(4, 2)
    # for i,ax in enumerate(fig.axes):
    #     ax.imshow(summary[i], cmap='gray')
    # plt.show()

    # Store and return the summary.
    fp = h5py.File(hdf5_path, 'r+')
    summary_ds = fp.create_dataset(summary_key, summary.shape, dtype='float32')
    summary_ds[...] = summary
    fp.close()
    return summary

def summary_msks_max(hdf5_path):
    return 0

def summary_msks_max_erosion(hdf5_path):
    """Flattens the stack of neuron masks into a (height x width) combined mask.
    Eliminates overlapping and neighboring pixels that belong to different
    neurons to preserve the original number of independent neurons. The summary
    is saved in the HDF5 file with a key including the md5 hash of the masks.

    # Arguments
        hdf5_path: path to the HDF5 file where the masks are stored.
    # Returns
        summary: 2D binary segmentation mask, shape (height, width).

    """

    fp = h5py.File(hdf5_path)
    assert 'msks/raw' in fp
    msks = fp.get('msks/raw')[...]
    summary_key = 'msks/max_erosion_%s' % md5(msks).hexdigest()

    # Check if summary has already been computed and stored.
    if summary_key in fp:
        summary = fp.get(summary_key)[...]
        fp.close()
        return summary
    else:
        fp.close()

    # Coordinates of all 1s in the stack of masks.
    zyx = list(zip(*np.where(msks == 1)))

    # Mapping (y,x) -> z.
    yx_z = {(y, x): [] for z, y, x in zyx}
    for z, y, x in zyx:
        yx_z[(y, x)].append(z)

    # Remove all elements with > 1 z.
    for k in list(yx_z.keys()):
        if len(yx_z[k]) > 1:
            del yx_z[k]
    assert np.max([len(v) for v in yx_z.values()]) == 1.

    # For (y,x), take the union of its z-values with its immediate neighbors' z-values.
    # Delete the (y,x) and its neighbors if |union| > 1.
    for y, x in list(yx_z.keys()):
        nbrs = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1), (y + 1, x + 1),
                (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)] + [(y, x)]
        nbrs = [k for k in nbrs if k in yx_z]
        allz = [yx_z[k][0] for k in nbrs]
        if len(np.unique(allz)) > 1:
            for k in nbrs:
                del yx_z[k]

    # The mask consists of the remaining (y,x) keys.
    yy, xx = [y for y, x in yx_z.keys()], [x for y, x in yx_z.keys()]
    summary = np.zeros(msks.shape[1:])
    summary[yy, xx] = 1.

    # Store and return summary.
    fp = h5py.File(hdf5_path, 'r+')
    summary_ds = fp.create_dataset(summary_key, summary.shape, dtype='int8')
    summary_ds[...] = summary
    fp.close()
    return summary


def augmentation_mean(series_summary, masks_summary):
    """Applies augmentations for mean-summarized series.
    # Arguments
        series_summary: summarized series as a numpy array with shape (h, w).
        masks_summary: summarized mask as a numpy array with shape (h, w).
    # Returns
        series_summarized_batch: batch of augmented series summaries containing
            k * 8 summaries. numpy array with shape (8, h, w, k).
        masks_summarized_batch: batch of augmented mask summaries. numpy array
            with shape (8, h, w).

    """
    return

def augmentation_kmeans(series_summary, masks_summary):
    """Applies augmentation for kmeans-summarized series.
    # Arguments
        series_summary: summarized series as a numpy array with shape (h, w, k).
        masks_summary: summarized mask as a numpy array with shape (h, w).
    # Returns
        series_summarized_batch: batch of augmented series summaries containing
            k * 8 summaries. numpy array with shape (k * 8, h, w, k).
        masks_summarized_batch: batch of augmented mask summaries. numpy array
            with shape (k * 8, h, w).
    """
    return

class ValidationCallback(object):
    def __init__(self, series_summaries, masks_summaries, fit_augmentation, fit_validation_metrics):
        self.series_summaries = series_summaries
        self.masks_summaries = masks_summaries
        self.fit_augmentation = fit_augmentation
        self.fit_validation_metrics = fit_validation_metrics
        # TODO: pre-compute and store augmented summaries.

        return

    def on_epoch_end(self, batch, logs):

        # TODO: load in the trained model, make predictions on each series,
        # compute their metrics, store metrics in logs.

        return

class NeuronSegmentation(object):
    """Neuron segmentation model. The interface to this model mimics that of
    many scikit-learn models. In particular, all of the model configuration
    is done in the constructor instead of throughout the various methods (
    e.g. fit(), predict()). The intention is to allow configurability and make
    it easy to test different combinations of hyper-parameters.

    # Arguments
        TODO
    # Returns
        TODO

    """
    def __init__(self,
                 checkpoints_dir='%s/tmp' % CHECKPOINTS_DIR,
                 network_func=unet2d,
                 imgs_summary_func=lambda p: imgs_summary_kmeans(p, 8),
                 msks_summary_func=summary_msks_max_erosion,
                 fit_shape=(128, 128, 8),
                 fit_augmentation_func=augmentation_kmeans,
                 fit_iters=10000,
                 fit_batch=20,
                 fit_validation_interval=1000,
                 fit_objective='binary_crossentropy',
                 fit_optimizer='adam',
                 fit_optimizer_arguments={'lr': 0.002},
                 predict_shape=(8, 512, 512),
                 predict_augmentation_func=augmentation_kmeans,
                 predict_threshold=0.5,
                 predict_verbose=True,
                 predict_save=True,
                 random_state=np.random):

        self.checkpoints_dir = checkpoints_dir
        self.network_func = network_func
        self.imgs_summary_func = imgs_summary_func
        self.msks_summary_func = msks_summary_func
        self.fit_shape = fit_shape
        self.fit_augmentation_func = fit_augmentation_func
        self.fit_shape = fit_shape
        self.fit_iters = fit_iters
        self.fit_batch = fit_batch
        self.fit_validation_interval = fit_validation_interval
        self.fit_objective = fit_objective
        self.fit_optimizer = fit_optimizer
        self.fit_optimizer_arguments = fit_optimizer_arguments
        self.predict_shape = predict_shape
        self.predict_augmentation_func = predict_augmentation_func
        self.predict_threshold = predict_threshold
        self.predict_verbose = predict_verbose
        self.predict_save = predict_save
        self.random_state = random_state

    def fit(self, hdf5_paths_train, hdf5_paths_validate):
        """Fit the model with the given training and validation data.

        # Arguments
            hdf5_paths_train: paths to the hdf5 files used for training.
            hdf5_paths_validate: paths to the hdf5 files used for validation.

        # Returns
            TODO

        """

        logger = logging.getLogger(funcname())

        # Summarize series and masks.
        logger.info('Computing summaries for training data')
        series_summaries_trn, masks_summaries_trn = [], []
        for p in tqdm(hdf5_paths_train):
            series_summaries_trn.append(self.imgs_summary_func(p))
            masks_summaries_trn.append(self.msks_summary_func(p))

        logger.info('Computing summaries for validation data')
        series_summaries_val, masks_summaries_val = [], []
        for p in tqdm(hdf5_paths_validate):
            series_summaries_trn.append(self.imgs_summary_func(p))
            masks_summaries_trn.append(self.msks_summary_func(p))

        assert True == False

        # Setup the training and validation generator.
        batch_gen_trn = self._batch_generator(series_summaries_trn, masks_summaries_trn)
        batch_gen_val = self._batch_generator(series_summaries_val, masks_summaries_val)

        # Setup the validation callback, which makes full-image predictions.
        valcb = ValidationCallback(series_summaries_val, masks_summaries_val, self.fit_augmentation)

        # Setup remaining callbacks.
        callbacks = [ valcb ]

        # Setup the network and fit.
        model = self._get_network(self.fit_shape)
        trained = model.fit_generator()

        # Find and print the best network path based on training history.

    def predict(self, hdf5_paths):
        """Make predictions with the given data.

        # Arguments
            hdf5_paths: paths to the hdf5 files for predictions.

        # Returns
            masks_predicted: list of predicted masks corresponding to each of
                the hdf5 files. Each predicted mask is a binary numpy array with
                the same height and width as its corresponding series.

        """

        # Summarize datasets.
        series_summaries = [self._series_summary(p, self.predict_shape) for p in hdf5_paths]
        masks_summaries = [self._masks_summary(p, self.predict_shape) for p in hdf5_paths]

        # Make predictions (w/ augmentation if specified).


        return masks_predicted

    def _batch_generator(self, series_summaries, masks_summaries):
        """Internal batch generator used for training keras models."""

        pass

    def _get_network(self, shape):
        """Internal function to make and return model."""
        func_map = {'unet2d': unet2d}
        func = self.network_type
        if type(self.network_type) == str:
            func = func_map[self.network_type]
        return func(shape)
