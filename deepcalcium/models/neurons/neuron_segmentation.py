import numpy as np
import types

from deepcalcium.models.networks import unet2d

def neurons_make_hdf5(tiff_paths, masks_array, hdf5_path):
    """Create an HDF5 file for neuron segmentation.

    # Arguments
        tiff_paths: list of paths to the TIFF images in the series. This assumes
            they are sorted in the correct order.

        masks_array: numpy array of individual neuron masks with shape
            (no. masks, height width).

        hdf5_path: path where the hdf5 file is stored.

    # Returns
        dataset_path: the path where the hdf5 file was stored.

    """
    return

def series_sumary_mean(hdf5_path, shape=None):
    return

def series_summary_kmeans(hdf5_path, shape=None):
    return

def masks_summary_max(hdf5_path, shape=None):
    return

def masks_summary_max_erosion(hdf5_path, shape=None):
    return

def augment_summaries(series_summary, masks_summary, flip_rotate=True, shuffle_channels=True):
    """Applies augmentation to the given summaries.

    # Arguments
        series_summary: numpy array 

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
        network_type: {'unet2d'}
            Model used for segmentation. Defaults to 'unet2d'.
            'unet2d': modified version of U-Net. See https://arxiv.org/abs/1707.06314.
        network_path: file system path to a pre-trained keras model and architecture.
        model_architectures: dictionary mapping the network_type to a function that
            instantiates the keras model. This can be useful for trying different
            hyper-parameters.
        series_summary_type: {'kmeans', 'mean'}
            Method for summarizing the image series, defaults to 'kmeans'.
            'kmeans': runs k-means clustering on the image series and uses the
                resulting centroids as k representative images for the series.
                k is determined by the fit_shape and predict_shape arguments.
                When fitting the model, the k centroids are treated as an
                image with k channels, and the channels can occur in any order.
            'mean': computes the mean summary image and uses it as a single
                channel input image.
        masks_summary_type: {'max_erosion', 'max'}
            Method for summarizing the neuron masks, defaults to 'max_erosion'.
            'max_erosion': computes the max summary of the neuron masks and
                applies erosion to each neuron to eliminate overlapping pixels
                and preserve the original number of neurons. This is motivated
                by the fact that the neurofinder metrics penalize for
                non-separated neurons.
            'max': computes the max summary of the neuron masks. This will
                leave overlapping neurons.
        fit_shape: Tuple defining the input/output shape for fitting the model.
            For series_summary 'kmeans', a 3-element tuple is expected with
            format (k, height, width). For series_summary 'mean', a 2-element
            tuple is expected with format (height, width).
        fit_iters: total number of iterations (i.e. gradient updates) for fitting.
        fit_batch: batch size for fitting.
        fit_augmentation: boolean specifying whether to apply augmentations to the
            training data.
        fit_validation_metrics: {'neurofinder', 'keras'}
            Metrics used to score validation data.
            'neurofinder': uses the neurofinder implementation of precision,
                recall, and F1 implemented in https://github.com/codeneuro/neurofinder-python
            'keras': uses the kers pixelwise metrics that are used for training.
        fit_validation_interval: number of gradient updates to compute before
            computing validation metrics. (aka number of updates in one "epoch").
        fit_objective: keras-compatible objective function or string function name.
        fit_optimizer: keras-compatible optimizer class or string optimizer name.
        fit_optimizer_arguments: dictionary of arguments passed to the optimizer.
        predict_shape: Tuple defining the input/output shape for predictions.
            Same semantics as fit_shape. If the given shape is larger than
            the series being predicted, the series will be padded to fit
            and then cropped.
        predict_augmentation: boolean specifying whether to apply test-time
            augmentation. For series_summary 'mean', this consists of averaging
            eight flips and rotations. For series_summary 'kmeans', k random
            permutations of the channels are averaged in addition to the eight
            flips and rotations.
        predict_threshold: decimal threshold used to round predictions to
            create the final binary mask that gets returned.
        predict_print_metrics: boolean specifying whether to compute and print
            metrics when making predictions. This is only done if the ground-truth
            mask is included in the HDF5 datasets.
        predict_save: boolean specifying whether to save the predictions. If
            True, the predicted neurons are outlined on the mean summary image.
            If the ground-truth neurons are included in the HDF5 dataset, they
            are also outlined on the image.
    """
    def __init__(self,
                 network_type='unet2d',
                 network_path=None,
                 model_architectures={'unet2d': unet2d},
                 series_summary_type='kmeans',
                 masks_summary_type='max_erode',
                 cached_summaries=True,
                 fit_shape=(8, 128, 128),
                 fit_iters=10000,
                 fit_batch=20,
                 fit_augmentation=True,
                 fit_validation_metrics='neurofinder',
                 fit_validation_interval=1000,
                 fit_objective='binary_crossentropy',
                 fit_optimizer='adam',
                 fit_optimizer_arguments={'lr': 0.002},
                 predict_shape=(8, 512, 512),
                 predict_augmentation=True,
                 predict_threshold=0.5,
                 predict_print_metrics=True,
                 predict_save=True,
                 random_state=np.random):

        self.network_type = network_type
        self.network_path = network_path
        self.series_summary_type = series_summary_type
        self.masks_summary_type = masks_summary_type
        self.cached_summaries = cached_summaries
        self.fit_shape = fit_shape
        self.fit_iters = fit_iters
        self.fit_batch = fit_batch
        self.fit_validation_metrics = fit_validation_metrics
        self.fit_validation_interval = fit_validation_interval
        self.fit_objective = fit_objective
        self.fit_optimizer = fit_optimizer
        self.fit_optimizer_arguments = fit_optimizer_arguments
        self.predict_shape = predict_shape
        self.predict_augmentation = predict_augmentation
        self.predict_threshold = predict_threshold
        self.predict_print_metrics = predict_print_metrics
        self.predict_save = predict_save
        self.random_state = random_state

        # Error checking.
        assert self.network_type in {'unet2d'} \
               or isinstance(self.network_type, types.FunctionType)
        assert os.path.exists(self.network_path) or self.network_path == None
        assert self.series_summary_type in {'kmeans', 'mean'}
        assert self.masks_summary_type in {'max', 'max_erode'}
        x = self.series_summary_type == 'kmeans'
        assert (x and len(self.fit_shape) == 3) or not x
        assert (x and len(self.predict_shape) == 3) or not x
        assert (x and self.fit_shape[0] == self.predict_shape[0]) or not x
        x = self.series_summary_type == 'mean'
        assert (x and len(self.fit_shape) == 2) or not x
        assert (x and len(self.predict_shape) == 2) or not x
        assert len(self.fit_shape) in {2,3}
        assert len(self.predict_shape) in {2,3}
        assert self.fit_objective in {'binary_crossentropy'}
        assert self.fit_optimizer in {'sgd', 'adam'}
        assert 0. <= self.predict_threshold <= 1.

        # Setup summary functions.
        fmap = {'kmeans': series_summary_kmeans,
                'mean': series_summary_mean,
                'max': masks_summary_max,
                'max_erosion': masks_summary_max_erosion}
        self._series_summary = fmap[self.series_summary_type]
        self._masks_summary = fmap[self.masks_summary_type]

    def fit(self, hdf5_paths_train, hdf5_paths_validate):
        """Fit the model with the given training and validation data.

        # Arguments
            hdf5_paths_train: paths to the hdf5 files used for training.
            hdf5_paths_validate: paths to the hdf5 files used for validation.

        """

        # Summarize datasets.
        series_summaries_trn = [self._series_summary(p, self.fit_shape) for p in hdf5_paths_train]
        series_summaries_val = [self._series_summary(p, self.fit_shape) for p in hdf5_paths_validate]
        masks_summaries_trn = [self._masks_summary(p, self.fit_shape) for p in hdf5_paths_train]
        masks_summaries_val = [self._masks_summary(p, self.fit_shape) for p in hdf5_paths_validate]

        # Setup the training generator.
        batch_gen_trn = self._batch_generator(series_summaries_trn, masks_summaries_trn)

        # Setup the validation callback, which makes full-image predictions.
        valcb = ValidationCallback(series_summaries_val, masks_summaries_val, self.fit_augmentation)

        # Setup remaining callbacks.
        callbacks = [ valcb, ]

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
