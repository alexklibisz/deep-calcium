import numpy as np

def make_neuron_dataset(tiff_paths, masks_array):
    return

class NeuronSegmentation(object):
    """Neuron segmentation.

    # Arguments
        model: {'unet2d'}
            Model used for segmentation. Defaults to 'unet2d'.
            'unet2d': modified version of U-Net. See https://arxiv.org/abs/1707.06314.

        model_path: file system path to a pre-trained keras model and architecture.

        series_summary: {'kmeans', 'mean'}
            Method for summarizing the image series, defaults to 'kmeans'.
            'kmeans': runs k-means clustering on the image series and uses the
                resulting centroids as k representative images for the series.
                k is determined by the fit_shape and predict_shape arguments.
                When fitting the model, the k centroids are treated as an
                image with k channels, and the channels can occur in any order.
            'mean': computes the mean summary image and uses it as a single
                channel input image.

        masks_summary: {'max_erosion', 'max'}
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

        fit_validation_metrics: {'neurofinder', 'pixelwise'}
            Metrics used to score validation data.
            'neurofinder': uses the neurofinder implementation of precision,
                recall, and F1 implemented in https://github.com/codeneuro/neurofinder-python
            'pixelwise': uses the pixelwise metrics that are used for training.

        fit_validation_proportion: proportion of series used for validation.
            This is different from https://arxiv.org/abs/1707.06314, which used
            a horizontal portion of each series for validation. This greatly
            simplifies the implementation.

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
                 model='unet2d',
                 model_path=None,
                 series_summary='kmeans',
                 masks_summary='max_erode',
                 fit_shape=(8, 128, 128),
                 fit_iters=10000,
                 fit_batch=20,
                 fit_validation_metrics='neurofinder',
                 fit_validation_interval=1000,
                 fit_validation_proportion=0.2,
                 fit_objective='binary_crossentropy',
                 fit_optimizer='adam',
                 fit_optimizer_arguments={'lr': 0.002},
                 predict_shape=(8, 512, 512),
                 predict_augmentation=True,
                 predict_threshold=0.5,
                 predict_print_metrics=True,
                 predict_save=True,
                 random_state=np.random):

        self.model = model
        self.model_path = model_path
        self.series_summary = series_summary
        self.masks_summary = masks_summary
        self.fit_shape = fit_shape
        self.fit_iters = fit_iters
        self.fit_batch = fit_batch
        self.fit_validation_metrics = fit_validation_metrics
        self.fit_validation_interval = fit_validation_interval
        self.fit_validation_proportion = fit_validation_proportion
        self.fit_objective = fit_objective
        self.fit_optimizer = fit_optimizer
        self.fit_optimizer_arguments = fit_optimizer_arguments
        self.predict_shape = predict_shape
        self.predict_augmentation = predict_augmentation
        self.predict_threshold = predict_threshold
        self.predict_print_metrics = predict_print_metrics
        self.predict_save = predict_save
        self.random_state = random_state

    def fit(self, dataset_paths_train, dataset_paths_validate):

        return

    def predict(self, dataset_paths):

        return
