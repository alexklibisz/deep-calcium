from __future__ import division
from scipy.misc import imsave
from skimage.color import gray2rgb, rgb2gray
from skvideo.io import vwrite
from regional import one
import keras.backend as K
import numpy as np
import logging

from deepcalcium.utils.runtime import funcname


def weighted_binary_crossentropy(yt, yp, weightpos=2., weightneg=1.):
    """Apply different weights to true positives and true negatives with
    binary crossentropy loss.

    # Arguments
        yt, yp: Keras ground-truth and predicted batch matrices.
        weightpos: weight multiplier for loss on true-positives.
        weightnet: weight multiplier for loss on true-negatives.

    # Returns
        matrix of loss scalars with shape (batch size x 1).

    """

    losspos = yt * K.log(yp + 1e-7)
    lossneg = (1 - yt) * K.log(1 - yp + 1e-7)
    return -1 * ((weightpos * losspos) + (weightneg * lossneg))


def prec(yt, yp):
    """Keras precision metric."""
    yp = K.round(yp)
    return K.sum(yp * yt) / (K.sum(yp) + K.epsilon())


def reca(yt, yp):
    """Keras recall metric."""
    yp = K.round(yp)
    tp = K.sum(yp * yt)
    fn = K.sum(K.clip(yt - yp, 0, 1))
    return K.sum(yp * yt) / (tp + fn + K.epsilon())


def F1(yt, yp):
    """Keras F1 metric."""
    p = prec(yt, yp)
    r = reca(yt, yp)
    return (2 * p * r) / (p + r + K.epsilon())


def jacc(yt, yp):
    """Keras Jaccard coefficient metric."""
    yp = K.round(yp)
    inter = K.sum(yt * yp)
    union = K.sum(yt) + K.sum(yp) - inter
    return inter / (union + 1e-7)


def jacc_loss(yt, yp):
    """Smooth Jaccard loss. Cannot round yp because that results in a
    non-differentiable function."""
    inter = K.sum(yt * yp)
    union = K.sum(yt) + K.sum(yp) - inter
    jsmooth = inter / (union + 1e-7)
    return 1 - jsmooth


def dice(yt, yp):
    """Standard dice coefficient. Dice and F1 are equivalent, worked out nicely here:
    https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/."""
    yp = K.round(yp)
    inter = K.sum(yt * yp)
    return (2. * inter) / (K.sum(yt) + K.sum(yp) + 1e-7)


def dice_loss(yt, yp):
    """Approximate dice coefficient loss function. Cannot round yp because
    that results in a non-differentiable function."""
    inter = K.sum(yt * yp)
    dsmooth = (2. * inter) / (K.sum(yt) + K.sum(yp) + 1e-7)
    return 1 - dsmooth


def dicesq(yt, yp):
    """Squared dice-coefficient metric. From https://arxiv.org/abs/1606.04797."""
    nmr = 2 * K.sum(yt * yp)
    dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
    return (nmr / dnm)


def dicesq_loss(yt, yp):
    return -1 * dicesq(yt, yp)


def posyt(yt, yp):
    """Proportion of positives in the ground-truth mask."""
    size = K.sum(K.ones_like(yt))
    return K.sum(yt) / (size + K.epsilon())


def posyp(yt, yp):
    """Proportion of positives in the predicted mask."""
    size = K.sum(K.ones_like(yp))
    return K.sum(K.round(yp)) / (size + K.epsilon())


# Augmentations that can be applied to a batch of 2D images and inverted.
# Structure is the augmentation name, the augmentation, and the inverse
# of the augmentation. Intended for test-time augmentation for segmentation.
INVERTIBLE_2D_AUGMENTATIONS = [
    ('identity',
     lambda x: x,
     lambda x: x),
    ('vflip',
     lambda x: x[:, ::-1, ...],
     lambda x: x[:, ::-1, ...]),
    ('hflip',
     lambda x: x[:, :, ::-1],
     lambda x: x[:, :, ::-1]),
    ('rot90',
     lambda x: np.rot90(x, 1, axes=(1, 2)),
     lambda x: np.rot90(x, -1, axes=(1, 2))),
    ('rot180',
     lambda x: np.rot90(x, 2, axes=(1, 2)),
     lambda x: np.rot90(x, -2, axes=(1, 2))),
    ('rot270',
     lambda x: np.rot90(x, 3, axes=(1, 2)),
     lambda x: np.rot90(x, -3, axes=(1, 2))),
    ('rot90vflip',
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...],
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...]),
    ('rot90hflip',
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1],
     lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1])
]


def dataset_to_mp4(s, m, mp4_path):
    """Converts the given series to an mp4 video. If the mask is given, adds an outline around each neuron.

    # Arguments
        s: imaging series as a (time x height x width) numpy array.
        m: neuron masks as a (no. neurons x height x width) numpy array.
        mp4_path: path where the mp4 file should be saved.

    # Returns
        Nothing

    """

    logger = logging.getLogger(funcname())
    logger.info('Preparing video %s.' % mp4_path)

    s = s.astype(np.float32)
    s = (s - np.min(s)) / (np.max(s) - np.min(s)) * 255

    # If mask is given make a color video with neurons outlined.
    if m is not None:
        video = np.zeros(s.shape + (3,), dtype=np.uint8)
        video[:, :, :, 0] = s
        video[:, :, :, 1] = s
        video[:, :, :, 2] = s

        outlines = np.zeros(s.shape[1:] + (3,), )
        for i in range(m.shape[0]):
            reg = one(list(zip(*np.where(m[i] == 1))))
            outlines += reg.mask(dims=m.shape[1:], fill=None,
                                 stroke='white', background='black')

        yy, xx, _ = np.where(outlines != 0)
        video[:, yy, xx, :] = [102, 255, 255]

    else:
        video = s.astype(np.uint8)

    vwrite(mp4_path, video)
    logger.info('Saved video %s.' % mp4_path)


def mask_outlines(img, mask_arrs=[], colors=[]):
    """Apply each of the given masks (numpy arrays) to the base img with the given colors.

    # Arguments
        img: base image as a (height x width) numpy array.
        mask_arrs: list of masks as (height x width) numpy arrays that should be outlined.
        colors: one color (e.g. 'red' or hex code) for each mask.

    # Returns
        img: The base image with outlines applied.

    """

    assert len(mask_arrs) == len(colors), 'One color per mask.'
    img = img.astype(np.float32)

    # Clip outliners, scale the img to [0,1].
    img = np.clip(img, 0, np.percentile(img, 99))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Convert the img to RGB.
    if len(img.shape) == 2:
        img_rgb = gray2rgb(img)

    # Build up an image of combined outlines.
    oln_rgb = np.zeros_like(img_rgb)
    for m, c in zip(mask_arrs, colors):
        if np.sum(m) == 0:
            continue
        r = one(list(zip(*np.where(m == 1))))
        o = r.mask(dims=img_rgb.shape[:2], fill=None,
                   stroke=c, background='black')
        yy, xx, cc = np.where(o != 0)
        oln_rgb[yy, xx, cc] = o[yy, xx, cc]

    # Merge the two images.
    # Helpful stackoverflow post: https://stackoverflow.com/questions/40895785
    oln_rgb = np.clip(oln_rgb, 0, 1)
    oln_msk = np.max(oln_rgb, axis=-1)
    img_msk = 1 - oln_msk
    oln_msk = gray2rgb(oln_msk)
    img_msk = gray2rgb(img_msk)
    mrg = (((oln_rgb * oln_msk) + (img_rgb * img_msk)) * 255).astype(np.uint8)

    return mrg
