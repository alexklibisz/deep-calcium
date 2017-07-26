from skimage.color import gray2rgb
from skvideo.io import vwrite
from regional import one
import numpy as np
import logging

from deepcalcium.utils.runtime import funcname


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
    clr_intensity = 1

    # Convert the img to RGB.
    if len(img.shape) == 2:
        img = gray2rgb(img)

    # Convert each mask into a region, then take the outlined mask
    # of that region and add it to the img.
    for m, c in zip(mask_arrs, colors):
        if np.sum(m) == 0:
            continue
        reg = one(list(zip(*np.where(m == 1))))
        oln = reg.mask(dims=img.shape[:2], fill=None,
                       stroke=c, background='black')
        oln /= np.max(oln)
        yy, xx, cc = np.where(oln != 0)
        img[yy, xx, cc] = oln[yy, xx, cc] * clr_intensity

    return img
