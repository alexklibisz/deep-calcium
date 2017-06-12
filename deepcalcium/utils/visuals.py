from skimage.color import gray2rgb
from skvideo.io import vwrite
from regional import one
import numpy as np
import logging

from deepcalcium.utils.runtime import funcname


def dataset_to_mp4(sequence, mask, mp4_path):
    '''Converts the given sequence to an mp4 video. If the mask is given, adds an outline around each neuron.'''

    logger = logging.getLogger(funcname())
    logger.info('Preparing video %s.' % mp4_path)

    # If mask is given make a color video with neuron centers marked.
    if mask is not None:
        s = sequence.get('s')[...]
        m = mask.get('m')[...]

        video = video * 1. / 255.

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

        s = sequence.get('s')
        video = s[...]
        video = video * 1. / 255.
        video = video.astype(np.uint8)

    vwrite(mp4_path, video)
    logger.info('Saved video %s.' % mp4_path)


def mask_outlines(image, mask_arrs=[], colors=[]):
    '''Apply each of the given masks (numpy arrays) to the base image with the given colors.'''

    assert len(mask_arrs) == len(colors), 'One color per mask.'

    # Convert the image to RGB.
    if len(image.shape) == 2:
        image = gray2rgb(image)

    # Clip outliners, scale the image to [0,1], multiply by 255.
    image = np.clip(image, 0, np.percentile(image, 99))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Convert each mask into a region, then take the outlined mask
    # of that region and add it to the image.
    for m, c in zip(mask_arrs, colors):
        reg = one(list(zip(*np.where(m == 1))))
        oln = reg.mask(dims=image.shape[:2], fill=None, stroke=c, background='black')
        oln /= np.max(oln)
        yy, xx, cc = np.where(oln != 0)
        image[yy, xx, cc] = oln[yy, xx, cc]

    return image
