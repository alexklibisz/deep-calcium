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

        video = np.zeros((*s.shape, 3), dtype=np.uint8)
        video[:, :, :, 0] = s
        video[:, :, :, 1] = s
        video[:, :, :, 2] = s

        outlines = np.zeros((*s.shape[1:], 3), )
        for i in range(m.shape[0]):
            reg = one(list(zip(*np.where(m[i] == 1))))
            outlines += reg.mask(dims=m.shape[1:], fill=None,
                                 stroke='white', background='black')

        yy, xx, _ = np.where(outlines != 0)
        video[:, yy, xx, :] = [102, 255, 255]

    else:

        s = sequence.get('s')
        video = s[...]
        video = video.astype(np.uint8)

    vwrite(mp4_path, video)
    logger.info('Saved video %s.' % mp4_path)
