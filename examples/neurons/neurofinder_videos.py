# Convert neurofinder datasets to MP4 videos.
# Save the videos in their respective dataset directories.
# Takes a long time to run for all the datasets.
import h5py
import logging
import sys
sys.path.append('.')

from deepcalcium.datasets.nf import nf_load_hdf5
from deepcalcium.utils.neurons import dataset_to_mp4

logging.basicConfig(level=logging.INFO)

names = ['neurofinder.01.00']
dataset_paths = nf_load_hdf5(names)
for dsp in dataset_paths:
    fp = h5py.File(dsp)
    s = fp.get('series/raw')[...]
    m = fp.get('masks/raw')[...] if 'masks/raw' in fp else None
    mp4_path = dsp.replace('.hdf5', '.mp4')
    dataset_to_mp4(s, m, mp4_path)
    fp.close()
