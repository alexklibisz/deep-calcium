# Convert neurofinder datasets to MP4 videos.
# Save the videos in their respective dataset directories.
# Takes a long time to run for all the datasets.
from os.path import exists
import logging
import sys
sys.path.append('.')

from deepcalcium.datasets.nf import nf_load_hdf5, neurofinder_names
from deepcalcium.utils.visuals import dataset_to_mp4

logging.basicConfig(level=logging.INFO)

name = ['neurofinder.04.00', 'neurofinder.04.01']
S, M = nf_load_hdf5(name)
for s, m in zip(S, M):
    mp4_path = s.filename.replace('.hdf5', '.mp4')
    if not exists(mp4_path):
        dataset_to_mp4(s, m, mp4_path)
    s.close()
    m.close()
