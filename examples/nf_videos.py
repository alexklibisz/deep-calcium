# Convert neurofinder datasets to MP4 videos.
# Save the videos in their respective dataset directories.
# Takes a long time to run for all the datasets.
from os.path import exists
import logging
import sys
sys.path.append('.')

from deepcalcium.datasets.nf import load_neurofinder, neurofinder_names
from deepcalcium.utils.visuals import dataset_to_mp4

logging.basicConfig(level=logging.INFO)

for name in neurofinder_names:
    S, M = load_neurofinder(name)
    mp4_path = s.filename.replace('.hdf5', '.mp4')
    if not exists(mp4_path):
        dataset_to_mp4(S[0], M[0], mp4_path)
    s.close()
    m.close()
