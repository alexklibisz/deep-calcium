# Convert neurofinder datasets to MP4 videos.
# Save the videos in their respective dataset directories.
# Takes a long time to run for all the datasets.
from os.path import exists
import logging
import sys
sys.path.append('.')

from deepcalcium.datasets.neurofinder import load_neurofinder, neurofinder_names
from deepcalcium.utils.misc import dataset_to_mp4

logging.basicConfig(level=logging.INFO)

for name in neurofinder_names:
    s, m = load_neurofinder(name)
    mp4_path = s.filename.replace('.hdf5', '.mp4')
    if not exists(mp4_path):
        dataset_to_mp4(s, m, mp4_path)
    s.close()
    m.close()
