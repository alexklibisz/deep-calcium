from glob import glob
import h5py

paths = sorted(glob('../sj-trace-classification/data/sj*hdf5'))

for p in paths:
    fp = h5py.File(p)
    print(p.split('/')[-1], fp.get('traces').shape, fp.get('spikes').shape)
    fp.close()
