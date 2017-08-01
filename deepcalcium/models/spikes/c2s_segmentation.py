import c2s
import h5py
import numpy as np
import os
import pickle as pkl

rng = np.random


def _dataset_attrs_func(dspath):
    fp = h5py.File(dspath)
    attrs = {k: v for k, v in fp.attrs.iteritems()}
    fp.close()
    return attrs


def _dataset_traces_func(dspath):
    fp = h5py.File(dspath)
    traces = fp.get('traces')[...]
    fp.close()
    return traces


def _dataset_spikes_func(dspath):
    fp = h5py.File(dspath)
    spikes = fp.get('spikes')[...]
    fp.close()
    return spikes


class C2SSegmentation(object):

    def __init__(self, cpdir='%s/.deep-calcium-datasets/tmp' % os.path.expanduser('~'),
                 dataset_attrs_func=_dataset_attrs_func,
                 dataset_traces_func=_dataset_traces_func,
                 dataset_spikes_func=_dataset_spikes_func):

        self.cpdir = cpdir
        self.dataset_attrs_func = dataset_attrs_func
        self.dataset_traces_func = dataset_traces_func
        self.dataset_spikes_func = dataset_spikes_func

        if not os.path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, dataset_paths, prop_trn=0.8, prop_val=0.2, error_margin=2):

        # Extract traces and spikes from datasets.
        traces = [self.dataset_traces_func(p) for p in dataset_paths]
        spikes = [self.dataset_spikes_func(p) for p in dataset_paths]
        attrs = [self.dataset_attrs_func(p) for p in dataset_paths]
        assert len(traces) == len(spikes) == len(attrs)

        # Split for validation and training.
        idxs = [list(range(x.shape[0])) for x in traces]
        idxs_trn = [rng.choice(ix, int(len(ix) * prop_trn), replace=False)
                    for ix in idxs]
        idxs_val = [sorted(list(set(ix) - set(ixt)))
                    for ix, ixt in zip(idxs, idxs_trn)]
        traces_trn = [traces[i][ix, :] for i, ix in enumerate(idxs_trn)]
        spikes_trn = [spikes[i][ix, :] for i, ix in enumerate(idxs_trn)]
        traces_val = [traces[i][ix, :] for i, ix in enumerate(idxs_val)]
        spikes_val = [spikes[i][ix, :] for i, ix in enumerate(idxs_val)]

        # Populate C2S data dictionaries.
        data_trn = []
        for i in range(len(attrs)):
            for t, s in zip(spikes_trn[i], traces_trn[i]):
                data_trn.append({'calcium': t[np.newaxis],
                                 'fps': attrs[i]['sample_rate'],
                                 'spikes': s[np.newaxis]})

        data_val = []
        for i in range(len(attrs)):
            for t, s in zip(spikes_val[i], traces_val[i]):
                data_val.append({'calcium': t[np.newaxis],
                                 'fps': attrs[i]['sample_rate'],
                                 'spikes': s[np.newaxis]})

        data_trn = data_trn[:2]
        data_val = data_val[:2]

        # Preprocess and train.
        data_trn = c2s.preprocess(data_trn, verbosity=1)
        # data_trn = c2s.train(data_trn)

        import pdb
        pdb.set_trace()

        # Predict and compute metrics.

        # Serialize data list.

        pass

    def predict(self, dataset_paths, model_path):

        # Extract traces and spikes from datasets.
        traces = [self.dataset_traces_func(p) for p in dataset_paths]
        spikes = [self.dataset_spikes_func(p) for p in dataset_paths]

        # Load C2S data dictionary.

        # Preprocess.

        # Predict.

        # Compute metrics.

        pass
