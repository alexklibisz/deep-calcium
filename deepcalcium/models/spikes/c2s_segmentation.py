from multiprocessing import Pool
from psutil import cpu_count
from time import time
import c2s
import h5py
import logging
import numpy as np
import os
import pickle as pkl

import sys
sys.path.append('.')
from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.spikes import np2k, F2_margin, prec_margin, reca_margin

rng = np.random
os.environ['CUDA_VISIBLE_DEVICES'] = ""


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


def c2s_preprocess_parallel(argsdict):
    logger = logging.getLogger(funcname())
    logger.info('%d start' % os.getpid())
    if len(argsdict['data']) > 1:
        return c2s.preprocess(**argsdict)
    return c2s.preprocess(**argsdict)[0]


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

    def fit(self, dataset_paths, model_path=None, folds=5, error_margin=2):

        logger = logging.getLogger(funcname())

        if not model_path:

            # Extract traces and spikes from datasets.
            traces = [self.dataset_traces_func(p) for p in dataset_paths]
            spikes = [self.dataset_spikes_func(p) for p in dataset_paths]
            attrs = [self.dataset_attrs_func(p) for p in dataset_paths]
            assert len(traces) == len(spikes) == len(attrs)

            # Populate C2S data dictionaries.
            data = []
            for i in range(len(attrs)):
                for t, s in zip(traces[i], spikes[i]):
                    data.append({'calcium': t[np.newaxis],
                                 'spikes': s[np.newaxis],
                                 'fps': attrs[i]['sample_rate']})

            # Preprocess in parallel. This is a slow process. Using lower
            # fps creates smaller vectors. Large vectors can crash the training.
            pool = Pool(max(1, cpu_count() - 2))
            args = [{'data': [d], 'fps': 10, 'verbosity':0} for d in data]
            data = pool.map(c2s_preprocess_parallel, args)

            # Serialize data.
            data_path = '%s/%d_data.pkl' % (self.cpdir, int(time()))
            fp = open(data_path, 'wb')
            pkl.dump(data, fp)
            fp.close()
            logging.info('Serialized model to %s' % data_path)

        else:
            fp = open(model_path, 'rb')
            data = pkl.load(fp)
            fp.close()

        import pdb
        pdb.set_trace()

        # Train.
        results = c2s.train(data)

        # Predict.
        data_trn = c2s.predict(data, results)

        # Evaluate using C2S metrics.
        downsample_factor = 10  # fps = 100 -> fps = 10.
        corr = np.nan_to_num(c2s.evaluate(
            data, 'corr', downsampling=downsample_factor), copy=False)
        print('Corr = %.5lf' % np.mean(corr))

        # # Compute metrics.
        # p, r = 0., 0.
        # for i, d in enumerate(data_trn):
        #     yt = d['spikes'][0, np.newaxis]
        #     yp = np.clip(d['predictions'][0, np.newaxis].round(), 0, 1)
        #
        #     p += np2k(prec_margin, yt, yp, margin=error_margin)
        #     r += np2k(reca_margin, yt, yp, margin=error_margin)
        #
        #     if i % 50 == 0 or i == len(data_trn) - 1:
        #         print '%03d: mean p=%-10.3lf mean r=%-10.3lf' % (i, (p / i), (r / i))
        #
        # p, r = 0., 0.
        # for i, d in enumerate(data_val):
        #     yt = d['spikes'][0, np.newaxis]
        #     yp = np.clip(d['predictions'][0, np.newaxis].round(), 0, 1)
        #
        #     p += np2k(prec_margin, yt, yp, margin=error_margin)
        #     r += np2k(reca_margin, yt, yp, margin=error_margin)
        #
        #     if i % 50 == 0 or i == len(data_val) - 1:
        #         print '%03d: mean p=%-10.3lf mean r=%-10.3lf' % (i, (p / i), (r / i))

        import pdb
        pdb.set_trace()

    def predict(self, dataset_paths, model_path):

        # Extract traces and spikes from datasets.
        traces = [self.dataset_traces_func(p) for p in dataset_paths]
        spikes = [self.dataset_spikes_func(p) for p in dataset_paths]

        # Load C2S data dictionary.

        # Preprocess.

        # Predict.

        # Compute metrics.

        pass
