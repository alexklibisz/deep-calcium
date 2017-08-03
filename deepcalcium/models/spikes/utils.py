import keras.backend as K
import numpy as np


def np2k(kfunc, yt, yp, **args):
    yt = K.variable(yt)
    yp = K.variable(yp)
    return K.get_value(kfunc(yt, yp, **args))


def maxpool1D(x, pool_size, pool_strides, padding='same'):
    """1D pooling along the rows of a 2D array. Requires reshaping to work
    with the keras pool2d function."""
    x = K.expand_dims(K.expand_dims(x, axis=0), axis=-1)
    x = K.pool2d(x, (1, pool_size), (1, pool_strides), padding=padding)
    return x[0, :, :, 0]


def prec(yt, yp):
    """Keras precision metric."""
    yp = K.round(yp)
    return K.sum(yp * yt) / (K.sum(yp) + K.epsilon())


def reca(yt, yp):
    """Keras recall metric."""
    yp = K.round(yp)
    tp = K.sum(yp * yt)
    fn = K.sum(K.clip(yt - yp, 0, 1))
    return K.sum(yp * yt) / (tp + fn + K.epsilon())


def F2(yt, yp, beta=2.0):
    p = prec(yt, yp)
    r = reca(yt, yp)
    return (1 + beta**2) * ((p * r) / (beta**2 * p + r + K.epsilon()))


def prec_margin(yt, yp, margin=1):
    L, S = 2 * margin + 1, 1
    return prec(maxpool1D(yt, L, S), maxpool1D(yp, L, S))


def reca_margin(yt, yp, margin=1):
    L, S = 2 * margin + 1, 1
    return reca(maxpool1D(yt, L, S), maxpool1D(yp, L, S))


def F2_margin(yt, yp, margin=1):
    L, S = 2 * margin + 1, 1
    return F2(maxpool1D(yt, L, S), maxpool1D(yp, L, S))


def ytspks(yt, yp):
    """On average, how many spikes in each yt spikes sample."""
    return K.sum(yt, axis=1)


def ypspks(yt, yp):
    """On average, how many spikes in each yp spikes prediction."""
    return K.sum(K.round(yp), axis=1)


def plot_traces_spikes(traces, spikes_true, spikes_pred=None, title='Traces and spikes', save_path=None, dpi=100):

    if save_path:
        import matplotlib
        matplotlib.use('agg')

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(traces.shape[0], 1, figsize=(20, traces.shape[0]))
    for i, ax in enumerate(axes):

        # Plot signal.
        t = traces[i]
        ax.plot(t, c='k', linewidth=0.5)

        # Scatter points for true spikes (blue circle).
        xxt, = np.where(spikes_true[i] == 1)
        ax.scatter(xxt, t[xxt], c='b', marker='o',
                   alpha=0.5, label='True spike.')

        if type(spikes_pred) == np.ndarray:

            # Scatter points for predicted spikes (red x).
            xxp, = np.where(spikes_pred[i].round() == 1)
            ax.scatter(xxp, t[xxp], c='r', marker='x',
                       alpha=0.5, label='False positive.')

            # Scatter points for correctly predicting spikes.
            xxc = np.intersect1d(xxt, xxp)
            ax.scatter(xxc, t[xxc], c='g', marker='x',
                       alpha=1., label='True positive.')

        if i == 0 or i == len(axes) - 1:
            ax.legend()

    plt.subplots_adjust(left=None, wspace=None, hspace=0.5, right=None)
    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.show()
