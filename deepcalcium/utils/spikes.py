import keras.backend as K
import numpy as np


def np2k(kfunc, yt, yp, **args):
    yt = K.variable(yt)
    yp = K.variable(yp)
    return K.get_value(kfunc(yt, yp, **args))


def weighted_binary_crossentropy(yt, yp, weightpos=2., weightneg=1.):
    """Apply different weights to true positives and true negatives with
    binary crossentropy loss.

    # Arguments
        yt, yp: Keras ground-truth and predicted batch matrices.
        weightpos: weight multiplier for loss on true-positives.
        weightnet: weight multiplier for loss on true-negatives.

    # Returns
        matrix of loss scalars with shape (batch size x 1).

    """

    losspos = yt * K.log(yp + 1e-7)
    lossneg = (1 - yt) * K.log(1 - yp + 1e-7)
    return -1 * ((weightpos * losspos) + (weightneg * lossneg))


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


def ytspks(yt, yp):
    """On average, how many spikes in each yt spikes sample."""
    return K.sum(yt, axis=1)


def ypspks(yt, yp):
    """On average, how many spikes in each yp spikes prediction."""
    return K.sum(K.round(yp), axis=1)


def plot_traces_spikes(traces, spikes_true=None, spikes_pred=None, title=None, save_path=None, dpi=100, fig_width=20, legend=True):

    if save_path:
        import matplotlib
        matplotlib.use('agg')

    import matplotlib.pyplot as plt

    figsize = (fig_width, traces.shape[0] * 1.7)
    fig, axes = plt.subplots(traces.shape[0], 1, figsize=figsize)
    axes = axes if traces.shape[0] > 1 else [axes]
    for i, ax in enumerate(axes):

        # Plot signal.
        t = traces[i]
        ax.plot(t, c='k', linewidth=1.0)

        # Scatter points for true spikes (blue circle).
        if type(spikes_true) == np.ndarray:
            xxt, = np.where(spikes_true[i] == 1)
            ax.scatter(xxt, t[xxt], c='cyan', marker='o', s=150,
                       alpha=0.8, label='Ground-truth spike')

        # Plot the line segments with predicted spikes.
        # There is likely a more efficient way to do this.
        if type(spikes_pred) == np.ndarray:
            xx, = np.where(spikes_pred[i].round() == 1)
            label = 'Predicted spikes'
            for x in xx:
                ax.plot([x, x + 1], t[[x, x + 1]], 'r', label=label)
                label = None  # Only label first one.

        if (i == 0 or i == len(axes) - 1) and legend:
            ax.legend(loc='lower left', ncol=3)

        ax.set_ylabel('Brightness')
        ax.set_xlabel('Time steps')

    plt.subplots_adjust(left=None, wspace=None, hspace=0.7, right=None)
    if title:
        plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=dpi, mode=save_path.split('.')[-1],
                    bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
