import keras.backend as K


def weighted_binary_crossentropy(yt, yp, wfp=1., wfn=5.):

    # Standard log loss.
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))

    # Compute weight matrix, scaled by the error at each pixel.
    fpmat = (1 - yt) * wfp
    fnmat = yt * wfn
    wmat = fnmat + fpmat
    return K.mean(loss * wmat)


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


def F1(yt, yp):
    """Keras F1 metric."""
    p = prec(yt, yp)
    r = reca(yt, yp)
    return (2 * p * r) / (p + r + K.epsilon())


def jacc(yt, yp):
    """Keras Jaccard coefficient metric."""
    yp = K.round(yp)
    inter = K.sum(yt * yp)
    union = K.sum(yt) + K.sum(yp) - inter
    return inter / (union + 1e-7)


def jacc_loss(yt, yp):
    """Smooth Jaccard loss. Cannot round yp because that results in a
    non-differentiable function."""
    inter = K.sum(yt * yp)
    union = K.sum(yt) + K.sum(yp) - inter
    jsmooth = inter / (union + 1e-7)
    return 1 - jsmooth


def dice(yt, yp):
    """Standard dice coefficient. Dice and F1 are equivalent, worked out nicely here:
    https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/."""
    yp = K.round(yp)
    inter = K.sum(yt * yp)
    return (2. * inter) / (K.sum(yt) + K.sum(yp) + 1e-7)


def dice_loss(yt, yp):
    """Approximate dice coefficient loss function. Cannot round yp because
    that results in a non-differentiable function."""
    inter = K.sum(yt * yp)
    dsmooth = (2. * inter) / (K.sum(yt) + K.sum(yp) + 1e-7)
    return 1 - dsmooth


def dicesq(yt, yp):
    """Squared dice-coefficient metric. From https://arxiv.org/abs/1606.04797."""
    nmr = 2 * K.sum(yt * yp)
    dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
    return (nmr / dnm)


def dicesq_loss(yt, yp):
    return -1 * dicesq(yt, yp)


def posyt(yt, yp):
    """Proportion of positives in the ground-truth mask."""
    size = K.sum(K.ones_like(yt))
    return K.sum(yt) / (size + K.epsilon())


def posyp(yt, yp):
    """Proportion of positives in the predicted mask."""
    size = K.sum(K.ones_like(yp))
    return K.sum(K.round(yp)) / (size + K.epsilon())
