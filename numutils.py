"""
numerical functions that are generic enough to be not dependent on a
specific context
"""

import numpy as np
import scipy.spatial.distance
import scipy.stats
import sklearn.mixture

def sumnormalize(ar):
    """
    returns the list of scalars having all values normalized by the sum,
    which means that the sum of these values is going to be 1
    """
    _sum = np.sum(ar)
    ar_norm = [ float(curr)/_sum for curr in ar]
    return ar_norm

def maxnormalize(ar):
    """
    returns the given array linearly normalized in order to have the
    maximum set at 1.
    """
    assert type(ar) is np.ndarray
    _max = np.amax(ar)
    ret = ar.astype('float')/float(_max)
    return ret

def sigmoid_derivative(X):
    f_x = sigmoid(X)
    return f_x * (1 - f_x)

def sigmoid(X):
    """
    a specific realization of the logistic function
    """
    return logistic(X,1,0)

def logistic(X,steepness,midpoint):
    """
    logistic function (sigmoid)
    should work with both scalars and numpy arrays
    """
    exponent = - steepness * ( X - midpoint )
    denominator = 1. + np.exp(exponent)
    ret = np.power(denominator, -1.)
    return ret

def logit(X):
    """
    inverse of logistic function
    """
    return np.log(X)-np.log(1-X)

def softmax(w, t = 1.0):
    """
    squashes any vector into a vector with values from 0 to 1.
    Sum of these values will be 1.
    Useful to create probabilities out of scores.
    """

    if any([np.isnan(z) for z in w]):
        ru.error("cannot do softmax if there is a nan in w=%s"%str(w))
        import ipdb;ipdb.set_trace()

    m = np.mean(w)
    w = w - m # softmax is invariant if a constant is added
    e = np.exp(np.array(w) / t)

    if float('inf') in e:
        ret = (np.ones(len(w))/float(len(w) - 1)) / 10000000.
        ret[np.argmax(e)] = 0.9999999
        return ret

    dist = e / np.sum(e)
    if any([np.isnan(z) for z in dist]):
        ru.error("nan is going to be returned by softmax. dist=%s"%str(dist))
        import ipdb;ipdb.set_trace()
    return dist

