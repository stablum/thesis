import config
import theano
import lasagne
from theano import tensor as T
import numpy as np
import collections

lr = config.lr_begin # can be replaced externally, for example by a theano variable

def calculate_lr(t):
    # decaying learning rate with annealing
    # see: https://www.willamette.edu/~gorr/classes/cs449/momrate.html
    ret = config.lr_begin / (
        1. + float(t)/config.lr_annealing_T
    )
    return ret

def check_grad(grad):
    for curr in grad.tolist():
        if np.abs(curr) > 10000:
            print("gradient %f too far from 0"%curr)
            import ipdb; ipdb.set_trace()

def sgd(A,grad,**kwargs):

    # minimization implies subtraction because we are dealing with gradients
    # of the negative loglikelihood
    if type(grad) is T.TensorVariable:
        try:
            # subtensor..
            ret = T.inc_subtensor(A, -1 * lr * grad)
            print("update: A is a subtensor %s"%(str(A)))
            return ret
        except TypeError as e:
            print("update: A is NOT a subtensor %s (exception %s)"%(str(A),str(e)))
            # not subtensor..
            # I didn't know any other way to check if A is a subtensor
            return A - lr * grad,
    else:
        check_grad(grad)
        A -= lr * grad
        return A,

def adam_np(A,grad):
    #print("adam.")
    assert A.shape == A.m.shape == A.v.shape,\
        "adam: shape of A (%d), m (%d) and v (%d) has to be the same"%(
            A.shape,
            A.m.shape,
            A.v.shape
        )
    beta1 = config.adam_beta1
    beta2 = config.adam_beta2

    new_m = beta1 * A.m + ( 1 - beta1 ) * grad
    new_v = beta2 * A.v + ( 1 - beta2 ) * (grad ** 2)
    m_hat = new_m / (1 - A.beta1_t) # bias correction
    v_hat = new_v / (1 - A.beta2_t)
    A -= (lr * m_hat) / ( v_hat ** 0.5 )
    A.m = new_m
    A.v = new_v
    A.t = A.t + 1 # this is the number of times this learnable object has been updated
    A.beta1_t = A.beta1_t * beta1
    A.beta2_t = A.beta2_t * beta2
    #print("done.")
    return A

def adam_for(l,two_dims=False):
    ts = []
    ms = []
    vs = []
    for param in l:
        t = 0
        if two_dims is True:
            t = np.array([[t]], dtype='float32')
        ts.append(t)
        m = np.zeros(param.shape, dtype='float32')
        v = np.zeros(param.shape, dtype='float32')

        if two_dims is True and len(param.shape) ==1:
            m = np.expand_dims(m,0)
            v = np.expand_dims(v,0)
        ms.append(m)
        vs.append(v)

    return ts,ms,vs

def adam_symbolic(
        A,
        g_t,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        t_prev=None,
        m_prev=None,
        v_prev=None
    ):
    """Adam updates (code adapted from Lasagne's implementation)

    Adam updates implemented as in [1]_.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """

    assert None not in [t_prev,m_prev,v_prev]

    updates = collections.OrderedDict()

    one = T.constant(1)

    t = t_prev + 1
    t.name = "t"
    a_t_numer = lr*T.sqrt(one-beta2**t)
    a_t_numer.name = "a_t_numer"
    a_t_denom = (one-beta1**t)
    a_t_denom.name="a_t_denom"
    a_t = a_t_numer/a_t_denom
    a_t.name="a_t"

    m_t_term1 = beta1*m_prev
    m_t_term1.name = "m_t_term1"
    m_t_term2 = (1-beta1)*g_t
    m_t_term2.name = "m_t_term2"
    m_t = m_t_term1 + m_t_term2
    m_t.name = "m_t"
    v_t_term1 = beta2*v_prev
    v_t_term1.name = "v_t_term1"
    v_t_term2 = (1-beta2)*(g_t**2)
    v_t_term2.name = "v_t_term2"
    v_t = v_t_term1 + v_t_term2
    v_t.name = "v_t"
    denom = (T.sqrt(v_t) + epsilon)
    denom.name = "denom"
    numer = a_t*m_t
    numer.name = "numer"
    step = numer/denom
    step.name = "step"
    new_A = A - step
    new_A.name = "new_A"
    return new_A, t, m_t, v_t

def get_func():
    d = {
        'adam_symbolic':adam_symbolic,
        'adam_np':adam_np,
        'adam_lasagne':lasagne.updates.adam,
        'sgd':sgd,
        'sgd_masked':sgd_masked
    }
    return d[config.update_algorithm]

def adam_masked(all_grads, params, masks, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = collections.OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):

        value = param.get_value(borrow=True)
        if param.name in masks.keys():
            mask = masks[param.name]
        else:
            mask = np.ones(value.shape).astype('float32')
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t_all = beta1*m_prev + (one-beta1)*g_t
        v_t_all = beta2*v_prev + (one-beta2)*g_t**2

        # changing only the parameters that are indicated by the mask
        m_t = m_t_all * mask + m_prev * (1-mask)
        v_t = v_t_all * mask + v_prev * (1-mask)

        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

def sgd_masked(all_grads,params,masks,**kwargs):
    updates = collections.OrderedDict()
    for param, grad in zip(params, all_grads):
        # minimization implies subtraction because we are dealing with gradients
        # of the negative loglikelihood
        if type(grad) is not T.TensorVariable:
            raise Exception("why is this grad not a T.TensorVariable but a {}??".format(type(grad)))


        if param.name in masks.keys():
            mask = masks[param.name]
        else:
            value = param.get_value(borrow=True)
            mask = np.ones(value.shape).astype('float32')
        try:
            # subtensor..
            ret = T.inc_subtensor(param, -1 * lr * grad * mask)
            print("update: param is a subtensor %s"%(str(param)))
            updates[param] = ret
        except TypeError as e:
            print("update: param is NOT a subtensor %s (exception %s)"%(str(param),str(e)))
            # not subtensor..
            # I didn't know any other way to check if param is a subtensor
            ret = param - lr * grad * mask
            updates[param] = ret
    return updates
