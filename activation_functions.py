import theano
from theano import tensor as T
import lasagne

possible_activations = {
    'sigmoid': T.nnet.sigmoid,

    # 2.37 seems to make a sigmoid a good approximation for erf(x),
    'pseudogelu': lambda x: x * T.nnet.sigmoid(x*2.37),

    'gelu': lambda x : x*T.erf(x),
    'elu': T.nnet.elu,
    'relu': T.nnet.relu,
    'linear': lasagne.nonlinearities.linear
}

def get(name):
    return possible_activations[name]
