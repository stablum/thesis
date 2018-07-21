import theano
from theano import tensor as T
import lasagne

safe_log_output_K = 1 #7

possible_activations = {
    'sigmoid': T.nnet.sigmoid,

    # 2.37 seems to make a sigmoid a good approximation for erf(x),
    'pseudogelu': lambda x: x * T.nnet.sigmoid(x*2.37),

    'gelu': lambda x : x*T.erf(x),
    'elu': T.nnet.elu,
    'relu': T.nnet.relu,
    'linear': lasagne.nonlinearities.linear,
    'tanh': lasagne.nonlinearities.tanh,
    'safe_log_output': lambda x : safe_log_output_K * lasagne.nonlinearities.tanh(x/safe_log_output_K),
    'pseudo_linear': lambda x : 20 * lasagne.nonlinearities.tanh(x/20)
}

def get(name):
    return possible_activations[name]
