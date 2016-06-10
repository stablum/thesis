import numpy as np
import os
from tqdm import tqdm
import sys
import time
import random
import theano
import theano.tensor as T
# local imports

import config

def create_training_set_matrix(training_set):
    return np.array([
        [_i,_j,_Rij]
        for (_i,_j),_Rij
        in training_set
    ])

def UV(R):
    U_values = np.random.random((config.K,R.shape[0]))
    V_values = np.random.random((config.K,R.shape[1]))
    U = theano.shared(U_values)
    V = theano.shared(V_values)
    return U,V

def wrong(x):
    return ((not np.isfinite(x)) or np.isnan(x) or x>10000. or x<-10000.)

def split_sets(R):
    Ritems = R.items()
    sel = np.random.permutation(len(Ritems))
    midpoint = int(len(Ritems)/2)
    training_set = []
    testing_set = []
    for curr in sel[:midpoint]:
        training_set.append(Ritems[curr])
    for curr in sel[midpoint+1:]:
        testing_set.append(Ritems[curr])
    return training_set, testing_set

def rating_error(Rij,U,i,V,j):
    ret = Rij - np.dot(U[:,i].T, V[:,j])
    return ret

def rmse(subset,U,V):
    errors = []
    for curr in subset:
        (i,j),Rij = curr
        eij = rating_error(Rij,U,i,V,i)
        errors.append(eij**2)
    return np.sqrt(np.mean(errors))

def check_grad(grad):
    for curr in grad.tolist():
        if np.abs(curr) > 10000:
            print "gradient %f too far from 0"%curr
            import ipdb; ipdb.set_trace()

def update(A, grad):

    # minimization implies subtraction because we are dealing with gradients
    # of the negative loglikelihood
    if type(grad) is T.TensorVariable:
        try:
            return T.inc_subtensor(A, -1 * config.lr * grad)
        except TypeError as e:
            # I don't know any other way to check if A is a subtensor
            return A - config.lr * grad
    else:
        check_grad(grad)
        A -= config.lr * grad

class Log(object):
    _file = None

    def __init__(self):
        prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        time_str = time.strftime('%Y%m%d_%H%M%S')
        log_filename = prefix + "_" + time_str + ".log"
        dirname = "logs"
        try:
            os.mkdir(dirname)
        except Exception as e:
            # cannot create dir (maybe it already exists?)
            # loudly ignore the exception
            print "cannot create dir %s: %s"%(dirname,str(e))

        full_path = os.path.join(dirname,log_filename)
        self._file = open(full_path, 'a')
        self("logging to %s"%full_path)
        self("learning rate: %f"%config.lr)
        self("K: %d"%config.K)
        self("n_epochs: %d"%config.n_epochs)

    def __call__(self,msg):

        time_str = time.strftime('%Y:%m:%d %H:%M:%S')
        msg = time_str + " " + msg
        print msg
        self._file.write(msg+"\n")

    def statistics(self,epoch_nr, training_set, testing_set, U, V):
        training_rmse = rmse(training_set,U,V)
        testing_rmse = rmse(testing_set,U,V)
        U_std, U_mean = np.std(U.tolist()), np.mean(U.tolist())
        V_std, V_mean = np.std(V.tolist()), np.mean(V.tolist())
        self("epoch %d"%epoch_nr)
        self("training RMSE: %s"%training_rmse)
        self("testing RMSE: %s"%testing_rmse)
        self("U mean: %f std: %f"%(U_mean,U_std))
        self("V mean: %f std: %f"%(V_mean,V_std))

class epochsloop(object):

    def __init__(self,R,U,V):
        np.set_printoptions(precision=4, suppress=True)
        self._log = Log()
        self.training_set, self.testing_set = split_sets(R)
        self.U = U
        self.V = V

    def __iter__(self):
        self._iter = iter(tqdm(range(config.n_epochs))) # internal "hidden" iterator
        return self

    def next(self):
        epoch_nr = self._iter.next() # will raise StopIteration when done
        if type(self.U) is T.sharedvar.TensorSharedVariable:
            _U = self.U.get_value()
            _V = self.V.get_value()
        else:
            _U = self.U
            _V = self.V
        self._log.statistics(epoch_nr, self.training_set,self.testing_set,_U,_V)
        random.shuffle(self.training_set)
        return self.training_set

