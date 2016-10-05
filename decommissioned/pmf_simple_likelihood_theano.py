#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
import sys
from tqdm import tqdm
import theano
from theano import tensor as T
from theano import function

# local imports

import movielens
import cftools
import config

theano_mode = 'DebugMode'
theano.mode = theano_mode
theano.config.optimizer = 'None'
theano.config.exception_verbosity='high'

def main():

    R,N,M = movielens.small()
    U,V = cftools.UV(R)
    i_mat = T.imatrix('i_mat')
    j_mat = T.imatrix('j_mat')
    Rij_mat = T.dmatrix('Rij_mat')

    def step(i_,j_,Rij_,_U,_V):
        cftools.test_value(i_, np.array([0.5]))
        cftools.test_value(j_, np.array([0.5]))
        cftools.test_value(Rij_, np.array([0.5]))
        i = i_[0]
        j = j_[0]
        Rij = Rij_[0]
        eij = Rij - T.dot(_U[:,i].T, _V[:,j])
        new_U = T.inc_subtensor(_U[:,i], config.lr * eij * _V[:,j])
        eij = Rij - T.dot(new_U[:,i].T, _V[:,j])
        new_V = T.inc_subtensor(_V[:,j], config.lr * eij * new_U[:,i])
        return {
            _U:new_U,
            _V:new_V
        }

    values, updates = theano.scan(
        step,
        sequences=[i_mat,j_mat,Rij_mat],
        non_sequences=[U,V]
    )
    scan_fn = function(
        [i_mat,j_mat,Rij_mat],
        [],#[ item[0] for item in updates.items()],
        updates=updates,
        mode=theano_mode
    )
    print "training pmf with theano..."

    for training_set in cftools.epochsloop(R,U,V):
        _i_mat,_j_mat,_Rij_mat = cftools.create_training_set_apart(training_set)
        assert _i_mat.shape == _j_mat.shape == _Rij_mat.shape

        scan_fn(_i_mat,_j_mat,_Rij_mat)

if __name__=="__main__":
    main()

