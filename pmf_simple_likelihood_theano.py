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

def main():
    R,N,M = movielens.small()
    U,V = cftools.UV(R)

    trainM = T.dmatrix('trainM')

    def step(curr,_U,_V):
        i = T.cast(curr[0],'int32')
        j = T.cast(curr[1],'int32')
        Rij = curr[2]
        eij = Rij - T.dot(_U[:,i].T, _V[:,j])
        new_U = T.inc_subtensor(_U[:,i], config.lr * eij * _V[:,j])
        eij = Rij - T.dot(new_U[:,i].T, _V[:,j])
        new_V = T.inc_subtensor(_V[:,j], config.lr * eij * _U[:,i])
        return {
            _U[:,i]:new_U[:,i],
            _V[:,j]:new_V[:,j]
        }

    values, updates = theano.scan(
        step,
        sequences=[trainM],
        non_sequences=[U,V]
    )
    scan_fn = function(
        [trainM],
        [],
        updates=updates,
        mode='FAST_RUN'
    )
    print "training pmf with theano..."

    for training_set in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        scan_fn(training_set_matrix)

if __name__=="__main__":
    main()

