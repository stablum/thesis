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
    U,V = cftools.UV_np(R)
    lambda_u = 1.
    lambda_v = 1.
    def step(_Rij,_ui,_vj):
        eij = _Rij - T.dot(_ui.T, _vj)
        error_term = (-1) * eij * _vj
        prior_term = lambda_u * _ui
        grad = error_term + prior_term
        new_ui = cftools.update(_ui,grad)
        eij = _Rij - T.dot(_ui.T, _vj)
        grad = (-1) * eij * _ui
        new_vj = cftools.update(_vj,grad)
        return new_ui,new_vj

    ui = T.dvector('ui')
    vj = T.dvector('vj')
    Rij = T.dscalar('Rij')
    cftools.lr = T.dscalar('lr')
    new_ui, new_vj = step(Rij, ui, vj)
    step_fn = function(
        [cftools.lr,Rij,ui,vj],
        [new_ui,new_vj],
        mode=config.theano_mode
    )
    print "training pmf with theano (without theano.scan)..."

    for training_set,_lr in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        print("\ndatapoints loop..")
        for i,j,_Rij in tqdm(training_set_matrix):
            i = int(i)
            j = int(j)
            _ui = U[:,i]
            _vj = V[:,j]
            _ui, _vj = step_fn(_lr,_Rij, _ui, _vj)
            U[:,i] = _ui
            V[:,j] = _vj

if __name__=="__main__":
    main()

