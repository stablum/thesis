#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
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

    lambda_v = 1.
    lambda_u = 1.

    mu_u_values = np.random.random(config.K)*10. - 5.
    mu_v_values = np.random.random(config.K)*10. - 5.
    mu_u = theano.shared(mu_u_values)
    mu_v = theano.shared(mu_v_values)

    lr = T.dscalar('lr')
    trainM = T.dmatrix('trainM')

    def step(curr,_lr,_U,_V,_mu_u,_mu_v):
        i = T.cast(curr[0],'int32')
        j = T.cast(curr[1],'int32')
        Rij = curr[2]

        eij = Rij - T.dot(_U[:,i].T, _V[:,j])
        grad = -1 * eij * _U[:,i] + (1./N) * lambda_u * (_U[:,i] - _mu_u)
        new_U = cftools.update(_U[:,i], grad)

        grad = -1 * eij * _V[:,j] + (1./N) * lambda_v * (_V[:,j] - _mu_v)
        eij = Rij - T.dot(new_U[:,i].T, _V[:,j])
        new_V = cftools.update(_V[:,j], grad)

        grad = lambda_u * (_mu_u - new_U[:,i])
        new_mu_u = cftools.update(_mu_u, grad)

        grad = lambda_v * (_mu_v - new_V[:,j])
        new_mu_v = cftools.update(_mu_v, grad)

        return {
            _U:new_U,
            _V:new_V,
            _mu_u:new_mu_u,
            _mu_v:new_mu_v
        }

    values, updates = theano.scan(
        step,
        sequences=[trainM],
        non_sequences=[lr, U, V, mu_u, mu_v]
    )

    scan_fn = function(
        [trainM,lr],
        [],
        updates=updates,
        mode='FAST_RUN'
    )

    print "training pmf with mu hyperpriors using theano..."

    for training_set in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        scan_fn(training_set_matrix, config.lr)
        print "mu_u",mu_u.get_value()
        print "mu_v",mu_v.get_value()

if __name__=="__main__":
    main()
