#!/usr/bin/env python3

import scipy
import ipdb
import theano
from theano import tensor as T
import lasagne
import numpy as np
import random
import sys
from tqdm import tqdm
import ipdb

# local imports

import movielens
import cftools
import config
import numutils as nu
import augmented_types as at

import update_algorithms

theano.config.exception_verbosity="high"
theano.config.optimizer='None'

update = update_algorithms.get_func()

g = theano.tensor.nnet.sigmoid

sigma = 1.

def main():
    R,N,M = movielens.small()

    U,V = cftools.UV_vectors(R)
    u_update_fns, v_update_fns = [], []

    lambda_u = 20.
    lambda_v = 20.

    def predict(ui,vj):
        prediction = g(T.dot(ui.T,vj))
        return prediction

    def predict_to_5(ui,vj):
        prediction = predict(ui,vj)
        ret = (prediction * (config.max_rating - 1. )) + 1.
        return ret

    def make_nll_term(Rij,ui,vj):
        eij = Rij - predict(ui,vj)
        ret = 0.5 * 1./(sigma**2) * eij
        return ret

    print("creating update functions..")

    vj_symbol = T.dvector('vj')
    ui_symbol = T.dvector('ui')
    Rij_symbol = T.dscalar('Rij')

    for i,ui in tqdm(list(enumerate(U)),desc='ui'):
        nll_term = make_nll_term(Rij_symbol,ui,vj_symbol)
        grads = T.grad(nll_term, ui)
        updates = update([grads],[ui],learning_rate = update_algorithms.lr)
        fn = theano.function([Rij_symbol,vj_symbol],[],updates=updates)
        u_update_fns.append(fn)

    for j,vj in tqdm(list(enumerate(V)),desc='vj'):
        nll_term = make_nll_term(Rij_symbol,ui_symbol,vj)
        grads = T.grad(nll_term, vj)
        updates = update([grads],[vj],learning_rate = update_algorithms.lr)
        fn = theano.function([Rij_symbol,ui_symbol],[],updates=updates)
        v_update_fns.append(fn)

    print("training pmf...")
    for training_set,lr in cftools.epochsloop(R,U,V,predict_to_5):
        cftools.lr = lr
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            Rij = (Rij - 1.) / (config.max_rating - 1.)



            eij = new_eij()
            gp = g_prime(np.dot(U[i].T,V[j]))
            error_term = (-1./sigma) * eij * gp * V[j]
            regularizer = lambda_u/M * U[i]
            grad = error_term + regularizer
            new_ui = update(U[i], grad)

            eij = new_eij()
            gp = g_prime(np.dot(U[i].T,V[j]))
            error_term = (-1./sigma) * eij * gp * U[i]
            regularizer = lambda_v/N * V[j]
            grad = error_term + regularizer
            new_vj = update(V[j], grad)

            U[i] = new_ui
            V[j] = new_vj

if __name__=="__main__":
    main()

