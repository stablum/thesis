#!/usr/bin/env python3

import scipy
import ipdb
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

update = update_algorithms.get_func()

g = nu.sigmoid
g_prime = lambda x : g(x) * (1 - g(x))

sigma = 1.

def main():
    R,N,M = movielens.small()

    U = [ at.NumpyArrayAdam.random(config.K) * 0.01 for _ in range(N)]
    V = [ at.NumpyArrayAdam.random(config.K) * 0.01 for _ in range(M)]
    lambda_u = 20.
    lambda_v = 20.

    def predict(ui,vj):
        prediction = g(np.dot(ui.T,vj))
        return prediction

    def predict_to_5(ui,vj):
        prediction = predict(ui,vj)
        ret = (prediction * (config.max_rating - 1. )) + 1.
        return ret

    def new_eij():
        ui = U[i]
        vj = V[j]
        prediction = predict(ui,vj)
        ret = Rij - prediction
        #print "eij",ret
        return ret

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

