#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
from tqdm import tqdm

# local imports

import movielens
import cftools
import config

def wrong(x):
    return ((not np.isfinite(x)) or np.isnan(x) or x>10000. or x<-10000.)

def main():
    np.set_printoptions(precision=4, suppress=True)

    lambda_v = 1.
    lambda_u = 1.

    mu_u = np.random.random(config.K)*10. - 5.
    mu_v = np.random.random(config.K)*10. - 5.

    R = movielens.small()

    N = R.shape[0]
    M = R.shape[1]
    U = np.random.random((config.K,N))
    V = np.random.random((config.K,M))

    def new_eij():
        ret = cftools.rating_error(Rij,U,i,V,j)
        #print "eij",ret
        return ret

    print "training pmf with mu hyperpriors..."
    for training_set in cftools.epochsloop(R,U,V):
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            eijv = new_eij()
            grad = -(eijv * U[:,i]) + (1./N) * lambda_v * (V[:,j] - mu_v)
            cftools.update(V[:,j], grad)

            eiju = new_eij()
            grad = -(eiju * V[:,j]) + (1./M) * lambda_u * (U[:,i] - mu_u)
            cftools.update(U[:,i], grad)

            grad = lambda_v * (mu_v - V[:,j])
            cftools.update(mu_v, grad)

            grad = lambda_u * (mu_u - U[:,i])
            cftools.update(mu_u, grad)

        print "mu_v",mu_v
        print "mu_u",mu_u

if __name__=="__main__":
    main()
