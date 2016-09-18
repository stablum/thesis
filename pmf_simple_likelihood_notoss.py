#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
import sys
from tqdm import tqdm

# local imports

import movielens
import cftools
import config

def main():
    R,N,M = movielens.small()

    U = np.random.random((config.K,N))
    V = np.random.random((config.K,M))

    def new_eij():
        ret = cftools.rating_error(Rij,U,i,V,j)
        #print "eij",ret
        return ret

    print "training pmf..."
    for training_set,lr in cftools.epochsloop(R,U,V):
        for curr in tqdm(training_set):
            (i,j),Rij = curr
            eij = new_eij()
            new_ui = U[:,i] + lr * eij * V[:,j]
            eij = new_eij()
            new_vj = V[:,j] + lr * eij * U[:,i]
            U[:,i] = new_ui
            V[:,j] = new_vj


if __name__=="__main__":
    main()

