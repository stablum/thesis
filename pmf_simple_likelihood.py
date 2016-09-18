#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
from tqdm import tqdm

# local imports

import movielens
import cftools

def main():
    K=100
    lr = 0.001
    n_epochs = 1000
    R = movielens.small()

    U = np.random.random((K,R.shape[0]))
    V = np.random.random((K,R.shape[1]))

    def new_eij():
        ret = cftools.rating_error(Rij,U,i,V,j)
        return ret

    training_set, testing_set = cftools.split_sets(R)

    print "training pmf..."
    for _ in tqdm(range(n_epochs)):
        random.shuffle(training_set)
        for curr in tqdm(training_set):
            (i,j),Rij = curr
            toss = np.random.randint(3)
            if toss == 0:
                pass
            elif toss == 1:
                eij = new_eij()
                cftools.update(V[:,j], eij * U[:,i])
            elif toss == 2:
                eij = new_eij()
                cftools.update(U[:,i], eij * V[:,j])


        print "training RMSE: ",cftools.rmse(training_set,U,V)
        print "testing RMSE: ",cftools.rmse(testing_set,U,V)

if __name__=="__main__":
    main()
