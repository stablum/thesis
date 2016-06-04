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
import numutils

def main():
    np.set_printoptions(precision=4, suppress=True)

    sigma = 1.
    sigma_u = 1.
    sigma_v = 1.
    sigma_wu = 1.
    sigma_wv = 1.

    R = movielens.small()

    N = R.shape[0]
    M = R.shape[1]
    U = np.random.random((config.K,N))
    V = np.random.random((config.K,M))

    # dimensionality of the "hyper latent" vectors
    D = config.K * 2

    # neural network (encoding) weights
    Wu = np.random.random((D,config.K))
    Wv = np.random.random((D,config.K))

    # hyper latent vectors
    Hu = np.random.random((D,N))
    Hv = np.random.random((D,M))

    def new_eij():
        ret = cftools.rating_error(Rij,U,i,V,j)
        #print "eij",ret
        return ret

    training_set, testing_set = cftools.split_sets(R)

    print "training pmf with hyperlatent neural vectors..."
    for _ in tqdm(range(config.n_epochs)):
        random.shuffle(training_set)
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            neural_output_v = numutils.sigmoid(np.dot(Wv.T,Hv[:,j]))
            grad_neural = V[:,j] - neural_output_v
            eijv = new_eij()
            grad = 1./sigma * eijv * U[:,i] + (sigma_v/N) * grad_neural
            V[:,j] = V[:,j] + config.lr * grad

            neural_output_u = numutils.sigmoid(np.dot(Wu.T,Hu[:,i]))
            grad_neural = U[:,i] - neural_output_u
            eijv = new_eij()
            grad = 1./sigma * eijv * V[:,j] + (sigma_u/M) * grad_neural
            U[:,i] = U[:,i] + config.lr * grad

            sigmoid_deriv_v = (neural_output_v) * (1 - neural_output_v)
            neural_output_wv_grad = np.dot(
                np.expand_dims(sigmoid_deriv_v,1),
                np.expand_dims(Hv[:,j],0)
            )

            neural_output_hv_grad = np.dot(
                sigmoid_deriv_v,
                Wv.T
            )

            sigmoid_deriv_u = (neural_output_u) * (1 - neural_output_u)
            neural_output_wu_grad = np.dot(
                np.expand_dims(sigmoid_deriv_u,1), # dimensions: K
                np.expand_dims(Hu[:,i],0) # dimensions: D
            ) # output dimensions: K*D

            neural_output_hu_grad = np.dot(
                sigmoid_deriv_u, # dimensions: K
                Wu.T # dimensions: K * D
            ) # output dimensions: D

            for k in range(config.K):

                error_v = neural_output_v[k] - V[k,j]
                error_u = neural_output_u[k] - U[k,i]

                f_prime = neural_output_wv_grad[k,:]
                error_term = 1./sigma_v * error_v * f_prime
                prior_term = 1./sigma_wv * Wv[:,k]
                grad = error_term + prior_term
                Wv[:,k] = Wv[:,k] + config.lr * grad

                f_prime = neural_output_wu_grad[k,:]
                error_term = 1./sigma_u * error_u * f_prime
                prior_term = 1./sigma_wu * Wu[:,k]
                grad = error_term + prior_term
                Wu[:,k] = Wu[:,k] + config.lr * grad

                f_prime = neural_output_hv_grad
                error_term = 1./sigma_v * error_v * f_prime
                prior_term = 1./sigma_wv * Wv[:,k]
                grad = error_term + prior_term
                Hv[:,k] = Hv[:,k] + config.lr * grad

                f_prime = neural_output_hu_grad
                error_term = 1./sigma_u * error_u * f_prime
                prior_term = 1./sigma_wu * Wu[:,k]
                grad = error_term + prior_term
                Hu[:,k] = Hu[:,k] + config.lr * grad

        print "training RMSE: ",cftools.rmse(training_set,U,V)
        print "testing RMSE: ",cftools.rmse(testing_set,U,V)

if __name__=="__main__":
    main()
