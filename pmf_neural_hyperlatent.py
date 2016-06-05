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
    sigma_hu = 1.
    sigma_hv = 1.

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

    print "training pmf with hyperlatent neural vectors..."
    for training_set in cftools.epochsloop(R,U,V):
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            neural_output_v = numutils.sigmoid(np.dot(Wv.T,Hv[:,j]))
            grad_neural = V[:,j] - neural_output_v
            eijv = new_eij()
            grad = (-1.)/sigma * eijv * U[:,i] + (1./(sigma_v*N)) * grad_neural
            cftools.update(V[:,j],grad)

            neural_output_u = numutils.sigmoid(np.dot(Wu.T,Hu[:,i]))
            grad_neural = U[:,i] - neural_output_u
            eijv = new_eij()
            grad = (-1.)/sigma * eijv * V[:,j] + (1./(sigma_u*M)) * grad_neural
            cftools.update(U[:,i],grad)

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
                error_term = 1./sigma_v * error_v * f_prime * Hv[:,k]
                prior_term = 1./sigma_wv * Wv[:,k]
                grad = error_term + prior_term
                cftools.update(Wv[:,k],grad)

                f_prime = neural_output_wu_grad[k,:]
                error_term = 1./sigma_u * error_u * f_prime * Hu[:,k]
                prior_term = 1./sigma_wu * Wu[:,k]
                grad = error_term + prior_term
                cftools.update(Wu[:,k],grad)

                f_prime = neural_output_hv_grad
                error_term = 1./sigma_v * error_v * f_prime * Wv[:,k]
                prior_term = 1./sigma_hv * Hv[:,k]
                grad = error_term + prior_term
                cftools.update(Hv[:,k],grad)

                f_prime = neural_output_hu_grad
                error_term = 1./sigma_u * error_u * f_prime * Wu[:,k]
                prior_term = 1./sigma_hu * Hu[:,k]
                grad = error_term + prior_term
                cftools.update(Hu[:,k],grad)

if __name__=="__main__":
    main()
