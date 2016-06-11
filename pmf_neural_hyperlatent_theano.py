#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
from tqdm import tqdm
from theano import tensor as T
from theano import function
from theano import pp
import theano
# local imports

import movielens
import cftools
import config
import numutils

#theano_mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'

theano_mode = 'DebugMode'
theano.config.optimizer = 'None'

theano.mode = theano_mode
theano.config.exception_verbosity = 'high'
#theano.config.compute_test_value = 'raise'

def main():
    np.set_printoptions(precision=4, suppress=True)

    sigma = 1.
    sigma_u = 1.
    sigma_v = 1.
    sigma_wu = 1.
    sigma_wv = 1.
    sigma_hu = 1.
    sigma_hv = 1.

    R,N,M = movielens.small()
    U,V = cftools.UV(R)
    trainM = T.dmatrix('trainM')
    cftools.test_value(trainM, np.random.randint(0,6,(3,np.floor(len(R.items())/2))).astype('int32'))

    # dimensionality of the "hyper latent" vectors
    D = config.K * 2
    print "D",D
    # neural network (encoding) weights
    Wu_values = np.random.random((D,config.K))
    Wv_values = np.random.random((D,config.K))
    Wu = theano.shared(Wu_values)
    Wu.name = 'Wu'
    Wv = theano.shared(Wv_values)
    Wv.name = 'Wv'

    # hyper latent vectors
    Hu_values = np.random.random((D,N))
    Hv_values = np.random.random((D,M))
    Hu = theano.shared(Hu_values)
    Hu.name = 'Hu'
    Hv = theano.shared(Hv_values)
    Hv.name = 'Hv'

    def make_sigmoid(x):
        cftools.test_value(x, np.random.random((config.K,1)))
        s = 1 / (1 + T.exp(-x))
        cftools.test_value(s, np.random.random((config.K,1)))
        return s

    def step(curr,_U,_V,_Wu,_Wv,_Hu,_Hv):

        i = T.cast(curr[0],'int32')
        j = T.cast(curr[1],'int32')
        Rij = curr[2]
        Rij.name = 'Rij'
        cftools.test_value(Rij, 4.5)

        def new_eij():
            uivj = T.dot(_U[:,i].T,_V[:,j])
            uivj.name = 'uivj'
            cftools.test_value(uivj, 4.)
            ret = Rij - uivj
            cftools.test_value(ret, 0.5)
            return ret
        hui = _Hu[:,[i]]
        hui.name = 'hui'
        cftools.test_value(hui, np.random.random((D,1)))
        WuHui = T.dot(_Wu.T,hui)
        WuHui.name = 'WuHui'
        cftools.test_value(WuHui, np.random.random((config.K)))
        neural_output_u = make_sigmoid(WuHui)
        cftools.test_value(neural_output_u, np.random.random((config.K,1)))
        neural_output_u.name = 'neural_output_u'
        neural_output_v = make_sigmoid(T.dot(_Wv.T,_Hv[:,[j]]))
        cftools.test_value(neural_output_v, np.random.random((config.K,1)))
        neural_output_v.name = 'neural_output_v'
        grad_neural_u = _U[:,[i]] - neural_output_u
        grad_neural_u.name = 'grad_neural_u'
        grad_neural_v = _V[:,[j]] - neural_output_v
        grad_neural_v.name = 'grad_neural_v'

        eiju = new_eij()
        eiju.name = 'eiju'
        grad = (-1.)/sigma * eiju * _V[:,[j]] + (1./(sigma_u*M)) * grad_neural_u
        grad.name = 'grad_u'
        new_U = cftools.update(_U[:,[i]],grad)

        eijv = new_eij()
        eijv.name = 'eijv'
        grad = (-1.)/sigma * eijv * new_U[:,[i]] + (1./(sigma_v*N)) * grad_neural_v
        grad.name = 'grad_v'
        new_V = cftools.update(_V[:,[j]],grad)

        sigmoid_deriv_u = (neural_output_u) * (1 - neural_output_u)
        sigmoid_deriv_u.name = 'sigmoid_deriv_u'
        cftools.test_value(sigmoid_deriv_u, np.random.random((config.K,1)))
        neural_output_wu_grad = T.outer(
            sigmoid_deriv_u.T, # dimensions:  1*K
            _Hu[:,[i]] # dimensions: 1*D
        ) # output dimensions: K*D
        neural_output_wu_grad.name = 'neural_output_wu_grad'
        cftools.test_value(neural_output_wu_grad, np.random.random((config.K,D)))

        neural_output_hu_grad = T.dot(
            _Wu, # dimensions: D * K
            sigmoid_deriv_u # dimensions: K * 1
        )[:,0] # output dimensions: D

        neural_output_hu_grad.name="neural_output_hu_grad"
        cftools.test_value(neural_output_hu_grad, np.random.random((D)))

        sigmoid_deriv_v = (neural_output_v) * (1 - neural_output_v)
        sigmoid_deriv_v.name = 'sigmoid_deriv_v'
        cftools.test_value(sigmoid_deriv_v, np.random.random((config.K,1)))

        neural_output_wv_grad = T.outer(
            sigmoid_deriv_v.T,
            _Hv[:,[j]]
        )
        neural_output_wv_grad.name = 'neural_output_wv_grad'
        cftools.test_value(neural_output_wv_grad, np.random.random((config.K,D)))

        neural_output_hv_grad = T.dot(
            _Wv,
            sigmoid_deriv_v
        )[:,0]
        neural_output_hv_grad.name="neural_output_hv_grad"
        cftools.test_value(neural_output_hv_grad, np.random.random((D)))

        new_Wu = _Wu
        new_Wv = _Wv
        new_Hu = _Hu
        new_Hv = _Hv
        for k in range(config.K):

            error_u = neural_output_u[k] - new_U[k,i]
            error_u.name = 'error_u'
            cftools.test_value(error_u, 0.5)
            error_v = neural_output_v[k] - new_V[k,j]
            error_v.name = 'error_v'
            cftools.test_value(error_v, 0.5)

            f_prime = neural_output_wu_grad[k,:]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_u * error_u, D) * f_prime * new_Hu[:,k]
            prior_term = 1./sigma_wu * new_Wu[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Wu = cftools.update(new_Wu[:,k],grad)

            f_prime = neural_output_wv_grad[k,:]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_v * error_v,D) * f_prime * new_Hv[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_wv * new_Wv[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Wv = cftools.update(new_Wv[:,k],grad)

            f_prime = neural_output_hu_grad
            error_term = T.tile(1./sigma_u * error_u, D) * f_prime * new_Wu[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hu * new_Hu[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Hu = cftools.update(new_Hu[:,k],grad)

            f_prime = neural_output_hv_grad
            error_term = T.tile(1./sigma_v * error_v, D) * f_prime * new_Wv[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hv * new_Hv[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Hv = cftools.update(new_Hv[:,k],grad)

        return {
            _U:new_U,
            _V:new_V,
            _Wu:new_Wu,
            _Wv:new_Wv,
            _Hu:new_Hu,
            _Hv:new_Hv
        }

    values, updates = theano.scan(
        step,
        sequences=[trainM],
        non_sequences=[U,V,Wu,Wv,Hu,Hv]
    )

    scan_fn = function(
        [trainM],
        [],
        updates=updates,
        mode=theano_mode
    )

    print "training pmf with hyperlatent neural vectors..."
    for training_set in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        scan_fn(training_set_matrix)

if __name__=="__main__":
    main()
