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
import collections
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
    U,V = cftools.UV_np(R)

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
    Hu = np.random.random((D,N))
    Hv = np.random.random((D,M))

    def make_sigmoid(x):
        cftools.test_value(x, np.random.random((config.K,1)))
        s = 1 / (1 + T.exp(-x))
        cftools.test_value(s, np.random.random((config.K,1)))
        return s

    def step(_Rij,_ui,_vj,_Wu,_Wv,_hui,_hvj):

        _Rij.name = 'Rij'
        cftools.test_value(_Rij, 4.5)

        def new_eij():
            uivj = T.dot(_ui.T,_vj)
            uivj.name = 'uivj'
            cftools.test_value(uivj, 4.)
            ret = Rij - uivj
            cftools.test_value(ret, 0.5)
            return ret

        cftools.test_value(_hui, np.random.random((D,1)))
        _hui.name = '_hui'
        WuHui = T.dot(_Wu.T,_hui)
        WuHui.name = 'WuHui'
        cftools.test_value(WuHui, np.random.random((config.K)))
        neural_output_u = make_sigmoid(WuHui)
        cftools.test_value(neural_output_u, np.random.random((config.K,1)))
        neural_output_u.name = 'neural_output_u'
        neural_output_v = make_sigmoid(T.dot(_Wv.T,_hvj))
        cftools.test_value(neural_output_v, np.random.random((config.K,1)))
        neural_output_v.name = 'neural_output_v'
        grad_neural_u = _ui - neural_output_u
        grad_neural_u.name = 'grad_neural_u'
        grad_neural_v = _vj - neural_output_v
        grad_neural_v.name = 'grad_neural_v'

        eiju = new_eij()
        eiju.name = 'eiju'
        grad = (-1.)/sigma * eiju * _vj + (1./(sigma_u*M)) * grad_neural_u
        grad.name = 'grad_u'
        _ui = cftools.update(_ui,grad)

        eijv = new_eij()
        eijv.name = 'eijv'
        grad = (-1.)/sigma * eijv * _ui + (1./(sigma_v*N)) * grad_neural_v
        grad.name = 'grad_v'
        _vj = cftools.update(_vj,grad)

        sigmoid_deriv_u = (neural_output_u) * (1 - neural_output_u)
        sigmoid_deriv_u.name = 'sigmoid_deriv_u'
        cftools.test_value(sigmoid_deriv_u, np.random.random((config.K,1)))
        neural_output_wu_grad = T.outer(
            sigmoid_deriv_u.T, # dimensions:  1*K
            _hui # dimensions: 1*D
        ) # output dimensions: K*D
        neural_output_wu_grad.name = 'neural_output_wu_grad'
        cftools.test_value(neural_output_wu_grad, np.random.random((config.K,D)))

        neural_output_hu_grad = T.dot(
            _Wu, # dimensions: D * K
            sigmoid_deriv_u # dimensions: K * 1
        ) # output dimensions: D

        neural_output_hu_grad.name="neural_output_hu_grad"
        cftools.test_value(neural_output_hu_grad, np.random.random((D)))

        sigmoid_deriv_v = (neural_output_v) * (1 - neural_output_v)
        sigmoid_deriv_v.name = 'sigmoid_deriv_v'
        cftools.test_value(sigmoid_deriv_v, np.random.random((config.K,1)))

        neural_output_wv_grad = T.outer(
            sigmoid_deriv_v.T,
            _hvj
        )
        neural_output_wv_grad.name = 'neural_output_wv_grad'
        cftools.test_value(neural_output_wv_grad, np.random.random((config.K,D)))

        neural_output_hv_grad = T.dot(
            _Wv,
            sigmoid_deriv_v
        )
        neural_output_hv_grad.name="neural_output_hv_grad"
        cftools.test_value(neural_output_hv_grad, np.random.random((D)))

        for k in range(config.K):

            error_u = neural_output_u[k] - _ui[k]
            error_u.name = 'error_u'
            cftools.test_value(error_u, 0.5)
            error_v = neural_output_v[k] - _vj[k]
            error_v.name = 'error_v'
            cftools.test_value(error_v, 0.5)

            f_prime = neural_output_wu_grad[k,:]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_u * error_u, D) * f_prime * _hui[k]
            prior_term = 1./sigma_wu * _Wu[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            _Wu = cftools.update(_Wu[:,k],grad)

            f_prime = neural_output_wv_grad[k,:]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_v * error_v,D) * f_prime * _hvj[k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_wv * _Wv[:,k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            _Wv = cftools.update(_Wv[:,k],grad)

            f_prime = neural_output_hu_grad
            error_term = T.tile(1./sigma_u * error_u, D) * f_prime * _Wu[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hu * _hui[k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            _hui = cftools.update(_hui[k],grad)

            f_prime = neural_output_hv_grad
            error_term = T.tile(1./sigma_v * error_v, D) * f_prime * _Wv[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hv * _hvj[k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            _hvj = cftools.update(_hvj[k],grad)

        return collections.OrderedDict([
            ('ui',_ui),
            ('vj',_vj),
            ('hui',_hui),
            ('hvj',_hvj),
        ])

    Rij = T.dscalar('Rij')
    ui = T.dvector('ui')
    vj = T.dvector('vj')
    hui = T.dvector('hui')
    hvj = T.dvector('hvj')
    new = step(Rij,ui,vj,Wu,Wv,hui,hvj)

    step_fn = function(
        [Rij,ui,vj,hui,hvj],
        new.values()
    )

    print "training pmf with hyperlatent neural vectors..."
    for training_set in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        print("\ndatapoints loop..")
        for i,j,_Rij in tqdm(training_set_matrix):
            i = int(i)
            j = int(j)
            _ui = U[:,i]
            _vj = V[:,j]
            _hui = Hu[:,i]
            _hvj = Hv[:,j]
            _ui,_vj,_hui,_hvj = step_fn(_Rij,_ui,_vj,_hui,_hvj)
            U[:,i] = _ui
            V[:,j] = _vj
            Hu[:,i] = _hui
            Hv[:,j] = _hvj

if __name__=="__main__":
    main()
