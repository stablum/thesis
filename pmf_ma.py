#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
from tqdm import tqdm
tqdm.monitor_interval = 0
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

def main():
    np.set_printoptions(precision=4, suppress=True)

    sigma = 1.
    sigma_u = 1.
    sigma_v = 1.
    sigma_wu = 1.
    sigma_wv = 1.
    sigma_hu = 1.
    sigma_hv = 1.
    sigma_m_u = 20.
    sigma_m_v = 20.
    sigma_a_u = 20.
    sigma_a_v = 20.
    R,N,M = movielens.small()
    U,V = cftools.UV_np(R)

    # dimensionality of the "hyper latent" vectors
    D = config.K * 2
    print "D",D
    # neural network (encoding) weights
    Wu = np.random.random((D,config.K))
    Wv = np.random.random((D,config.K))
    m_u = np.random.random((config.K))
    m_v = np.random.random((config.K))
    a_u = np.random.random((config.K))
    a_v = np.random.random((config.K))

    # hyper latent vectors
    Hu = np.random.random((D,N))
    Hv = np.random.random((D,M))

    def make_sigmoid(x):
        cftools.test_value(x, np.random.random((config.K,1)))
        s = 1 / (1 + T.exp(-x))
        cftools.test_value(s, np.random.random((config.K,1)))
        return s

    def step(_Rij,_ui,_vj,_Wu,_Wv,_hui,_hvj,_m_u,_m_v,_a_u,_a_v):

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
        neural_output_u.name = 'neural_output_u'
        cftools.test_value(neural_output_u, np.random.random((config.K,1)))
        computed_u = neural_output_u * _m_u + _a_u
        neural_output_v = make_sigmoid(T.dot(_Wv.T,_hvj))
        cftools.test_value(neural_output_v, np.random.random((config.K,1)))
        neural_output_v.name = 'neural_output_v'
        computed_v = neural_output_v * _m_v + _a_v
        grad_computed_u = _ui - computed_u
        grad_computed_u.name = 'grad_computed_u'
        grad_computed_v = _vj - computed_v
        grad_computed_v.name = 'grad_computed_v'

        eiju = new_eij()
        eiju.name = 'eiju'
        grad = (-1.)/sigma * eiju * _vj + (1./(sigma_u*M)) * grad_computed_u
        grad.name = 'grad_u'
        new_ui = cftools.update(_ui,grad)

        eijv = new_eij()
        eijv.name = 'eijv'
        grad = (-1.)/sigma * eijv * _ui + (1./(sigma_v*N)) * grad_computed_v
        grad.name = 'grad_v'
        new_vj = cftools.update(_vj,grad)

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

            error_u = grad_computed_u[k]
            error_u.name = 'error_u'
            cftools.test_value(error_u, 0.5)
            error_v = grad_computed_v[k]
            error_v.name = 'error_v'
            cftools.test_value(error_v, 0.5)

            #W*
            f_prime = sigmoid_deriv_u[k]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_u * error_u * (-1) * _m_u[k], D) * f_prime * _hui[k]
            prior_term = (1./(N*M*sigma_wu)) * _Wu[:,k]
            #cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Wu = cftools.update(_Wu[:,k],grad)

            f_prime = sigmoid_deriv_v[k]
            cftools.test_value(f_prime, np.random.random((D)))
            error_term = T.tile(1./sigma_v * error_v * (-1) * _m_v[k],D) * f_prime * _hvj[k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = (1./(N*M*sigma_wv)) * _Wv[:,k]
            #cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_Wv = cftools.update(_Wv[:,k],grad)

            #H*
            f_prime = sigmoid_deriv_u[k]
            error_term = 1./sigma_u * T.tile(error_u,D) * (-1) * _m_u[k] *  f_prime * _Wu[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hu * _hui[k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_hui = cftools.update(_hui[k],grad)

            f_prime = sigmoid_deriv_v[k]
            error_term = 1./sigma_v * T.tile(error_v,D) * (-1) *_m_v[k] * f_prime * _Wv[:,k]
            cftools.test_value(error_term, np.random.random((D)))
            prior_term = 1./sigma_hv * _hvj[k]
            cftools.test_value(prior_term, np.random.random((D)))
            grad = error_term + prior_term
            new_hvj = cftools.update(_hvj[k],grad)

        #m*
        error_term = 1./sigma * grad_computed_u * (-1) * neural_output_u[k]
        prior_term = (1./(sigma_m_u * N * M)) * _m_u
        grad = error_term + prior_term
        new_m_u = cftools.update(_m_u,grad)

        error_term = 1./sigma * grad_computed_v * (-1) * neural_output_v[k]
        prior_term = (1./(sigma_m_v * N * M)) * _m_v
        grad = error_term + prior_term
        new_m_v = cftools.update(_m_v,grad)

        #a*
        error_term = 1./sigma * grad_computed_u * (-1)
        prior_term = (1./(sigma_a_u * N * M))*_a_u
        grad = error_term + prior_term
        new_a_u = cftools.update(_a_u,grad)

        error_term = 1./sigma * grad_computed_v * (-1) #FIXME: check (-1) in derivation
        prior_term = (1./(sigma_a_v * N * M))*_a_v
        grad = error_term + prior_term
        new_a_v = cftools.update(_a_v,grad)

        return collections.OrderedDict([
            ('ui',new_ui),
            ('vj',new_vj),
            ('Wu',new_Wu),
            ('Wv',new_Wv),
            ('hui',new_hui),
            ('hvj',new_hvj),
            ('m_u',new_m_u),
            ('m_v',new_m_v),
            ('a_u',new_a_u),
            ('a_v',new_a_v),
        ])

    Rij = T.dscalar('Rij')
    ui = T.dvector('ui')
    vj = T.dvector('vj')
    hui = T.dvector('hui')
    hvj = T.dvector('hvj')
    Wu_ = T.dmatrix('Wu')
    Wv_ = T.dmatrix('Wv')
    m_u_ = T.dvector('m_u')
    m_v_ = T.dvector('m_v')
    a_u_ = T.dvector('a_u')
    a_v_ = T.dvector('a_v')
    cftools.lr = T.dscalar('lr')
    new = step(Rij,ui,vj,Wu_,Wv_,hui,hvj,m_u_,m_v_,a_u_,a_v_)

    step_fn = function(
        [cftools.lr,Rij,ui,vj,Wu_,Wv_,hui,hvj,m_u_,m_v_,a_u_,a_v_],
        new.values()
    )

    print "training pmf with hyperlatent neural vectors..."
    for training_set,_lr in cftools.epochsloop(R,U,V):
        training_set_matrix = cftools.create_training_set_matrix(training_set)
        print("\ndatapoints loop..")
        for i,j,_Rij in tqdm(training_set_matrix):
            i = int(i)
            j = int(j)
            _ui = U[:,i]
            _vj = V[:,j]
            _hui = Hu[:,i]
            _hvj = Hv[:,j]
            _ui,_vj,Wu,Wv,_hui,_hvj,m_u,m_v,a_u,a_v = step_fn(_lr,_Rij,_ui,_vj,Wu,Wv,_hui,_hvj,m_u,m_v,a_u,a_v)
            U[:,i] = _ui
            V[:,j] = _vj
            Hu[:,i] = _hui
            Hv[:,j] = _hvj
        print 'Wu',Wu
        print 'Wv',Wv
        print 'm_u',m_u
        print 'm_v',m_v
        print 'a_u',a_u
        print 'a_v',a_v

if __name__=="__main__":
    main()
