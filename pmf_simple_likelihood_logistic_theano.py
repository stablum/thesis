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
theano.config.on_unused_input='ignore'

update =update_algorithms.get_func()

#g = lambda x:x
g = theano.tensor.nnet.sigmoid

sigma = 1.
sigma_u = 100.
sigma_v = 1000.
#log = print
log = lambda *args: None

def main():
    R,N,M,N_compressed,M_compressed = movielens.latest22m()

    U,V = cftools.UV_vectors_np(R)
    U_t, U_m, U_v = update_algorithms.adam_for(U)
    V_t, V_m, V_v = update_algorithms.adam_for(V)

    def make_predict_to_1(ui,vj):
        prediction = g(T.dot(ui.T,vj))
        return prediction

    def make_predict_to_5(ui,vj):
        prediction = make_predict_to_1(ui,vj)
        ret = (prediction * (config.max_rating - 1. )) + 1.
        return ret

    def make_objective_term(Rij,ui,vj):
        eij = ( Rij - make_predict_to_1(ui,vj) ) ** 2
        ret = 0.5 * 1./(sigma**2) * eij # error term (gaussian centered in the prediction)

        # 0-mean gaussian prior on the latent vector.
        # since this term refers to a specific <ui,vj> tuple, then
        # the update following the prior quantity has to be divided
        # by how many terms (error term) contain that vector
        ret += (0.5/(N_compressed * sigma_u)) * T.sum(ui**2)
        ret += (0.5/(M_compressed * sigma_v)) * T.sum(vj**2)
        return ret

    print("creating update functions..")

    ui_sym = T.dvector('ui')
    vj_sym = T.dvector('vj')
    Rij_sym = T.dscalar('Rij')
    t_prev_sym = T.dscalar('t_prev')
    m_prev_sym = T.dvector('m_prev')
    v_prev_sym = T.dvector('v_prev')

    # instead of calculating a different count of latent vectors of each
    # (other side) latent vector, a global estimate (average) is performed
    nll_term = make_objective_term(Rij_sym,ui_sym,vj_sym)
    grads_ui = T.grad(nll_term, ui_sym)
    grads_vj = T.grad(nll_term, vj_sym)
    updates_kwargs = dict(t_prev=t_prev_sym,m_prev=m_prev_sym,v_prev=v_prev_sym)
    new_for_ui = list(update(ui_sym,grads_ui,**updates_kwargs))
    new_for_vj = list(update(vj_sym,grads_vj,**updates_kwargs))
    common = [ t_prev_sym,m_prev_sym,v_prev_sym,Rij_sym,ui_sym,vj_sym ]
    ui_update_fn = theano.function(common,new_for_ui)
    vj_update_fn = theano.function(common,new_for_vj)
    predict_to_5_fn = theano.function([ui_sym,vj_sym], [make_predict_to_5(ui_sym,vj_sym)])
    predict_to_1_fn = theano.function([ui_sym,vj_sym], [make_predict_to_1(ui_sym,vj_sym)])

    print("training pmf...")
    for training_set,lr in cftools.epochsloop(R,U,V,predict_to_5_fn):
        cftools.lr = lr
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            Rij = (Rij - 1.) / (config.max_rating - 1.)
            #U_t[i], U_m[i], U_v[i], U[i] = ui_update_fn(U_t[i],U_m[i],U_v[i],Rij,U[i],V[j])
            #V_t[j], V_m[j], V_v[j], V[j] = vj_update_fn(V_t[j],V_m[j],V_v[j],Rij,U[i],V[j])
            log("Rij",Rij)
            log("predict_to_1_fn",predict_to_1_fn(U[i],V[j]))
            log("predict_to_5_fn",predict_to_5_fn(U[i],V[j]))
            new_ui, U_t[i], U_m[i], U_v[i] = ui_update_fn(U_t[i],U_m[i],U_v[i],Rij,U[i],V[j])
            log("U[i]",U[i],"new_ui",new_ui,"diff",U[i]-new_ui)

            new_vj, V_t[j], V_m[j], V_v[j] = vj_update_fn(V_t[j],V_m[j],V_v[j],Rij,U[i],V[j])
            log("V[j]",V[j],"new_vj",new_vj,"diff",V[j]-new_vj)

            U[i] = new_ui
            V[j] = new_vj

if __name__=="__main__":
    main()

