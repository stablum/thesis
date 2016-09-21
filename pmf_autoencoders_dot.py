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

update =update_algorithms.get_func()
adam_shared = lasagne.updates.adam # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g = theano.tensor.nnet.sigmoid

sigma = 1.
sigma_u = 100.
sigma_v = 1000.

fc_dim = 10

#log = print
log = lambda *args: None

def main():
    dataset = movielens.load(config.movielens_which)

    U,V = cftools.UV_vectors_np(dataset)
    U_t, U_m, U_v = update_algorithms.adam_for(U)
    V_t, V_m, V_v = update_algorithms.adam_for(V)

    def make_net(input_var):
        l_in = lasagne.layers.InputLayer((1,config.K),input_var=input_var)
        l_fc = lasagne.layers.DenseLayer(l_in,fc_dim,nonlinearity=g)
        net_output = lasagne.layers.get_output(l_fc)
        net_params = lasagne.layers.get_all_params([l_in,l_fc])
        return net_output, net_params

    def make_predict_to_1(ui,vj):
        o_ui,net_ui_params = make_net(ui.reshape((1, config.K)))
        o_ui.name = "o_ui"
        o_vj,net_vj_params = make_net(vj.reshape((1, config.K)))
        o_vj.name = "o_vj"
        dot = T.dot(o_ui,o_vj.T)
        dot.name = "dot"
        prediction = g(dot)
        prediction.name = "prediction"
        print("prediction:",prediction.type,prediction.ndim)
        prediction_scalar = prediction.reshape((),ndim=0)
        prediction_scalar.name = "prediction_scalar"
        print("prediction_scalar:",prediction_scalar.type,prediction_scalar.ndim)
        return prediction_scalar, net_ui_params+net_vj_params

    def make_predict_to_5(predict_to_1_sym):
        ret = (predict_to_1_sym * (config.max_rating - 1. )) + 1.
        return ret

    def make_objective_term(Rij,ui,vj,predict_to_1_sym):
        eij = ( Rij - predict_to_1_sym ) ** 2
        ret = 0.5 * 1./(sigma**2) * eij # error term (gaussian centered in the prediction)

        # 0-mean gaussian prior on the latent feature vector.
        # since this term refers to a specific <ui,vj> tuple, then
        # the update following the prior quantity has to be divided
        # by how many terms (error term) contain that vector
        ret += (0.5/(dataset.N_compressed * sigma_u)) * T.sum(ui**2)
        ret += (0.5/(dataset.M_compressed * sigma_v)) * T.sum(vj**2)
        return ret

    print("creating update functions..")

    ui_sym = T.fvector('ui')
    vj_sym = T.fvector('vj')
    Rij_sym = T.fscalar('Rij')
    t_prev_sym = T.fscalar('t_prev')
    m_prev_sym = T.fvector('m_prev')
    v_prev_sym = T.fvector('v_prev')

    predict_to_1_sym,params = make_predict_to_1(ui_sym,vj_sym)

    # instead of calculating a different count of latent vectors of each
    # (other side) latent vector, a global estimate (average) is performed
    obj_term = make_objective_term(Rij_sym,ui_sym,vj_sym,predict_to_1_sym)

    grads_ui = T.grad(obj_term, ui_sym)
    grads_vj = T.grad(obj_term, vj_sym)
    grads_params = [
        T.grad(obj_term,curr)
        for curr
        in params
    ]
    updates_kwargs = dict(t_prev=t_prev_sym,m_prev=m_prev_sym,v_prev=v_prev_sym)
    new_for_ui = list(update(ui_sym,grads_ui,**updates_kwargs))
    new_for_vj = list(update(vj_sym,grads_vj,**updates_kwargs))
    params_updates = adam_shared(grads_params,params,learning_rate=config.lr_begin)

    common = [ t_prev_sym,m_prev_sym,v_prev_sym,Rij_sym,ui_sym,vj_sym ]
    ui_update_fn = theano.function(common,new_for_ui)
    vj_update_fn = theano.function(common,new_for_vj)
    params_update_fn = theano.function([Rij_sym,ui_sym,vj_sym],[], updates=params_updates)
    predict_to_5_fn = theano.function([ui_sym,vj_sym], [make_predict_to_5(predict_to_1_sym)])
    predict_to_1_fn = theano.function([ui_sym,vj_sym], [predict_to_1_sym])

    def train_with_datapoint(i,j,Rij,lr):
        Rij = (Rij - 1.) / (config.max_rating - 1.)
        log("Rij",Rij)
        log("predict_to_1_fn",predict_to_1_fn(U[i],V[j]))
        log("predict_to_5_fn",predict_to_5_fn(U[i],V[j]))
        new_ui, U_t[i], U_m[i], U_v[i] = ui_update_fn(U_t[i],U_m[i],U_v[i],Rij,U[i],V[j])
        log("U[i]",U[i],"new_ui",new_ui,"diff",U[i]-new_ui)

        new_vj, V_t[j], V_m[j], V_v[j] = vj_update_fn(V_t[j],V_m[j],V_v[j],Rij,U[i],V[j])
        log("V[j]",V[j],"new_vj",new_vj,"diff",V[j]-new_vj)

        U[i] = new_ui
        V[j] = new_vj

        params_update_fn(Rij,U[i],V[j])

    print("training pmf...")
    cftools.mainloop(train_with_datapoint,dataset,U,V,predict_to_5_fn)

if __name__=="__main__":
    main()

