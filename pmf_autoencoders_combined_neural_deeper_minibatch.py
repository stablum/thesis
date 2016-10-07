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
hid_dim = 100
#log = print
log = lambda *args: None

def main():
    dataset = movielens.load(config.movielens_which)

    U,V = cftools.UV_vectors_np(dataset)
    U_t, U_m, U_v = update_algorithms.adam_for(U)
    V_t, V_m, V_v = update_algorithms.adam_for(V)

    def make_net(input_var,in_dim,hid_dim,out_dim):
        l_in = lasagne.layers.InputLayer((None,in_dim),input_var=input_var)
        l_hid = lasagne.layers.DenseLayer(l_in,hid_dim,nonlinearity=g)
        l_out = lasagne.layers.DenseLayer(l_hid,out_dim,nonlinearity=g)
        net_output = lasagne.layers.get_output(l_out)
        net_params = lasagne.layers.get_all_params([l_in,l_hid,l_out])
        return net_output, net_params

    def make_predict_to_1(ui,vj):
        o_ui,net_ui_params = make_net(ui,config.K,hid_dim,fc_dim)
        o_ui.name = "o_ui"
        o_vj,net_vj_params = make_net(vj,config.K,hid_dim,fc_dim)
        o_vj.name = "o_vj"
        combined = T.concatenate([o_ui,o_vj],axis=0)
        combined.name = "combined"
        prediction,net_combined_params = make_net(combined,2*config.K,hid_dim,1)
        prediction.name = "prediction"
        print("prediction:",prediction.type,prediction.ndim)
        prediction_scalar = prediction.reshape((),ndim=0)
        prediction_scalar.name = "prediction_scalar"
        print("prediction_scalar:",prediction_scalar.type,prediction_scalar.ndim)
        return prediction_scalar, net_ui_params+net_vj_params+net_combined_params

    def make_predict_to_5(predict_to_1_sym):
        ret = (predict_to_1_sym * (config.max_rating - 1. )) + 1.
        return ret

    def make_objective_term(ui_mb,vj_mb,Rij_mb,predict_to_1_sym):
        eij = ( Rij_mb - predict_to_1_sym ) ** 2
        ret = 0.5 * 1./(sigma**2) * eij # error term (gaussian centered in the prediction)

        # 0-mean gaussian prior on the latent feature vector.
        # since this term refers to a specific <ui,vj> tuple, then
        # the update following the prior quantity has to be divided
        # by how many terms (error term) contain that vector
        ret += (0.5/(dataset.N_compressed * sigma_u)) * T.sum(ui_mb**2,axis=1)
        ret += (0.5/(dataset.M_compressed * sigma_v)) * T.sum(vj_mb**2,axis=1)
        ret = T.sum(ret,axis=0)
        return ret

    print("creating update functions..")

    ui_mb_sym = T.fmatrix('ui_mb')
    vj_mb_sym = T.fmatrix('vj_mb')
    Rij_mb_sym = T.fvector('Rij_mb')

    t_mb_prev_sym = T.fvector('t_mb_prev')
    m_mb_prev_sym = T.fmatrix('m_mb_prev')
    v_mb_prev_sym = T.fmatrix('v_mb_prev')

    predict_to_1_sym,params = make_predict_to_1(ui_mb_sym,vj_mb_sym)

    # instead of calculating a different count of latent vectors of each
    # (other side) latent vector, a global estimate (average) is performed
    obj_term = make_objective_term(ui_mb_sym,vj_mb_sym,Rij_mb_sym,predict_to_1_sym)

    grads_ui = T.grad(obj_term, ui_mb_sym)
    grads_vj = T.grad(obj_term, vj_mb_sym)
    grads_params = [
        T.grad(obj_term,curr)
        for curr
        in params
    ]
    updates_kwargs = dict(t_prev=t_mb_prev_sym,m_prev=m_mb_prev_sym,v_prev=v_mb_prev_sym)
    new_for_ui = list(update(ui_mb_sym,grads_ui,**updates_kwargs))
    new_for_vj = list(update(vj_mb_sym,grads_vj,**updates_kwargs))
    params_updates = adam_shared(grads_params,params,learning_rate=config.lr_begin)

    common = [ t_mb_prev_sym,m_mb_prev_sym,v_mb_prev_sym,Rij_mb_sym,ui_mb_sym,vj_mb_sym ]
    ui_update_fn = theano.function(common,new_for_ui)
    vj_update_fn = theano.function(common,new_for_vj)
    params_update_fn = theano.function([Rij_mb_sym,ui_mb_sym,vj_mb_sym],[], updates=params_updates)
    predict_to_5_fn = theano.function([ui_mb_sym,vj_mb_sym], [make_predict_to_5(predict_to_1_sym)])
    predict_to_5_fn.name="predict_to_5_fn"
    predict_to_1_fn = theano.function([ui_mb_sym,vj_mb_sym], [predict_to_1_sym])
    predict_to_1_fn.name="predict_to_1_fn"

    ui_mb_l = []
    vj_mb_l = []
    Rij_mb_l = []
    U_t_mb_l = []
    U_m_mb_l = []
    U_v_mb_l = []
    V_t_mb_l = []
    V_m_mb_l = []
    V_v_mb_l = []
    def train_with_datapoint(i,j,Rij,lr):

        ui_mb_l.append(U[i])
        vj_mb_l.append(V[j])
        Rij_mb_l.append(Rij)
        U_t_mb_l.append(U_t[i])
        U_m_mb_l.append(U_m[i])
        U_v_mb_l.append(U_v[i])
        V_t_mb_l.append(V_t[j])
        V_m_mb_l.append(V_m[j])
        V_v_mb_l.append(V_v[j])
        if len(ui_mb_l) >= config.minibatch_length:
            ui_mb = np.vstack(ui_mb_l)
            vj_mb = np.vstack(vj_mb_l)
            Rij_mb = np.vstack(Rij_mb_l)

            U_t_mb = np.vstack(U_t_mb_l )
            U_m_mb = np.vstack(U_m_mb_l )
            U_v_mb = np.vstack(U_v_mb_l )
            V_t_mb = np.vstack(V_t_mb_l )
            V_m_mb = np.vstack(V_m_mb_l )
            V_v_mb = np.vstack(V_v_mb_l )

            Rij_mb = (Rij_mb - 1.) / (config.max_rating - 1.)
            log("Rij_mb",Rij_mb)
            log("predict_to_1_fn",predict_to_1_fn(ui_mb,vj_mb))
            log("predict_to_5_fn",predict_to_5_fn(ui_mb,vj_mb))
            new_ui_mb, new_U_t_mb, new_U_m_mb, new_U_v_mb = ui_update_fn(U_t_mb,U_m_mb,U_v_mb,Rij_mb,ui_mb,vj_mb)
            log("ui_mb",ui_mb,"new_ui_mb",new_ui_mb,"diff",ui_mb-new_ui_mb)

            new_vj_mb, new_V_t_mb, new_V_m_mb, new_V_v_mb = vj_update_fn(V_t_mb,V_m_mb,V_v_mb,Rij_mb,ui_mb,vj_mb)
            log("vj_mb",vj_mb,"new_vj_mb",new_vj_mb,"diff",vj_mb-new_vj)
            params_update_fn(ui_mb,vj_mb,Rij_mb)

            ui_mb_l = []
            vj_mb_l = []
            Rij_mb_l = []

            U_t_mb_l = []
            U_m_mb_l = []
            U_v_mb_l = []
            V_t_mb_l = []
            V_m_mb_l = []
            V_v_mb_l = []
    print("training pmf...")
    cftools.mainloop(train_with_datapoint,dataset,U,V,predict_to_5_fn)

if __name__=="__main__":
    main()

