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
import activation_functions
import update_algorithms
import model_build

update =update_algorithms.get_func()
adam_shared = lasagne.updates.adam # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g_in = activation_functions.get(config.g_in)
g_rij = activation_functions.get(config.g_rij)
g_latent = activation_functions.get('linear')
g_feat = activation_functions.get('linear')

sigma = 1.
sigma_u = 100.
sigma_v = 1000.

chan_out_dim = config.chan_out_dim
hid_dim = config.hid_dim
#log = print
log = lambda *args: print(*args)#None

class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr

class VaeChan(object):

    def __init__(self,r_sym, feat_distr_sym,name):
        """
        r_sym represents a single rating
        feat_distr_sym is either a distribution on ui or vj
        """

        self.feat_sample_sym = model_build.reparameterization_trick(feat_distr_sym,name,dim=config.K)
        feat_dim = 1 + config.K

        x_sym = T.concatenate([r_sym,self.feat_sample_sym],axis=1)
        x_sym.name = "x_"+name

        # variational approximation 'q' network
        (
            recognition_layers,#(l_sampling,l_out_mu,l_out_log_sigma,l_merge_distr),
            __ignored,
            q_phi_params,
            q_regularizer,
            l_sampling
        ) = model_build.make_net(
            x_sym,
            feat_dim,
            hid_dim,
            config.K,
            "vae_q_"+name,
            g_in,
            g_latent,
            stochastic_output=True,
            return_only_sample=False # because the sample is done externally to the net
        )

        # p_theta(r,feat1|feat2) "p(x|z)"
        p_theta_hid_layers = model_build.make_hid_part(
            l_sampling,
            hid_dim,
            "vae_p_"+name,
            g_in
        )
        r_out_layer = lasagne.layers.DenseLayer(
            p_theta_hid_layers[-1],
            1,
            nonlinearity=g_rij,
            name=name+"_out"
        )
        feat_out_layer = lasagne.layers.DenseLayer(
            p_theta_hid_layers[-1],
            config.K,
            nonlinearity=g_feat,
            name=name+"_out"
        )

        p_theta_layers = p_theta_hid_layers + [r_out_layer,feat_out_layer]
        p_theta_params = lasagne.layers.get_all_params(p_theta_layers)

        outputting_layers = recognition_layers + [r_out_layer, feat_out_layer]
        (
            self.sample_lea,
            self.mu_lea,
            self.log_sigma_lea,
            self.latent_distr_lea,
            self.r_out_lea,
            self.feat_out_lea
        ) = lasagne.layers.get_output(outputting_layers,deterministic=False)

        (
            self.sample_det,
            self.mu_det,
            self.log_sigma_det,
            self.latent_distr_det,
            self.r_out_det,
            self.feat_out_det
        ) = lasagne.layers.get_output(outputting_layers,deterministic=True)

        self.sigma_lea = T.exp(self.log_sigma_lea)
        self.r_sym = r_sym
        self.feat_distr_sym = feat_distr_sym
        self.q_phi_params = q_phi_params
        self.p_theta_params = p_theta_params

    @cached_property
    def latent_distr_lea(self):
        ret = T.concatenate([self.mu_lea,self.sigma_lea],axis=1)
        return ret

    @cached_property
    def obj(self):
        z_sample = self.sample_lea
        z_mu = self.mu_lea
        z_sigma = self.sigma_lea
        x_sigma = 1
        z_dim = config.K
        x_orig = T.concatenate([self.r_sym,self.feat_sample_sym],axis=1)
        x_out = T.concatenate([self.r_out_lea,self.feat_out_lea],axis=1)

        z_sigma_fixed = z_sigma
        z_sigma_inv = 1/(z_sigma_fixed)
        det_z_sigma = T.prod(z_sigma)
        C = 1./(T.sqrt(((2*np.pi)**z_dim) * det_z_sigma))
        log_q_z_given_x = - 0.5*T.dot(z_sigma_inv, ((z_sample-z_mu)**2).T) + T.log(C) # log(C) can be omitted
        q_z_given_x = C * T.exp(log_q_z_given_x)
        log_p_x_given_z = -(1/(x_sigma))*(((x_orig-x_out)**2).sum()) # because p(x|z) is gaussian
        log_p_z = - (z_sample**2).sum() # gaussian prior with mean 0 and cov I
        #reconstruction_error_const = (0.5*(x_dim*np.log(np.pi)+1)).astype('float32')
        reconstruction_error_proper = 0.5*T.sum((x_orig-x_out)**2)
        reconstruction_error = reconstruction_error_proper #+ reconstruction_error_const
        regularizer = self.kl_normal_diagonal_vs_unit(z_mu,z_sigma,z_dim)
        obj = reconstruction_error + regularizer
        obj_scalar = obj.reshape((),ndim=0)
        return obj_scalar

    def kl_normal_diagonal_vs_unit(self,mu1,sigma_diag1,dim):
        # KL divergence of a multivariate normal over a normal with 0 mean and I cov
        log_det1 = T.sum(T.log(sigma_diag1)) #sum log is better than log prod
        mu_diff = -mu1
        ret = 0.5 * (
            - log_det1
            #- dim
            + T.sum(sigma_diag1) # trace
            + T.sum(mu_diff**2) # mu^T mu
        )
        return ret

    def kl_for_r(self,r_train):
        ret = 0.5 * (self.r_out_lea - r_train)**2
        return ret

class Model(object):

    def __init__(self):

        self.ui_mb_sym = T.fmatrix('ui_mb')
        self.vj_mb_sym = T.fmatrix('vj_mb')
        self.Rij_mb_sym = T.fmatrix('Rij_mb')

        self.vae_chan_latent_u = VaeChan(self.Rij_mb_sym,self.vj_mb_sym,"vaeu")
        self.vae_chan_latent_v = VaeChan(self.Rij_mb_sym,self.ui_mb_sym,"vaev")

    @cached_property
    def reconstruction_obj(self):
        ret = self.vae_chan_latent_u.kl_for_r(self.Rij_mb_sym)
        ret = ret + self.vae_chan_latent_v.kl_for_r(self.Rij_mb_sym)
        ret = T.sum(ret)
        ret = ret.reshape((),ndim=0)
        return ret

    @cached_property
    def obj(self):
        ret = self.vae_chan_latent_u.obj \
            + self.vae_chan_latent_v.obj \
            + self.reconstruction_obj
        return ret

    @cached_property
    def predict_to_1_lea(self):
        ret = 0.5 * (self.vae_chan_latent_u.r_out_lea + self.vae_chan_latent_v.r_out_lea)
        return ret

    @cached_property
    def predict_to_1_det(self):
        #ret = 0.5 * (self.vae_chan_latent_u.r_out_det + self.vae_chan_latent_v.r_out_det)
        ret = self.vae_chan_latent_v.r_out_det
        return ret

    @cached_property
    def params(self):
        ret = self.vae_chan_latent_u.q_phi_params \
            + self.vae_chan_latent_u.p_theta_params \
            + self.vae_chan_latent_v.q_phi_params \
            + self.vae_chan_latent_v.p_theta_params
        return ret

    @cached_property
    def grads_ui(self):
        return T.grad(self.obj, self.ui_mb_sym)

    @cached_property
    def grads_vj(self):
        return T.grad(self.obj, self.vj_mb_sym)

    @cached_property
    def grads_params(self):
        ret =  [
            T.grad(self.obj,curr)
            for curr
            in self.params
        ]
        return ret

def main():
    dataset = movielens.load(config.movielens_which)

    U,V = cftools.UV_vectors_np(dataset,latent_len=2*config.K)# *2 because it has to store mu and sigma
    U_t, U_m, U_v = update_algorithms.adam_for(U)
    V_t, V_m, V_v = update_algorithms.adam_for(V)


    def make_predict_to_5(predict_to_1_sym):
        ret = (predict_to_1_sym * (config.max_rating - 1. )) + 1.
        return ret

    print("creating update functions..")

    t_mb_prev_sym = T.fmatrix('t_mb_prev')
    t_mb_prev_sym = T.addbroadcast(t_mb_prev_sym,1)
    m_mb_prev_sym = T.fmatrix('m_mb_prev')
    v_mb_prev_sym = T.fmatrix('v_mb_prev')

    model = Model()

    #updates_kwargs = dict(t_prev=t_mb_prev_sym,m_prev=m_mb_prev_sym,v_prev=v_mb_prev_sym)
    #new_for_ui = list(update(model.ui_mb_sym,model.grads_ui,**updates_kwargs))
    #new_for_vj = list(update(model.vj_mb_sym,model.grads_vj,**updates_kwargs))
    new_for_vj = [model.vae_chan_latent_v.latent_distr_det]
    new_for_ui = [model.vae_chan_latent_u.latent_distr_det]
    params_updates = adam_shared(model.grads_params,model.params,learning_rate=config.lr_begin)

    #common = [ t_mb_prev_sym,m_mb_prev_sym,v_mb_prev_sym,model.Rij_mb_sym,model.ui_mb_sym,model.vj_mb_sym ]
    common = [ model.Rij_mb_sym,model.ui_mb_sym,model.vj_mb_sym ]
    ui_update_fn = theano.function(common,new_for_ui)
    ui_update_fn.name="ui_update_fn"
    vj_update_fn = theano.function(common,new_for_vj)
    vj_update_fn.name="vj_update_fn"
    params_update_fn = theano.function(
        [model.Rij_mb_sym,model.ui_mb_sym,model.vj_mb_sym],
        [],
        updates=params_updates
    )
    params_update_fn.name = "params_update_fn"
    theano.printing.pprint(model.predict_to_1_det)
    predict_to_1_fn = theano.function(
        [model.vae_chan_latent_u.latent_distr_det, model.vae_chan_latent_v.latent_distr_det],
        [model.predict_to_1_det]
    )
    predict_to_1_fn.name="predict_to_1_fn"

    predict_to_5_fn = theano.function(
        [model.vae_chan_latent_u.latent_distr_det, model.vae_chan_latent_v.latent_distr_det],
        [make_predict_to_5(model.predict_to_1_det)]
    )
    predict_to_5_fn.name="predict_to_5_fn"

    ui_mb_l = []
    vj_mb_l = []
    Rij_mb_l = []
    U_t_mb_l = []
    U_m_mb_l = []
    U_v_mb_l = []
    V_t_mb_l = []
    V_m_mb_l = []
    V_v_mb_l = []
    indices_mb_l = []
    def train_with_datapoint(i,j,Rij,lr):
        nonlocal indices_mb_l
        nonlocal ui_mb_l
        nonlocal vj_mb_l
        nonlocal Rij_mb_l
        nonlocal U_t_mb_l
        nonlocal U_m_mb_l
        nonlocal U_v_mb_l
        nonlocal V_t_mb_l
        nonlocal V_m_mb_l
        nonlocal V_v_mb_l

        indices_mb_l.append((i,j))
        ui_mb_l.append(U[i])
        vj_mb_l.append(V[j])
        Rij_mb_l.append(Rij)
        U_t_mb_l.append(U_t[i])
        U_m_mb_l.append(U_m[i])
        U_v_mb_l.append(U_v[i])
        V_t_mb_l.append(V_t[j])
        V_m_mb_l.append(V_m[j])
        V_v_mb_l.append(V_v[j])
        if len(ui_mb_l) >= config.minibatch_size:
            ui_mb = np.vstack(ui_mb_l).astype('float32')
            #print('ui_mb.shape',ui_mb.shape)
            vj_mb = np.vstack(vj_mb_l).astype('float32')
            #print('vj_mb.shape',vj_mb.shape)
            Rij_mb = np.vstack(Rij_mb_l).astype('float32')
            #print('Rij_mb.shape',Rij_mb.shape)

            U_t_mb = np.vstack(U_t_mb_l ).astype('float32')
            #print('U_t_mb.shape',U_t_mb.shape)
            U_m_mb = np.vstack(U_m_mb_l ).astype('float32')
            #print('U_m_mb.shape',U_m_mb.shape)
            U_v_mb = np.vstack(U_v_mb_l ).astype('float32')
            #print('U_v_mb.shape',U_v_mb.shape)
            V_t_mb = np.vstack(V_t_mb_l ).astype('float32')
            V_m_mb = np.vstack(V_m_mb_l ).astype('float32')
            V_v_mb = np.vstack(V_v_mb_l ).astype('float32')

            Rij_mb = (Rij_mb - 1.) / (config.max_rating - 1.)
            #log("Rij_mb",Rij_mb)
            #log("predict_to_1_fn",predict_to_1_fn(ui_mb,vj_mb))
            #log("predict_to_5_fn",predict_to_5_fn(ui_mb,vj_mb))
            #print("before ui_update_fn, vj_mb.shape=",vj_mb.shape)
            #print("before ui_update_fn, ui_mb.shape=",ui_mb.shape)
            #new_ui_mb, new_U_t_mb, new_U_m_mb, new_U_v_mb = ui_update_fn(
            new_ui_mb, = ui_update_fn(
                #U_t_mb,U_m_mb,U_v_mb,Rij_mb,ui_mb,vj_mb
                Rij_mb,ui_mb,vj_mb
            )
            #log("ui_mb",ui_mb,"new_ui_mb",new_ui_mb,"diff",ui_mb-new_ui_mb)
            #print("before vj_update_fn, vj_mb.shape=",vj_mb.shape)
            #print("before vj_update_fn, ui_mb.shape=",ui_mb.shape)
            #new_vj_mb, new_V_t_mb, new_V_m_mb, new_V_v_mb = vj_update_fn(
            new_vj_mb, = vj_update_fn(
                #V_t_mb,V_m_mb,V_v_mb,Rij_mb,ui_mb,vj_mb
                Rij_mb,ui_mb,vj_mb
            )
            #log("vj_mb",vj_mb,"new_vj_mb",new_vj_mb,"diff",vj_mb-new_vj_mb)

            for pos,(i,j) in enumerate(indices_mb_l):
                U[i] = new_ui_mb[pos,:]
                V[j] = new_vj_mb[pos,:]
                #U_t[i] = new_U_t_mb[pos,:]
                #U_m[i] = new_U_m_mb[pos,:]
                #U_v[i] = new_U_v_mb[pos,:]
                #V_t[j] = new_V_t_mb[pos,:]
                #V_m[j] = new_V_m_mb[pos,:]
                #V_v[j] = new_V_v_mb[pos,:]
            params_update_fn(Rij_mb,ui_mb,vj_mb)

            ui_mb_l = []
            vj_mb_l = []
            Rij_mb_l = []

            U_t_mb_l = []
            U_m_mb_l = []
            U_v_mb_l = []
            V_t_mb_l = []
            V_m_mb_l = []
            V_v_mb_l = []
            indices_mb_l = []
    print("training pmf...")
    cftools.mainloop(train_with_datapoint,dataset,U,V,predict_to_5_fn)

if __name__=="__main__":
    main()

