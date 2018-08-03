#!/usr/bin/env python3

import scipy
import ipdb
import theano
import theano.sparse
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode
import lasagne
print(lasagne.__file__)
import lasagne_sparse

import numpy as np
import random
import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
import ipdb
import scipy.sparse

# local imports

import movielens
import cftools
import config
import numutils as nu
import augmented_types as at
import activation_functions
import update_algorithms
import model_build
import kl
import utils
import regularization
import persistency
np.set_printoptions(edgeitems=200)

#update = adam_masked = update_algorithms.adam_masked # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g_rij = activation_functions.get(config.g_rij)
g_latent = activation_functions.get(config.g_latent)
g_hid = activation_functions.get(config.g_hid)
g_log_sigma = activation_functions.get("safe_log_output")
g_transform = activation_functions.get(config.g_transform)
g_flow = activation_functions.get(config.g_flow)
weights_regularization = regularization.get(config.regularization_type)
sigma = 1.
sigma_u = 100.
sigma_v = 1000.
num_leading_axes = 1
kl_annealing_epsilon = getattr(config,"kl_annealing_epsilon",0.05)

chan_out_dim = config.chan_out_dim
hid_dim = config.hid_dim
latent_dim = config.K
TK=config.TK
flow_type = getattr(config,"flow_type","planar")
enforce_invertibility = getattr(config, "enforce_invertibility",True)
log = lambda *args,**kwargs : print(*args,**kwargs)


class Model(model_build.Abstract):

    def __init__(self,dataset):
        # having a reference to the dataset is required as some information such
        # as the number of items is needed for the network structure
        self.dataset = dataset

        self.Ri_mb_sym = theano.sparse.csr_matrix(name='Ri_mb',dtype='float32')
        self.make_net()

    def create_new_realnvp_mask(self):
        cutpoint=float(latent_dim)/2
        ones = np.ones((np.floor(cutpoint).astype('int'),))
        zeros = np.zeros((np.ceil(cutpoint).astype('int'),))
        m = np.concatenate([ones,zeros])
        np.random.shuffle(m)
        m = m.astype('float32')
        return m

    def input_dropout(self, layer):
        if config.input_dropout_p > 0:
            layer = lasagne_sparse.SparseInputDropoutLayer(
                layer,
                p=config.input_dropout_p,
                rescale=True,
                name="inputdrop_"+layer.name
            )
        return layer

    def make_net(self):
        self.l_in = self.input_dropout(lasagne.layers.InputLayer(
            (config.minibatch_size, self.input_dim,),
            input_var=self.Ri_mb_sym,
            name="input_layer"
        ))

        self.l_hid_enc = self.l_in
        latent0_layer_type = lasagne_sparse.SparseInputDenseLayer
        if config.n_hid_layers == 0:
            pass
        else: # 1 or multiple hidden layers
            for hid_count in range(config.n_hid_layers):
                self.l_hid_enc = self.wrap(latent0_layer_type(
                    self.l_hid_enc, # replace field with last hidden layer
                    num_units=hid_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_hid,
                    name="hidden_enc_layer_{}".format(hid_count)
                ))
                latent0_layer_type = lasagne.layers.DenseLayer

        self.l_latent0_mu = latent0_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_latent,
            name="latent0_mu"
        )
        self.l_latent0_log_sigma = latent0_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_log_sigma,
            name="latent0_log_sigma"
        )

        self.l_transformations_w = []
        self.l_transformations_b = []
        self.l_transformations_aw = []
        self.l_transformations_ab = []
        self.l_transformations_u = []
        self.l_transformations_u_hat = []
        self.l_transformations_masks = []
        for k in range(TK):
            if flow_type == "planar":
                w_output_dim = latent_dim
            elif flow_type =="realnvp":
                w_output_dim = latent_dim**2

            w = latent0_layer_type(
                self.l_hid_enc,
                num_units=w_output_dim,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_latent,
                name="transformation_w"
            )
            b = latent0_layer_type(
                self.l_hid_enc,
                num_units=1,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_latent,
                name="transformation_b"
            )
            u = latent0_layer_type(
                self.l_hid_enc,
                num_units=latent_dim,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_latent,
                name="transformation_u"
            )
            self.l_transformations_w.append(w)
            self.l_transformations_b.append(b)
            if flow_type == "planar":
                self.l_transformations_u.append(u)
                if enforce_invertibility == True:
                    u_hat = model_build.ILTTEnforceInvertibilityLayer(
                        [
                            self.l_transformations_w[k],
                            self.l_transformations_u[k]
                        ],
                        dim=latent_dim,
                        name="ILTTEnforceInvertibilityLayer{}".format(k+1) #1-based displaying
                    )
                else:
                    u_hat = u
                self.l_transformations_u_hat.append(u_hat)
            elif flow_type == "realnvp":
                aw = latent0_layer_type(
                    self.l_hid_enc,
                    num_units=w_output_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_latent,
                    name="transformation_w"
                )
                ab = latent0_layer_type(
                    self.l_hid_enc,
                    num_units=latent_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_latent,
                    name="transformation_b"
                )
                self.l_transformations_aw.append(aw)
                self.l_transformations_ab.append(ab)
            else:
                raise Exception("don't understand flow_type="+flow_type)
        self.l_latent0_merge = lasagne.layers.ConcatLayer(
            [
                self.l_latent0_mu,
                self.l_latent0_log_sigma
            ],
            name="latent0_concat"
        )
        self.l_latent0_sampling = model_build.SamplingLayer(
            self.l_latent0_merge,
            dim=latent_dim,
            name="latent0_sampling"
        )

        self.l_transformations = []
        self.l_transformations_s_masked = []
        l_prev = self.l_latent0_sampling
        for k in range(TK):
            if flow_type == "planar":
                l = model_build.ILTTLayer(
                    [
                        l_prev,
                        self.l_transformations_w[k],
                        self.l_transformations_b[k],
                        self.l_transformations_u_hat[k]
                    ],
                    dim=latent_dim,
                    name="ILTT{}".format(k+1), #1-based displaying
                    nonlinearity=g_flow
                )
                l_prev = l
            elif flow_type == "realnvp":
                m = self.create_new_realnvp_mask()
                self.l_transformations_masks.append(m)
                m1 = m # first half
                m2 = 1-m # second half
                l_z_first_half = model_build.MaskLayer(l_prev,mask=m1,dim=latent_dim,name="MaskLayer_first_half_{}".format(k))
                l_z_second_half = model_build.MaskLayer(l_prev,mask=m2,dim=latent_dim,name="MaskLayer_second_half_{}".format(k))
                l_s = model_build.ProducedDenseLayer(
                    [
                        l_z_first_half,
                        self.l_transformations_w[k],
                        self.l_transformations_b[k]
                    ],
                    dim=latent_dim,
                    name="ProducedDenseLayer{}".format(k),
                    nonlinearity=g_flow
                )
                l_s_masked = model_build.MaskLayer(l_s,mask=m2,dim=latent_dim,name="s_masked_{}".format(k))
                self.l_transformations_s_masked.append(l_s_masked)
                l_s_exp = lasagne.layers.ExpressionLayer(l_s_masked, T.exp,name="s_exp_{}".format(k))
                l_zt_second_half = lasagne.layers.ElemwiseMergeLayer(
                    [
                        l_z_second_half,
                        l_s_exp
                    ],
                    T.mul,
                    name = "zt_second_half_{}".format(k)
                )
                l_a = model_build.ProducedDenseLayer(
                    [
                        l_z_first_half,
                        self.l_transformations_aw[k],
                        self.l_transformations_ab[k]
                    ],
                    dim=latent_dim,
                    name="Affine{}".format(k),
                    nonlinearity=lasagne.nonlinearities.linear
                )
                l_a_mask = model_build.MaskLayer(
                    l_a,
                    mask=m2,
                    dim=latent_dim,
                    name="l_a_mask_{}".format(k)
                )
                l = lasagne.layers.ElemwiseMergeLayer(
                    [
                        l_z_first_half,
                        l_zt_second_half,
                        l_a_mask
                    ],
                    T.add,
                    name="realnvp_{}".format(k)
                )

            self.l_transformations.append(l)

        if len(self.l_transformations) == 0:
            self.l_transformation_last = self.l_latent0_sampling
        else:
            self.l_transformation_last = self.l_transformations[-1] #NK-th

        if config.n_hid_layers == 0:
            self.l_hid_mu_dec = self.l_hid_log_sigma_dec = self.l_transformation_last
        else: # 1 or multiple hidden layers
            curr_l_hid = self.l_transformation_last
            for hid_count in range( config.n_hid_layers - 1 ):
                curr_l_hid = self.wrap(lasagne.layers.DenseLayer(
                    curr_l_hid, # replace with last hidden layer
                    num_units=hid_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_hid,
                    name="hidden_dec_layer_{}".format(hid_count)
                ))

            self.l_hid_mu_dec = self.wrap(lasagne.layers.DenseLayer(
                curr_l_hid,
                num_units=hid_dim,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_hid,
                name="hidden_dec_layer_mu"
            ))
            self.l_hid_log_sigma_dec = self.wrap(lasagne.layers.DenseLayer(
                curr_l_hid,
                num_units=hid_dim,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_hid,
                name="hidden_dec_layer_sigma"
            ))


        self.l_out_mu = lasagne.layers.DenseLayer(
            self.l_hid_mu_dec,
            num_units=self.input_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_rij,
            name="out_mu_layer"
        )
        self.l_out_log_sigma= lasagne.layers.DenseLayer(
            self.l_hid_log_sigma_dec,
            num_units=self.input_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_log_sigma,
            name="out_log_sigma_layer"
        )
        self.l_out = lasagne.layers.ConcatLayer(
            [
                self.l_out_mu,
                self.l_out_log_sigma
            ],
            name="out_concat_layer"
        )
        self.log("all layers: ", self.all_layers)


    @utils.cached_property
    def Rij_mb_dense(self):
        ret = theano.sparse.dense_from_sparse(self.Ri_mb_sym)
        return ret

    @utils.cached_property
    def mask(self):
        mask_plus  = self.Rij_mb_dense >  0.0000000001
        mask_minus = self.Rij_mb_dense < -0.0000000001
        ret = mask_plus + mask_minus
        return ret

    @utils.cached_property
    def mask_sum(self):
        ret = T.cast(T.sum(self.mask),'float32')
        ret.name = "mask_sum"
        return ret

    @utils.cached_property
    def mask_mb_sum(self):
        """
        total mask resulting from summing (collapsing) over the minibatch
        axis. Useful, for example, for filtering the gradient of the weights
        matrix of a first layer if the input is sparse and only the weights
        connected to visible units need to be updated
        """
        _sum = T.sum(
            self.mask,
            axis=0,
            keepdims=True
        )
        ret = _sum > 0
        return ret

    @utils.cached_property
    def mask_enc_W(self):
        ret = T.tile(self.mask_mb_sum.T, (1,hid_dim)) # notice the transpose
        ret.name = "mask_enc_W_ret"
        return ret

    @utils.cached_property
    def mask_dec_W(self):
        ret = T.tile(self.mask_mb_sum, (hid_dim,1)) # notice absence of transpose
        ret.name = "mask_dec_W_ret"
        return ret

    @utils.cached_property
    def loss_sq(self):
        ret = T.sqr(self.Ri_mb_sym - self.predict_to_1_lea)
        ret.name = "loss_sq"
        return ret

    @utils.cached_property
    def likelihood_term_logdetsigma(self):
        masked_log_sigma = self.out_log_sigma_lea * self.mask
        masked_log_sigma.name="likelihood_masked_log_sigma"
        term_logdetsigma = -T.sum(masked_log_sigma)
        term_logdetsigma.name = "likelihood_term_logdetsigma"
        return term_logdetsigma

    @utils.cached_property
    def likelihood_term_scaled_error(self):
        sigma = T.exp(self.out_log_sigma_lea)
        sigma.name="likelihood_sigma"
        masked_loss_sq = self.loss_sq * self.mask
        masked_loss_sq.name="likelihood_masked_loss_sq"
        inv_sigma = 1./sigma
        inv_sigma.name = "likelihood_inv_sigma"
        term_scaled_error = - T.sum(masked_loss_sq * inv_sigma)
        term_scaled_error.name = "likelihood_term_scaled_error"
        return term_scaled_error

    @utils.cached_property
    def likelihood(self):
        # this is the expected reconstruction error
        # the 1/2 coefficient is EXTERNAL to this term,
        # being config.regression_error_coef
        # a.k.a. likelihood!!!

        if getattr(config, "add_constant_terms",True):
            term_constant = -self.mask_sum * np.array(2*np.pi).astype('float32')
            term_constant.name="likelihood_term_constant"
        else:
            term_constant = 0
        ret = term_constant + self.likelihood_term_logdetsigma + self.likelihood_term_scaled_error
        ret.name = "likelihood_ret"
        ret = model_build.scalar(ret)
        ret.name = "likelihood_scalar_ret"
        return ret

    @utils.cached_property
    def regularizer_latent0_kl(self):
        sigma = T.exp(self.latent0_log_sigma_lea)
        sigma.name = "rl0kl_sigma"
        vanilla_kl = kl.kl_normal_diagonal_vs_unit(self.latent0_mu_lea,sigma,latent_dim)
        vanilla_kl.name="rl0kl_vanilla_kl"
        """ DEACTIVATING original max-free nats
        m = config.free_nats * theano.tensor.ones_like(vanilla_kl)
        m.name = "rl0kl_m"
        ret = theano.tensor.maximum(vanilla_kl,m)
        ret.name = "rl0kl_ret"
        return ret
        """
        return vanilla_kl

    @utils.cached_property
    def marginal_latent0_kl(self):
        term1 = - self.latent0_log_sigma_lea
        term1.name="ml0kl_term1"
        term2 = 0.5*(T.pow(self.latent0_sigma_lea,2) + T.pow(self.latent0_mu_lea,2))
        term2.name="ml0kl_term2"
        term3 = - 0.5
        ret = term1 + term2 + term3
        ret.name="ml0kl_ret"
        return ret

    @utils.cached_property
    def marginal_latent0_kl_mean(self):
        return T.mean(self.marginal_latent0_kl)

    @utils.cached_property
    def marginal_latent0_kl_std(self):
        return T.std(self.marginal_latent0_kl)

    @utils.cached_property
    def excluded_loss(self):
        ret = (self.loss_sq * (1-self.mask)).mean()
        return ret

    @utils.cached_property
    def kl_annealing(self):
        t = self.epoch_nr
        aT = config.kl_annealing_T
        d = T.minimum(t,aT)
        d = T.maximum(d,1)
        ret = d/aT
        ret.name = "kl_annealing"
        return ret

    @utils.cached_property
    def latent_kl(self):
        if config.TK == 0:
            ret = self.regularizer_latent0_kl
        else: # TK>0
            ret = -self.latentK_term_obj
            ret -= self.latent0_entropy_term_obj
            ret -= self.transformation_term_obj
        ret.name="latent_kl"
        return ret

    @utils.cached_property
    def latent_kl_average(self):
        ret = self.latent_kl / config.minibatch_size
        ret.name="latent_kl"
        return ret

    @utils.cached_property
    def elbo(self):
        """
        the ELBO is expressed as two terms: likelihood, which is positive
        (has to be maximized)
        and KL between approximate posterior and prior on latent0
        which is negative (has to be minimized)
        """
        # WARNING: regression_error_coef should be 0.5!!!
        ret = self.likelihood * config.regression_error_coef
        if config.regularization_latent_kl > 0. :
            ret -= self.latent_kl * self.kl_annealing * config.regularization_latent_kl
        ret.name="elbo_ret"
        return ret

    @utils.cached_property
    def obj(self):
        """
        the objective function should be the negative ELBO
        because the ELBO needs to be maximized,
        and objective functions are minimized
        """
        ret = -self.elbo
        ret /= config.minibatch_size # it's an average!
        if config.regularization_lambda > 0.:
            ret += self.regularizer * config.regularization_lambda / self.n_datapoints
        return ret

    @utils.cached_property
    def latentK_term_obj(self):
        if getattr(config, "add_constant_terms",True):
            term_constant= - 0.5 * self.mask_sum * np.log(np.array(2*np.pi)).astype('float32')
            term_constant.name="lK_term_constant"
        else:
            term_constant = 0
        lK_squared = self.latentK_lea ** 2
        lK_squared.name = "lK_squared"
        term_l2 = - 0.5 * T.sum(lK_squared)
        term_l2.name="lK_term_l2"
        ret = term_constant + term_l2
        ret.name = "lK_ret"
        ret = model_build.scalar(ret)
        ret.name = "lK_ret_scalar"
        return ret

    @utils.cached_property
    def latent0_entropy_term_obj(self):
        if getattr(config, "add_constant_terms",True):
            term_constant = 0.5 * self.mask_sum * np.log(np.array(2*np.pi)).astype('float32')
        else:
            term_constant = 0
        term_dim = 0.5 * latent_dim
        masked_log_sigma = self.latent0_log_sigma_lea
        term_logdetsigma = 0.5 * T.sum(masked_log_sigma)
        ret = term_constant + term_dim + term_logdetsigma
        ret.name = "latent0_entropy_term_obj"
        ret = model_build.scalar(ret)
        return ret

    @utils.cached_property
    def transformation_ws(self):
        ret = []
        for l in self.l_transformations_w:
            o = lasagne.layers.get_output(l,deterministic=False)
            o.name="transformation_w!!"
            ret.append(o)
        return ret

    @utils.cached_property
    def transformation_bs(self):
        ret = []
        for l in self.l_transformations_b:
            o = lasagne.layers.get_output(l,deterministic=False)
            o = o.T
            o.name="transformation_b!!"
            ret.append(o)
        return ret

    @utils.cached_property
    def transformation_us(self):
        ret = []
        for l in self.l_transformations_u:
            o = lasagne.layers.get_output(l,deterministic=False)
            o.name="transformation_u!!"
            ret.append(o)
        return ret

    @utils.cached_property
    def transformation_u_hats(self):
        ret = []
        for l in self.l_transformations_u_hat:
            o = lasagne.layers.get_output(l,deterministic=False)
            o.name="transformation_u_hat!!"
            ret.append(o)
        return ret

    @utils.cached_property
    def transformation_zs(self):
        ret = []
        for l in self.l_transformations:
            o = lasagne.layers.get_output(l,deterministic=False)
            o.name="transformation_z!!"
            ret.append(o)
        return ret

    @utils.cached_property
    def transformation_term_obj(self):
        global flow_type
        if flow_type == 'planar':
            return self.transformation_term_obj_planar
        elif flow_type == 'realnvp':
            return self.transformation_term_obj_realnvp
        else:
            raise Exception("don't understand flow_type="+flow_type)

    @utils.cached_property
    def transformation_term_obj_planar(self):
        ret = T.constant(0).astype('float32')
        zkminusone = self.latent0_sample_lea
        h = g_transform
        self.aa = []
        for k,w,b,u_hat,z in zip(
                range(TK),
                self.transformation_ws,
                self.transformation_bs,
                self.transformation_u_hats,
                self.transformation_zs
        ):
            b.name="transformation_b!"
            zkminusone.name = "z_{}".format(k-1)
            zw,_updates = theano.scan(
                fn=T.dot,
                sequences=[zkminusone,w]
            )
            zw.name="zw!"
            #zw.tag.test_value = np.ones((3,3))
            print(type(zw))
            a = zw + b
            a = a.T
            a.name="a!"
            self.aa.append(a)
            self.aa.append(b)
            self.aa.append(zw)
            a_range = T.arange(a.shape[0])
            a_range.name="a_range"
            h_a = h(a)
            h_a.name="h_a!"
            #h_prime_output = T.grad(h_a,a)

            def do_grad_h (i,ai):
                #ai = a[i,:]
                hai = h(ai)
                hai.name="hai"
                shai = model_build.scalar(hai)
                shai.name="shai"
                print("shai",shai)
                gr = T.grad( shai, ai)
                gr.name="shai_grad"
                return gr

            hprime,_updates = theano.scan(
                fn=do_grad_h,
                sequences=[a_range,a],
            )
            d,_updates = theano.scan(
                fn=T.dot,
                sequences=[u_hat,w]
            )
            d.name="d!"

            m = hprime * d
            m.name="m!"
            s = 1+m
            s.name="s!"
            ret += T.log(T.abs_(s))
            zkminusone = z
            zkminusone.name="zk-1!"
        ret.name = "transformation_term_obj"
        ret = model_build.scalar(T.sum(ret))
        return ret

    @utils.cached_property
    def transformation_term_obj_realnvp(self):
        logdetJ = 0
        for k in range(TK):
            logdetJ += T.sum(self.l_transformations_s_masked_lea[k])
        return logdetJ

    @utils.cached_property
    def l_transformations_s_masked_lea(self):
        ret = []
        for curr in self.l_transformations_s_masked:
            ret.append(lasagne.layers.get_output(curr,deterministic=False))
        return ret

    @utils.cached_property
    def out_mu_lea(self):
        ret = lasagne.layers.get_output(self.l_out_mu,deterministic=False)
        return ret

    @utils.cached_property
    def out_mu_det(self):
        ret = lasagne.layers.get_output(self.l_out_mu,deterministic=True)
        return ret

    @utils.cached_property
    def out_log_sigma_lea(self):
        ret = lasagne.layers.get_output(self.l_out_log_sigma,deterministic=False)
        if config.spherical_likelihood:
            ret = T.zeros_like(ret)
        return ret

    @utils.cached_property
    def out_log_sigma_det(self):
        ret = lasagne.layers.get_output(self.l_out_log_sigma,deterministic=True)
        if config.spherical_likelihood:
            ret = T.zeros_like(ret)
        return ret

    @utils.cached_property
    def latent0_mu_lea(self):
        ret = lasagne.layers.get_output(self.l_latent0_mu,deterministic=False)
        return ret

    @utils.cached_property
    def latent0_log_sigma_lea(self):
        ret = lasagne.layers.get_output(self.l_latent0_log_sigma,deterministic=False)
        return ret

    @utils.cached_property
    def latent0_sigma_lea(self):
        ret = T.exp(self.latent0_log_sigma_lea)
        return ret

    @utils.cached_property
    def latent0_sample_lea(self):
        ret = lasagne.layers.get_output(self.l_latent0_sampling,deterministic=False)
        return ret

    @utils.cached_property
    def latent0_sample_det(self):
        ret = lasagne.layers.get_output(self.l_latent0_sampling,deterministic=True)
        return ret

    @utils.cached_property
    def latentK_lea(self):
        ret = lasagne.layers.get_output(self.l_transformation_last,deterministic=False)
        return ret

    @utils.cached_property
    def latentK_det(self):
        ret = lasagne.layers.get_output(self.l_transformation_last,deterministic=True)
        return ret

    @utils.cached_property
    def predict_to_1_lea(self):
        ret = self.out_mu_lea
        return ret

    @utils.cached_property
    def predict_to_1_det(self):
        ret = self.out_mu_det
        return ret

    @utils.cached_property
    def params(self):
        ret = lasagne.layers.get_all_params(self.l_out, trainable=True)

        if config.spherical_likelihood:
            removand = []
            for param in ret:
                layer_name = param.name.split('.')[0]
                if layer_name in [
                        'hidden_dec_layer_sigma',
                        'out_log_sigma_layer'
                ]:
                    removand.append(param)
            for curr in removand:
                ret.remove(curr)
        return ret

    @utils.cached_property
    def grads_params(self):
        ret = []
        for curr in self.params:
            grad = T.grad(self.obj,curr,add_names=True)
            grad.name = str(curr.name)+"_grad"
            print("grad.name",grad.name)

            # filtering first and last layer's gradients according
            # to which ratings were observed
            if curr.name in self.all_masks.keys():
                grad = grad * self.all_masks[curr.name]
                grad.name = str(curr.name) + "_masked_grad"

            ret.append(grad)
        return ret

    @utils.cached_property
    def all_masks(self):
        ret = {
            "hidden_enc_layer.W": self.mask_enc_W,
            "out_layer.W": self.mask_dec_W
        }
        return ret

    @utils.cached_property
    def regularizer(self):
        ret = lasagne.regularization.regularize_layer_params(
            self.all_layers,
            weights_regularization
        )
        return ret

def log_percentiles(quantity,name,_log):
    for q in [1,2,5, 10,20, 50,80,90,95,98,99]:
        percentile = np.percentile(quantity,q)
        _log("{} percentile {}: {}".format(name,q,percentile))

def main():
    dataset = movielens.load(config.movielens_which)

    def make_predict_to_5(predict_to_1_sym):
        ret = cftools.unpreprocess(predict_to_1_sym,dataset)
        return ret

    log("creating model..")

    model = Model(dataset)
    model.log = log
    log("parameters shapes:",[p.get_value().shape for p in model.params])

    log("creating marginal_latent0_kl diagnostic functions..")
    marginal_latent0_kl_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [
            model.marginal_latent0_kl
        ],
    )
    marginal_latent0_kl_fn.name = "marginal_latent0_kl_fn"

    log("creating out_log_sigmas diagnostic functions..")
    out_log_sigmas_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [
            model.out_log_sigma_lea
        ],
    )
    out_log_sigmas_fn.name = "out_log_sigmas_fn"

    log("done.")
    #theano.printing.pprint(model.predict_to_1_det)

    predict_to_1_fn = model_build.make_function( # FIXME: change name
        [model.Ri_mb_sym],
        [model.predict_to_1_det]
    )
    predict_to_1_fn.name="predict_to_1_fn"

    predict_to_5_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [make_predict_to_5(model.predict_to_1_det)]
    )
    predict_to_5_fn.name="predict_to_5_fn"

    likelihood_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [model.likelihood]
    )
    likelihood_fn.name = "likelihood_fn"

    Ri_mb_l = []
    indices_mb_l = []

    total_loss = 0
    total_kls = []
    total_objs = []
    total_likelihoods = []
    total_out_log_sigmas = []
    kl_annealing_soft_free_nats=1.0 # used only when config.soft_free_nats=True
    f_an = None
    f_kl = None

    def epoch_hook(*args,**kwargs):
        _log = kwargs.pop('log',log)
        _epochsloop = kwargs['epochsloop']
        _epoch_nr = kwargs.pop('epoch_nr')
        def meanstd(quantity):
            v = quantity.get_value()
            m = np.mean(v,axis=None)
            s = np.std(v,axis=None)
            log(quantity.name,"mean:",m,"std:",s)
        nonlocal total_loss
        nonlocal total_kls
        nonlocal total_objs
        nonlocal total_likelihoods
        nonlocal total_out_log_sigmas
        nonlocal model
        _log("\ntotal_loss:",total_loss)
        _log("total_kls mean:",np.mean(total_kls))
        _log("total_kls std:",np.std(total_kls))
        mean_total_kls_per_dim = np.mean(total_kls,axis=0) # squashes over the datapoints
        log_percentiles(mean_total_kls_per_dim,"mean_total_kls_per_dim",_log)
        log_percentiles(total_objs,"objs",_log)
        log_percentiles(total_likelihoods,"likelihoods",_log)
        log_percentiles(total_out_log_sigmas,"out_log_sigmas",_log)
        for curr in model.params:
            meanstd(curr)

        def validation_splits():
            ret = cftools.split_minibatch_rrows(
                _epochsloop.validation_set,
                'split validation set for objs'
            )
            return ret # generator, needs to be refreshed.

        validation_objs = []
        for curr in tqdm(validation_splits(),desc="objs validation set"):
            _obj = obj_fn(curr,_epoch_nr)
            validation_objs.append(_obj)
        log_percentiles(validation_objs,"objs validation set",_log)
        validation_likelihoods = []
        for curr in tqdm(validation_splits(),desc="likelihoods validation set"):
            _l = likelihood_fn(curr)
            validation_likelihoods.append(_l)
        log_percentiles(validation_likelihoods,"likelihoods validation set",_log)

        _log("\n\n")

        total_loss = 0
        total_kls = []
        total_objs = []
        total_likelihoods = []
        total_out_log_sigmas = []

        epoch_nr = kwargs['epochsloop'].epoch_nr
        _lr = update_algorithms.calculate_lr(epoch_nr)
        persistency.save(model,_lr,epoch_nr)

    def train_with_rrow(i,Ri,epoch_nr): # Ri is an entire sparse row of ratings from a user
        nonlocal indices_mb_l
        nonlocal Ri_mb_l
        nonlocal total_loss
        nonlocal total_kls
        nonlocal total_likelihoods
        nonlocal total_out_log_sigmas
        nonlocal kl_annealing_soft_free_nats
        nonlocal f_kl
        nonlocal f_an
        if f_kl is None:
            f_kl = open("kl_average.log","w+")
        if f_an is None:
            f_an = open("kl_annealing.log","w+")

        indices_mb_l.append((i,))
        Ri_mb_l.append(Ri)
        if len(Ri_mb_l) >= config.minibatch_size:
            Ri_mb = scipy.sparse.vstack(Ri_mb_l)
            Ri_mb.data = cftools.preprocess(Ri_mb.data,dataset) # FIXME: method of Dataset?
            params_update_args = [
                Ri_mb,
                epoch_nr
            ]
            if getattr(config,"soft_free_nats",False):
                params_update_args.append(kl_annealing_soft_free_nats)
            tmp = params_update_fn(*params_update_args)
            _loss,_lh,_regkl,_trans = tmp[0:4]
            _lr = tmp[-3]
            _kl_annealing = tmp[-2]
            _latent_kl_average = tmp[-1]
            if getattr(config,"soft_free_nats",False):
                if _latent_kl_average > config.free_nats * 1.05:
                    kl_annealing_soft_free_nats *= (1.0 + kl_annealing_epsilon)
                else:
                    kl_annealing_soft_free_nats *= (1.0 -  kl_annealing_epsilon)

            f_kl.write(_latent_kl_average+"\n")
            f_an.write(_kl_annealing+"\n")
            if getattr(config, "verbose", False):
                print("_grads")
                _grads = tmp[4:-(3+len(model.all_layers))]
                _layers_outputs = tmp[-(3+len(model.all_layers)):]
                do_exit = False
                for p,g in zip(model.params,_grads):
                    print(p.name,"g: min",g.min(),"max",g.max(), "val:",p.get_value().min(),p.get_value().max())
                    if np.isnan(g).any():
                        print("is nan.")
                        do_exit = True
                for l,o in zip(model.all_layers,_layers_outputs):
                    print("layer",l.name,"min",o.min(),"max",o.max())
                    if type(o) is scipy.sparse.csr.csr_matrix:
                        o = o.toarray()
                    if np.isnan(o).any():
                        print("is nan.")
                        do_exit = True
                if do_exit:
                    print('was nan')
                    sys.exit(0)
                print("_loss:",_loss,"_lh:",_lh,"_regkl:",_regkl,"_trans:",_trans)
            _kls, = marginal_latent0_kl_fn(Ri_mb)
            _out_log_sigmas, = out_log_sigmas_fn(Ri_mb)
            _obj, = obj_fn(Ri_mb, epoch_nr)
            _likelihood, = likelihood_fn(Ri_mb)
            total_kls.append(_kls)
            total_loss += _loss
            total_objs.append(_obj)
            total_likelihoods.append(_likelihood)
            total_out_log_sigmas.append(_out_log_sigmas)
            Ri_mb_l = []
            indices_mb_l = []

    looper = cftools.LooperRrows(
        train_with_rrow,
        dataset,
        predict_to_5_fn,
        epoch_hook=epoch_hook,
        model=model
    )
    # model needs n_datapoints to divide regularizing lambda
    model.n_datapoints = looper.n_datapoints

    log("creating parameter update function..")
    _ = model.obj # trigger getter
    """
    a_fn = model_build.make_function(
        [model.Ri_mb_sym],
        model.aa
    )
    a_fn.name="a_fn"
    """
    params_update_inputs = [
        model.Ri_mb_sym,
        model.epoch_nr
    ]

    if getattr(config,"soft_free_nats",False):
        params_update_inputs.append(model.kl_annealing)

    params_update_outputs = [
        model.obj,
        model.likelihood,
        model.regularizer_latent0_kl,
        model.transformation_term_obj,
    ]+model.grads_params+model.all_layers_outputs
    params_update_outputs+=[
        model.lr,
        model.kl_annealing,
        model.latent_kl_average
    ]

    params_update_fn = model_build.make_function(
        params_update_inputs,
        params_update_outputs,
        updates=model.params_updates
    )
    #import ipdb; ipdb.set_trace()
    params_update_fn.name = "params_update_fn"

    obj_fn = model_build.make_function(
        [
            model.Ri_mb_sym,
            model.epoch_nr
         ],
        [model.obj]
    )
    obj_fn.name = "obj_fn"


    log("training ...")
    #import ipdb; ipdb.set_trace()
    looper.start()

if __name__=="__main__":
    main()

