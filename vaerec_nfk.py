#!/usr/bin/env python3

import scipy
import ipdb
import theano
import theano.sparse
from theano import tensor as T
import lasagne
import lasagne_sparse

import numpy as np
import random
import sys
from tqdm import tqdm
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

update =update_algorithms.get_func()
adam_masked = update_algorithms.adam_masked # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g_rij = activation_functions.get(config.g_rij)
g_latent = activation_functions.get(config.g_latent)
g_hid = activation_functions.get(config.g_hid)
g_log_sigma = lasagne.nonlinearities.linear
g_transform = activation_functions.get(config.g_transform)
weights_regularization = regularization.get(config.regularization_type)
sigma = 1.
sigma_u = 100.
sigma_v = 1000.
num_leading_axes = 1

chan_out_dim = config.chan_out_dim
hid_dim = config.hid_dim
latent_dim = config.K
TK = config.TK
#log = print
log = lambda *args: print(*args)#None

def scalar(sometensor):
    return sometensor.reshape((),ndim=0)

class Model(object):

    def __init__(self,dataset):
        # having a reference to the dataset is required as some information such
        # as the number of items is needed for the network structure
        self.dataset = dataset

        self.Ri_mb_sym = theano.sparse.csr_matrix(name='Ri_mb',dtype='float32')
        self.make_net()
        self._n_datapoints = None # to be set later

    @property
    def n_datapoints(self):
        assert self._n_datapoints is not None
        return self._n_datapoints

    @n_datapoints.setter
    def n_datapoints(self,val):
        self._n_datapoints = val

    def input_dropout(self, layer):
        if config.input_dropout_p > 0:
            layer = lasagne_sparse.SparseInputDropoutLayer(
                layer,
                p=config.input_dropout_p,
                rescale=True,
                name="inputdrop_"+layer.name
            )
        return layer

    def dropout(self,layer):
        if config.dropout_p > 0:
            layer = lasagne.layers.DropoutLayer(
                layer,
                p=config.dropout_p,
                rescale=False,
                name="drop_"+layer.name
            )
        return layer

    def make_net(self):
        self.l_in = self.input_dropout(lasagne.layers.InputLayer(
            (config.minibatch_size, self.input_dim,),
            input_var=self.Ri_mb_sym,
            name="input_layer"
        ))

        self.l_hid_enc = self.l_in
        latent_layer_type = lasagne_sparse.SparseInputDenseLayer
        if config.n_hid_layers == 0:
            pass
        else: # 1 or multiple hidden layers
            for hid_count in range(config.n_hid_layers):
                self.l_hid_enc = self.dropout(latent_layer_type(
                    self.l_hid_enc, # replace field with last hidden layer
                    num_units=hid_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_hid,
                    name="hidden_enc_layer_{}".format(hid_count)
                ))
                latent_layer_type = lasagne.layers.DenseLayer

        self.l_latent0_mu = latent_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_latent,
            name="latent0_mu"
        )
        self.l_latent0_log_sigma = latent_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_log_sigma,
            name="latent0_log_sigma"
        )
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
        l_prev = self.l_latent0_sampling
        for k in range(TK):
            l = model_build.ILTTLayer(
                l_prev,
                dim=latent_dim,
                name="ILLT{}".format(k+1), #1-based displaying
                nonlinearity=g_transform
            )
            l_prev = l
            self.l_transformations.append(l)

        if config.n_hid_layers == 0:
            self.l_hid_mu_dec = self.l_transformations[-1]
        else: # 1 or multiple hidden layers
            curr_l_hid = self.l_transformations[-1] # K-th transformation layer
            for hid_count in range( config.n_hid_layers - 1 ):
                curr_l_hid = self.dropout(lasagne.layers.DenseLayer(
                    curr_l_hid, # replace with last hidden layer
                    num_units=hid_dim,
                    num_leading_axes=num_leading_axes,
                    nonlinearity=g_hid,
                    name="hidden_dec_layer_{}".format(hid_count)
                ))

            self.l_hid_mu_dec = self.dropout(lasagne.layers.DenseLayer(
                curr_l_hid,
                num_units=hid_dim,
                num_leading_axes=num_leading_axes,
                nonlinearity=g_hid,
                name="hidden_dec_layer_mu"
            ))
            self.l_hid_log_sigma_dec = self.dropout(lasagne.layers.DenseLayer(
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
        print("all layers: ",lasagne.layers.get_all_layers(self.l_out))

    @utils.cached_property
    def input_dim(self):
        if config.regression_type == "user":
            return self.dataset.M
        elif config.regression_type == "item":
            return self.dataset.N
        elif config.regression_type == "user+item":
            return self.dataset.M + self.dataset.N
        else:
            raise Exception("config.regression_type not valid")

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
        return T.cast(T.sum(self.mask),'float32')

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
        return ret

    @utils.cached_property
    def mask_dec_W(self):
        ret = T.tile(self.mask_mb_sum, (hid_dim,1)) # notice absence of transpose
        return ret

    @utils.cached_property
    def loss_sq(self):
        ret = T.sqr(self.Ri_mb_sym - self.predict_to_1_lea)
        return ret

    @utils.cached_property
    def regression_error_obj(self):
        # a.k.a. likelihood term
        # this is the expected reconstruction error
        # the 1/2 coefficient is external to this term,
        # being config.regression_error_coef
        term_constant= - self.mask_sum * np.array(2*np.pi).astype('float32')
        masked_log_sigma = self.out_log_sigma_lea * self.mask
        term_logdetsigma= - T.sum(masked_log_sigma)
        sigma = T.exp(self.out_log_sigma_lea)
        masked_loss_sq = self.loss_sq * self.mask
        inv_sigma = 1./sigma
        term_scaled_error= - T.sum(masked_loss_sq * inv_sigma)

        # the terms have all minuses inside
        ret = term_constant + term_logdetsigma + term_scaled_error

        ret = scalar(ret) # to scalar
        ret.name = "regression_error_obj"
        return ret

    @utils.cached_property
    def latentK_term_obj(self):
        term_constant= - 0.5 * self.mask_sum * np.array(2*np.pi).astype('float32')
        term_l2 = - 0.5 * T.sum(self.latentK_lea ** 2)
        ret = term_constant + term_l2
        ret.name = "latentK_term_obj"
        ret = scalar(ret)
        return ret

    @utils.cached_property
    def latent0_entropy_term_obj(self):
        term_constant = 0.5 * self.mask_sum * np.array(2*np.pi).astype('float32')
        term_dim = 0.5 * latent_dim
        masked_log_sigma = self.latent0_log_sigma_lea
        term_logdetsigma = 0.5 * T.sum(masked_log_sigma)
        ret = term_constant + term_dim + term_logdetsigma
        ret.name = "latent0_entropy_term_obj"
        ret = scalar(ret)
        return ret

    @utils.cached_property
    def transformation_ws(self):
        ret = map(lambda l: l.w, self.l_transformations)
        return ret

    @utils.cached_property
    def transformation_bs(self):
        ret = map(lambda l: l.b, self.l_transformations)
        return ret

    @utils.cached_property
    def transformation_us(self):
        ret = map(lambda l: l.u, self.l_transformations)
        return ret

    @utils.cached_property
    def transformation_zs(self):
        ret = map(
            lambda l: lasagne.layers.get_output(l,deterministic=False),
            self.l_transformations
        )
        return ret

    @utils.cached_property
    def transformation_term_obj(self):
        ret = 0
        zkminusone = self.latent0_sample_lea
        h = g_transform
        for k,w,b,u,z in zip(
                range(TK),
                self.transformation_ws,
                self.transformation_bs,
                self.transformation_us,
                self.transformation_zs
        ):
            zkminusone.name = "z_{}".format(k-1)
            zw = T.dot(zkminusone,w)
            a = zw + b
            a_range = T.arange(a.shape[0])
            h_a = h(a)
            #h_prime_output = T.grad(h_a,a)

            def do_grad_h (i,ai):
                #ai = a[i,:]
                hai = h(ai)
                shai = scalar(hai)
                print("shai",shai)
                gr = T.grad( shai, ai)
                return gr

            hprime,_updates = theano.scan(
                fn=do_grad_h,
                sequences=[a_range,a],
            )
            d = T.dot(w.T,u)

            m = hprime * d
            s = 1+m
            ret += T.log(T.abs_(s))
            zkminusone = z
        ret.name = "transformation_term_obj"
        ret = scalar(T.sum(ret))
        return ret

    @utils.cached_property
    def excluded_loss(self):
        ret = (self.loss_sq * (1-self.mask)).mean()
        return ret

    @utils.cached_property
    def elbo(self):
        ret = 0
        ret += self.regression_error_obj * config.regression_error_coef
        ret += self.latentK_term_obj
        ret += self.latent0_entropy_term_obj
        ret += self.transformation_term_obj
        ret.name = "elbo_before_sum"
        ret = scalar(T.sum(ret))
        ret.name = "elbo_after_sum"
        return ret

    @utils.cached_property
    def obj(self):
        ret = - self.elbo
        ret /= config.minibatch_size # it's an average!
        if config.regularization_lambda > 0.:
            ret += self.regularizer * config.regularization_lambda / self.n_datapoints
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
    def latent0_sample_lea(self):
        ret = lasagne.layers.get_output(self.l_latent0_sampling,deterministic=False)
        return ret

    @utils.cached_property
    def latent0_sample_det(self):
        ret = lasagne.layers.get_output(self.l_latent0_sampling,deterministic=True)
        return ret

    @utils.cached_property
    def latentK_lea(self):
        ret = lasagne.layers.get_output(self.l_transformations[-1],deterministic=False)
        return ret

    @utils.cached_property
    def latentK_det(self):
        ret = lasagne.layers.get_output(self.l_transformations[-1],deterministic=True)
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
            grad = T.grad(self.obj,curr)

            # filtering first and last layer's gradients according
            # to which ratings were observed
            if curr.name in self.all_masks.keys():
                grad = grad * self.all_masks[curr.name]

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
        ret = lasagne.regularization.regularize_network_params(
            self.l_out,
            weights_regularization
        )
        return ret

def main():
    dataset = movielens.load(config.movielens_which)

    def make_predict_to_5(predict_to_1_sym):
        ret = cftools.unpreprocess(predict_to_1_sym,dataset)
        return ret

    print("creating model..")

    model = Model(dataset)
    print("parameters shapes:",[p.get_value().shape for p in model.params])

    predict_to_1_fn = theano.function( # FIXME: change name
        [model.Ri_mb_sym],
        [model.predict_to_1_det]
    )
    predict_to_1_fn.name="predict_to_1_fn"

    predict_to_5_fn = theano.function(
        [model.Ri_mb_sym],
        [make_predict_to_5(model.predict_to_1_det)]
    )
    predict_to_5_fn.name="predict_to_5_fn"

    Ri_mb_l = []
    indices_mb_l = []

    total_loss = 0
    def epoch_hook(*args,**kwargs):
        def meanstd(quantity):
            v = quantity.get_value()
            m = np.mean(v,axis=None)
            s = np.std(v,axis=None)
            print(quantity.name,"mean:",m,"std:",s)
        nonlocal total_loss
        print("\ntotal_loss:",total_loss)
        for curr in model.params:
            meanstd(curr)
        print("\n\n")

        total_loss = 0

    def train_with_rrow(i,Ri,lr): # Ri is an entire sparse row of ratings from a user
        nonlocal indices_mb_l
        nonlocal Ri_mb_l
        nonlocal total_loss

        indices_mb_l.append((i,))
        Ri_mb_l.append(Ri)
        if len(Ri_mb_l) >= config.minibatch_size:
            Ri_mb = scipy.sparse.vstack(Ri_mb_l).astype('float32')
            Ri_mb.data = cftools.preprocess(Ri_mb.data,dataset) # FIXME: method of Dataset?
            _loss, = params_update_fn(Ri_mb)
            total_loss += _loss
            Ri_mb_l = []
            indices_mb_l = []
    looper = cftools.LooperRrows(
        train_with_rrow,
        dataset,
        predict_to_5_fn,
        epoch_hook=epoch_hook
    )
    # model needs n_datapoints to divide regularizing lambda
    model.n_datapoints = looper.n_datapoints

    print("creating parameter updates...")
    params_updates = adam_masked(
        model.grads_params,
        model.params,
        model.all_masks,
        learning_rate=config.lr_begin * config.minibatch_size
    )

    print("creating parameter update function..")
    params_update_fn = theano.function(
        [model.Ri_mb_sym],
        [model.regression_error_obj],
        updates=params_updates
    )
    params_update_fn.name = "params_update_fn"
    print("done.")
    print("training ...")
    looper.start()
if __name__=="__main__":
    main()

