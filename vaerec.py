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
import persistency

#update = adam_masked = update_algorithms.adam_masked # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g_rij = activation_functions.get(config.g_rij)
g_latent = activation_functions.get(config.g_latent)
g_hid = activation_functions.get(config.g_hid)
g_log_sigma = lasagne.nonlinearities.linear
weights_regularization = regularization.get(config.regularization_type)
sigma = 1.
sigma_u = 100.
sigma_v = 1000.
num_leading_axes = 1

chan_out_dim = config.chan_out_dim
hid_dim = config.hid_dim
latent_dim = config.K
#log = print
log = lambda *args: print(*args)#None

class Model(object):

    def __init__(self,dataset):
        # having a reference to the dataset is required as some information such
        # as the number of items is needed for the network structure
        self.dataset = dataset

        self.Ri_mb_sym = theano.sparse.csr_matrix(name='Ri_mb',dtype='float32')
        self.make_net()
        self._n_datapoints = None # to be set later

    @utils.cached_property
    def update(self):
        ret = update_algorithms.get_func()
        return ret

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

        self.l_latent_mu = latent_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_latent,
            name="latent_mu"
        )
        self.l_latent_log_sigma = latent_layer_type(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_log_sigma,
            name="latent_log_sigma"
        )
        self.l_latent_merge = lasagne.layers.ConcatLayer(
            [
                self.l_latent_mu,
                self.l_latent_log_sigma
            ],
            name="latent_concat"
        )
        self.l_latent_sampling = model_build.SamplingLayer(
            self.l_latent_merge,
            dim=latent_dim,
            name="latent_sampling"
        )

        if config.n_hid_layers == 0:
            self.l_hid_mu_dec = self.l_hid_log_sigma_dec = self.l_latent_sampling
        else: # 1 or multiple hidden layers
            curr_l_hid = self.l_latent_sampling
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
        log("all layers: ", self.all_layers)

    @property
    def all_layers(self):
        return lasagne.layers.get_all_layers(self.l_out)

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
    def likelihood(self):
        # this is the expected reconstruction error
        # the 1/2 coefficient is external to this term,
        # being config.regression_error_coef
        # a.k.a. likelihood!!!
        term1 = -self.mask_sum * np.array(2*np.pi).astype('float32')
        masked_log_sigma = self.out_log_sigma_lea * self.mask
        term2 = -T.sum(masked_log_sigma)
        sigma = T.exp(self.out_log_sigma_lea)
        masked_loss_sq = self.loss_sq * self.mask
        inv_sigma = 1./sigma
        term3 = - T.sum(masked_loss_sq * inv_sigma)
        ret = term1 + term2 + term3
        ret = ret.reshape((),ndim=0) # to scalar
        return ret

    @utils.cached_property
    def regularizer_latent_kl(self):
        sigma = T.exp(self.latent_log_sigma_lea)
        ret = kl.kl_normal_diagonal_vs_unit(self.latent_mu_lea,sigma,latent_dim)
        return ret

    @utils.cached_property
    def marginal_latent_kl(self):
        term1 = - self.latent_log_sigma_lea
        term2 = 0.5*(T.pow(self.latent_sigma_lea,2) + T.pow(self.latent_mu_lea,2))
        term3 = - 0.5
        return term1 + term2 + term3

    @utils.cached_property
    def marginal_latent_kl_mean(self):
        return T.mean(self.marginal_latent_kl)

    @utils.cached_property
    def marginal_latent_kl_std(self):
        return T.std(self.marginal_latent_kl)

    @utils.cached_property
    def excluded_loss(self):
        ret = (self.loss_sq * (1-self.mask)).mean()
        return ret

    @utils.cached_property
    def elbo(self):
        """
        the ELBO is expressed as two terms: likelihood, which is positive
        (has to be maximized)
        and KL between approximate posterior and prior on latent
        which is negative (has to be minimized)
        """
        ret = self.likelihood * config.regression_error_coef
        if config.regularization_latent_kl > 0.:
            ret -= self.regularizer_latent_kl * config.regularization_latent_kl
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
    def latent_mu_lea(self):
        ret = lasagne.layers.get_output(self.l_latent_mu,deterministic=False)
        return ret

    @utils.cached_property
    def latent_log_sigma_lea(self):
        ret = lasagne.layers.get_output(self.l_latent_log_sigma,deterministic=False)
        return ret

    @utils.cached_property
    def latent_sigma_lea(self):
        ret = T.exp(self.latent_log_sigma_lea)
        return ret

    @utils.cached_property
    def predict_to_1_lea(self):
        ret = self.out_mu_lea
        return ret

    @utils.cached_property
    def predict_to_1_det(self):
        ret = self.out_mu_det
        return ret

    @property
    def params_for_persistency(self):
        params_values = lasagne.layers.get_all_param_values(self.all_layers)
        return params_values

    @params_for_persistency.setter
    def params_for_persistency(self,params):
        lasagne.layers.set_all_param_values(self.all_layers, params)

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
        ret = lasagne.regularization.regularize_layer_params(
            self.all_layers,
            weights_regularization
        )
        return ret

    @property
    def params_updates_values(self):
        ret = []
        for k in list(self.params_updates.keys()):
            ret.append(k.get_value())
        return ret

    @params_updates_values.setter
    def params_updates_values(self,vals):
        for new_value,k in zip(vals,list(self.params_updates.keys())):
            k.set_value(new_value)

    @property
    def params_updates(self):
        log("creating parameter updates...")
        if '_params_updates' not in dir(self):
            self._params_updates = self.update (
                self.grads_params,
                self.params,
                self.all_masks,
                learning_rate=config.lr_begin * config.minibatch_size
            )
        return self._params_updates

    @params_updates.setter
    def params_updates(self,val):
        assert val is not None
        self._params_updates = val

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
    log("parameters shapes:",[p.get_value().shape for p in model.params])

    log("creating marginal_latent_kl diagnostic functions..")
    marginal_latent_kl_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [
            model.marginal_latent_kl
        ],
    )
    marginal_latent_kl_fn.name = "marginal_latent_kl_fn"

    log("creating out_log_sigmas diagnostic functions..")
    out_log_sigmas_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [
            model.out_log_sigma_lea
        ],
    )
    out_log_sigmas_fn.name = "marginal_latent_kl_fn"

    log("done.")
    theano.printing.pprint(model.predict_to_1_det)

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

    def epoch_hook(*args,**kwargs):
        _log = kwargs.pop('log',log)
        _epochsloop = kwargs['epochsloop']
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
            _obj = obj_fn(curr)
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

        persistency.save(model,kwargs['lr'],kwargs['epochsloop'].epoch_nr)

    def train_with_rrow(i,Ri,lr): # Ri is an entire sparse row of ratings from a user
        nonlocal indices_mb_l
        nonlocal Ri_mb_l
        nonlocal total_loss
        nonlocal total_kls
        nonlocal total_likelihoods
        nonlocal total_out_log_sigmas

        indices_mb_l.append((i,))
        Ri_mb_l.append(Ri)
        if len(Ri_mb_l) >= config.minibatch_size:
            Ri_mb = scipy.sparse.vstack(Ri_mb_l)
            Ri_mb.data = cftools.preprocess(Ri_mb.data,dataset) # FIXME: method of Dataset?
            _loss, = params_update_fn(Ri_mb)
            _kls, = marginal_latent_kl_fn(Ri_mb)
            _out_log_sigmas, = out_log_sigmas_fn(Ri_mb)
            _obj, = obj_fn(Ri_mb)
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
    params_update_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [model.obj],
        updates=model.params_updates
    )
    params_update_fn.name = "params_update_fn"

    obj_fn = model_build.make_function(
        [model.Ri_mb_sym],
        [model.obj]
    )
    obj_fn.name = "obj_fn"

    log("training ...")
    #import ipdb; ipdb.set_trace()
    looper.start()

if __name__=="__main__":
    main()

