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

update =update_algorithms.get_func()
adam_shared = lasagne.updates.adam # FIXME: generalize like the 'update' placeholder
#g = lambda x:x
g_in = activation_functions.get(config.g_in)
g_rij = activation_functions.get(config.g_rij)
g_latent = activation_functions.get('elu')
g_hid = activation_functions.get('elu')
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

    def make_net(self):
        self.l_in = lasagne.layers.InputLayer(
            (config.minibatch_size, self.dataset.M,),
            input_var=self.Ri_mb_sym,
            name="input_layer"
        )

        self.l_hid_enc = lasagne_sparse.SparseInputDenseLayer(
            self.l_in,
            num_units=hid_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_hid,
            name="hidden_enc_layer"
        )

        self.l_latent = lasagne.layers.DenseLayer(
            self.l_hid_enc,
            num_units=latent_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_latent,
            name="latent_layer"
        )

        self.l_hid_dec = lasagne.layers.DenseLayer(
            self.l_latent,
            num_units=hid_dim,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_hid,
            name="hidden_dec_layer"
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_hid_dec,
            num_units=self.dataset.M,
            num_leading_axes=num_leading_axes,
            nonlinearity=g_rij,
            name="out_layer"
        )

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
        ret = (self.Ri_mb_sym - self.predict_to_1_lea) ** 2
        return ret

    @utils.cached_property
    def regression_error_obj(self):
        masked_loss_sq = self.loss_sq * self.mask
        ret = masked_loss_sq.sum() # on both axes
        ret = ret
        ret = ret.reshape((),ndim=0) # to scalar
        return ret

    @utils.cached_property
    def excluded_loss(self):
        ret = (self.loss_sq * (1-self.mask)).mean()
        return ret

    @utils.cached_property
    def obj(self):
        ret = self.regression_error_obj * config.regression_error_coef
        if config.regularization_lambda > 0.:
            ret += 0
        return ret

    @utils.cached_property
    def out_lea(self):
        ret = lasagne.layers.get_output(self.l_out,deterministic=False)
        return ret

    @utils.cached_property
    def out_det(self):
        ret = lasagne.layers.get_output(self.l_out,deterministic=True)
        return ret

    @utils.cached_property
    def predict_to_1_lea(self):
        ret = self.out_lea
        return ret

    @utils.cached_property
    def predict_to_1_det(self):
        ret = self.out_det
        return ret

    @utils.cached_property
    def params(self):
        ret = lasagne.layers.get_all_params(self.l_out, trainable=True)
        return ret

    @utils.cached_property
    def grads_params(self):
        ret = []
        for curr in self.params:
            grad = T.grad(self.obj,curr)

            # filtering first and last layer's gradients according
            # to which ratings were observed
            if curr.name == "hidden_enc_layer.W":
                grad = grad * self.mask_enc_W
            if curr.name == "out_layer.W":
                grad = grad * self.mask_dec_W
            ret.append(grad)
        return ret

def main():
    dataset = movielens.load(config.movielens_which)

    def make_predict_to_5(predict_to_1_sym):
        ret = (predict_to_1_sym * (config.max_rating - 1. )) + 1.
        return ret

    print("creating model..")

    model = Model(dataset)
    print("parameters shapes:",[p.get_value().shape for p in model.params])
    print("creating parameter updates...")
    params_updates = adam_shared(model.grads_params,model.params,learning_rate=config.lr_begin)

    print("creating parameter update function..")
    params_update_fn = theano.function(
        [model.Ri_mb_sym],
        [model.regression_error_obj],
        updates=params_updates
    )
    params_update_fn.name = "params_update_fn"
    print("done.")

    theano.printing.pprint(model.predict_to_1_det)

    predict_to_1_fn = theano.function(
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
            Ri_mb = scipy.sparse.vstack(Ri_mb_l)

            Ri_mb.data = (Ri_mb.data - 1.) / (config.max_rating - 1.)
            _loss, = params_update_fn(Ri_mb)
            total_loss += _loss
            Ri_mb_l = []
            indices_mb_l = []
    print("training ...")
    cftools.mainloop_rrows(train_with_rrow,dataset,predict_to_5_fn,epoch_hook=epoch_hook)

if __name__=="__main__":
    main()

