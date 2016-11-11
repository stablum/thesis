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
g_latent = activation_functions.get('linear')
g_feat = activation_functions.get('linear')

sigma = 1.
sigma_u = 100.
sigma_v = 1000.

chan_out_dim = config.chan_out_dim
hid_dim = config.hid_dim
#log = print
log = lambda *args: print(*args)#None

class Model(object):

    def __init__(self):
        self.Rij_mb_sym = T.fmatrix('Rij_mb')

    @utils.cached_property
    def regression_error_obj(self):
        ret = 0 #FIXME
        ret = T.sum(ret)
        ret = ret.reshape((),ndim=0)
        return ret

    @utils.cached_property
    def obj(self):
        ret = self.regression_error_obj * config.regression_error_coef
        if config.regularization_lambda > 0.:
            ret += 0
        return ret

    @utils.cached_property
    def predict_to_1_lea(self):
        ret = 0.5 * 0 #FIXME
        return ret

    @utils.cached_property
    def predict_to_1_det(self):
        ret = 0.5 * 0 # FIXME
        #ret = self.vae_chan_latent_v.r_out_det
        return ret

    @utils.cached_property
    def params(self):
        ret = []
        return ret

    @utils.cached_property
    def grads_params(self):
        ret =  [
            T.grad(self.obj,curr)
            for curr
            in self.params
        ]
        return ret

def main():
    dataset = movielens.load(config.movielens_which)

    def make_predict_to_5(predict_to_1_sym):
        ret = (predict_to_1_sym * (config.max_rating - 1. )) + 1.
        return ret

    print("creating update functions..")

    model = Model()

    params_updates = adam_shared(model.grads_params,model.params,learning_rate=config.lr_begin)

    params_update_fn = theano.function(
        [model.Rij_mb_sym],
        [model.Rij_mb_sym],
        updates=params_updates
    )
    params_update_fn.name = "params_update_fn"

    if False:# FIXME
        theano.printing.pprint(model.predict_to_1_det)

        predict_to_1_fn = theano.function(
            [],
            [model.predict_to_1_det]
        )
        predict_to_1_fn.name="predict_to_1_fn"

        predict_to_5_fn = theano.function(
            [],
            [make_predict_to_5(model.predict_to_1_det)]
        )
        predict_to_5_fn.name="predict_to_5_fn"
    predict_to_5_fn = None # FIXME
    Ri_mb_l = []
    indices_mb_l = []
    def train_with_rrow(i,Ri,lr): # Ri is an entire sparse row of ratings from a user
        nonlocal indices_mb_l
        nonlocal Ri_mb_l

        indices_mb_l.append((i,))
        Ri_mb_l.append(Ri)
        if len(Ri_mb_l) >= config.minibatch_size:
            Ri_mb = scipy.sparse.vstack(Ri_mb_l)
            Ri_mb = (Ri_mb - 1.) / (config.max_rating - 1.)
            print("Ri_mb.shape",Ri_mb.shape)
            params_update_fn(Ri_mb)

            Ri_mb_l = []
            indices_mb_l = []
    print("training pmf...")
    cftools.mainloop_rrows(train_with_rrow,dataset,predict_to_5_fn)

if __name__=="__main__":
    main()

