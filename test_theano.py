#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
import sys
from tqdm import tqdm
import theano
from theano import tensor as T
from theano import function

# local imports

import movielens
import cftools
import config

theano.config.exception_verbosity='high'

def main():
    a_vec = T.dvector('a_vec')
    b_vec = T.dvector('b_vec')
    zucca = T.dscalar('zucca')

    def step(a,b,z):
        return a+b+z

    values, updates = theano.scan(
        step,
        sequences=[a_vec,b_vec],
        non_sequences=[zucca]
    )

    scan_fn = function(
        [a_vec,b_vec,zucca],
        [values],
        updates=updates,
        mode='FAST_RUN'
    )

    foo = scan_fn(
        np.array([1,2]),
        np.array([3,4]),
        42
    )

    print('foo %s'%(str(foo)))

if __name__=="__main__":
    main()

