#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random

def main():
    K=3
    lr = 0.0001
    R0 = np.array([
        [1, 0, 5,1],
        [1, 1, 5,1],
        [1, 1, 4,1],
        [3, 3, 0,3],
        [3, 0, 1,3],
        [5, 0, 1,5],
        [5, 5, 1,5],
        [5, 1, 1,5],
        [5, 2, 1,5],
        [5, 4, 1,5],
        [5, 3, 0,5],
        [5, 2, 1,5],
        [0, 3, 1,5],
        [5, 0, 0,5],
        [5, 5, 0,5],
        [5, 4, 0,5],
        [5, 3, 0,5],
        [5, 2, 0,5],
        [5, 1, 0,5],
        [0, 0, 5,5],
        [0, 3, 0,3],
        [0, 1, 0,3],
        [0, 5, 0,3],
    ])
    _mask = (R0==0)
    R = np.ma.array(
        R0,
        mask=_mask
    )
    U = np.random.random((K,R.shape[0]))
    V = np.random.random((K,R.shape[1]))

    def new_eij():
        ret = R[i,j] - np.dot(U[:,i].T, V[:,j])
        print "eij",ret
        return ret

    for _ in xrange(1000000):
        wis,wjs = np.where(-R.mask)
        pos = random.randint(0,len(wis)-1)
        i,j = wis[pos],wjs[pos]
        eij = new_eij()
        V[:,j] = V[:,j] + lr * eij * U[:,i]
        eij = new_eij()
        U[:,i] = U[:,i] + lr * eij * V[:,j]
    print "old R",R
    newR = np.round(np.dot(U.T,V)).astype('int')
    print "new R",newR
    print "diff",R-newR
if __name__=="__main__":
    main()
