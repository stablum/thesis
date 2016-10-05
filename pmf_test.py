#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random

def main():
    K=8
    lr = 0.0001
    R0 = np.array([
        [1, 0, 5],
        [1, 1, 5],
        [3, 3, 1],
        [3, 3, 0],
        [3, 0, 2],
        [4, 0, 5],
        [4, 3, 0]
    ])
    _mask = (R0==0)
    R = np.ma.array(
        R0,
        mask=_mask
    )
    U = np.random.random((K,R.shape[0]))
    V = np.random.random((K,R.shape[1]))

    def newE():
        ret =  R - np.dot(U.T,V)
        print "total error",np.sum(np.abs(ret))
        return ret

    for _ in xrange(100000):
        #print "E",E
        #print "U.T . V",np.dot(U.T,V)
        #sel_is = np.random.randint(0,R0.shape[0],R0.shape[1])
        #sel_js = np.random.randint(0,R0.shape[1],R0.shape[0])
        sel_is = []
        sel_js = []
        E = newE()
        for j in range(R.shape[1]):
            boolcol = R.mask[:,j]
            is_avail = np.linspace(0,R.shape[0],R.shape[0]+1)[-boolcol]
            _ri = random.randint(0,len(is_avail)-1)
            _r = is_avail[_ri]
            sel_is.append(int(_r))
        for i in range(R.shape[0]):
            boolrow = R.mask[i,:]
            is_avail = np.linspace(0,R.shape[1],R.shape[1]+1)[-boolrow]
            _ri = random.randint(0,len(is_avail)-1)
            _r = is_avail[_ri]
            sel_js.append(int(_r))

        Ejs = np.choose(sel_is,newE())
        #print "Ejs",Ejs
        #print "sel_is",sel_is
        #import ipdb; ipdb.set_trace()
        V = V + lr * Ejs
        Eis = np.choose(sel_js,newE().T)
        U = U + lr * Eis
        #ipdb.set_trace()
        #Ejs = np.mean(newE(),0)
        #Eis = np.mean(newE(),1)
    print "E",newE()
    print "old R",R
    print "new R",np.dot(U.T,V)

if __name__=="__main__":
    main()
