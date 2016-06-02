#!/usr/bin/env python

import scipy
import ipdb
import numpy as np
import random
from tqdm import tqdm

# local imports

import movielens
import cftools

def wrong(x):
    return ((not np.isfinite(x)) or np.isnan(x) or x>10000. or x<-10000.)

def main():
    np.set_printoptions(precision=4, suppress=True)

    K=100
    lr = 0.001
    n_epochs = 1000
    lambda_v = 1.
    lambda_u = 1.

    mu_u = np.random.random(K)*10. - 5.
    mu_v = np.random.random(K)*10. - 5.

    R = movielens.small()

    N = R.shape[0]
    M = R.shape[1]
    U = np.random.random((K,N))
    V = np.random.random((K,M))

    def new_eij():
        ret = cftools.rating_error(Rij,U,i,V,j)
        #print "eij",ret
        return ret

    training_set, testing_set = cftools.split_sets(R)

    print "training pmf with mu hyperpriors..."
    for _ in tqdm(range(n_epochs)):
        random.shuffle(training_set)
        for curr in tqdm(training_set):
            (i,j),Rij = curr

            toss = np.random.randint(5)
            if toss == 0:
                pass
            elif toss == 1:
                eijv = new_eij()
                #if wrong(eijv):
                #    print "error introduced in eijv=%s"%(str(eijv))
                #    import ipdb; ipdb.set_trace()
                grad = eijv * U[:,i] + (1./N) * lambda_v * (V[:,j] - mu_v)
                V[:,j] = V[:,j] + lr * grad
                #if any([wrong(curr) for curr in V[:,j].tolist()]):
                #    print "error introduced in V[:,%d]=%s"%(j,str(V[:,j]))
                #    import ipdb; ipdb.set_trace()

            elif toss == 2:
                eiju = new_eij()
                #if wrong(eiju):
                #    print "error introduced in eiju=%s"%(str(eiju))
                #    import ipdb; ipdb.set_trace()
                grad = eiju * V[:,j] + (1./M) * lambda_u * (U[:,i] - mu_u)
                U[:,i] = U[:,i] + lr * grad
                #if any([wrong(curr) for curr in U[:,i].tolist()]):
                #    print "error introduced in U[:,%d]=%s"%(i,str(U[:,i]))
                #    import ipdb; ipdb.set_trace()

            elif toss == 3:
                grad = lambda_v * (V[:,j] - mu_v)
                mu_v = mu_v + lr * grad

            elif toss == 4:
                grad = lambda_u * (U[:,i] - mu_u)
                mu_u = mu_u + lr * grad

        print "mu_v",mu_v
        print "mu_u",mu_u
        print "training RMSE: ",cftools.rmse(training_set,U,V)
        print "testing RMSE: ",cftools.rmse(testing_set,U,V)

if __name__=="__main__":
    main()
