#!/usr/bin/env python3

import movielens
import sys
import numpy as np
from matplotlib import pyplot as plt

def main():
    dataset = movielens.load(sys.argv[1])
    print("number of users N:",dataset.N)
    print("number of items M:", dataset.M)
    print("mean:", dataset.mean)
    print("std:", dataset.std)
    print("count:", dataset.count)
    kk = sorted(list(dataset.count.keys()))
    for k in kk:
        print(k,dataset.count[k])

    between = kk[1] - kk[0]
    bins = np.hstack([np.array(kk),np.array([kk[-1]+between])])-between/2
    ticks = np.hstack([np.array(kk),np.array([kk[-1]+between])])[:-1]
    print("bins",bins)
    print("ticks",ticks)
    plt.hist(dataset.all_ratings, bins=bins, width=between*0.75)
    plt.xticks(ticks)
    plt.savefig("text/"+sys.argv[1]+".png")
    plt.show()
    #import ipdb; ipdb.set_trace()

if __name__=="__main__":
    main()
