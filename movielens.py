#!/usr/bin/env python

import os
import pandas as pd
import scipy
import scipy.sparse

from tqdm import tqdm

small_path = os.path.join(".","ml-latest-small","ratings.csv")

def new_index(l):
    # returns a dictionary with old_index:new_compressed_index entries
    # this because the new ratings matrix should be as small as possible

    d = dict(enumerate(sorted(list(set(l)))))
    ret = {
        v: k
        for k, v
        in d.items()
    }
    return ret

def flat_to_R(flat):

    # making straightforward movieId -> column id access possible
    user_new_index = new_index(flat.userId)
    movie_new_index = new_index(flat.movieId)

    N = len(user_new_index)
    M = len(movie_new_index)
    R = scipy.sparse.dok_matrix((N,M))
    for index, row in tqdm(list(flat.iterrows())):
        i = user_new_index[row.userId]
        j = movie_new_index[row.movieId]
        R[i,j] = row.rating
    return R

def small():
    print "loading %s ..."%small_path
    flat = pd.read_csv(small_path)
    print flat.columns

    print "converting to sparse matrix ..."
    R = flat_to_R(flat)
    print "converted to sparse matrix."
    return R

def main():
    pass

if __name__=="__main__":
    main()
