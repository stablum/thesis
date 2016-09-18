#!/usr/bin/env python

import os
import pandas as pd
import scipy
import scipy.sparse

from tqdm import tqdm
tqdm.pandas()

small_path = os.path.join(".","ml-latest-small","ratings.csv")
latest22m_path = os.path.join(".","ml-latest","ratings.csv")

def new_index(l):
    # returns a dictionary with old_index:new_compressed_index entries
    # this because the new ratings matrix should be as small as possible

    d = dict(enumerate(sorted(list(set(l)))))
    ret = {
        v: k
        for k, v
        in list(d.items())
    }
    return ret

def nomem_to_R(path):
    pass

def flat_to_R(reopen):

    # making straightforward movieId -> column id access possible
    user_id_set = set()
    movie_id_set = set()
    reader = reopen()
    for chunk in tqdm(reader,desc="getting all user and movie ids"):
        user_id_set.update(chunk.userId)
        movie_id_set.update(chunk.movieId)

    user_new_index = new_index(user_id_set)
    movie_new_index = new_index(movie_id_set)

    N = len(user_new_index)
    M = len(movie_new_index)
    R = scipy.sparse.dok_matrix((N,M))

    reader = reopen()
    for chunk in tqdm(reader,desc="getting the ratings chunk by chunk"):
        #for index, row in tqdm(list(chunk.iterrows()),desc="getting ratings in chunk"):
        for index, row in tqdm(chunk.iterrows(),desc="getting ratings in chunk"):
        #for index, row in chunk.iterrows():
            i = user_new_index[row.userId]
            j = movie_new_index[row.movieId]
            R[i,j] = row.rating
    return R

def load(path):
    print("loading %s ..."%path)
    reopen = lambda : pd.read_csv(path,iterator=True,chunksize=128*1024)
    reader = reopen()
    num_ratings = 0
    for chunk in tqdm(reader,desc="counting ratings"):
        num_ratings += float(len(chunk))
    print("num_ratings:",num_ratings)

    print("converting to sparse matrix ...")
    R = flat_to_R(reopen)
    print("converted to sparse matrix.")
    N = R.shape[0]
    print("N",N)
    M = R.shape[1]
    print("M",M)
    N_compressed = num_ratings/M
    print("N_compressed",N_compressed)
    M_compressed = num_ratings/N
    print("M_compressed",M_compressed)
    return R,N,M,N_compressed,M_compressed

def small():
    return load(small_path)

def latest22m():
    return load(latest22m_path)

def main():
    pass

if __name__=="__main__":
    main()
