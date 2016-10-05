#!/usr/bin/env python

import os
import sys
import pandas as pd
import scipy
import scipy.sparse
import config

from tqdm import tqdm
tqdm.pandas()

paths = {}
paths['small'] = os.path.join(".","ml-latest-small","ratings.csv")
paths['22m'] = os.path.join(".","ml-latest","ratings.csv.shuf")
paths['1m'] = os.path.join(".","ml-1m","ratings.dat.csv.shuf")

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

class DataSet(object):
    def __init__(self, path, chunk_len=128*1024):
        self.path = path
        self.chunk_len = chunk_len

        user_id_set = set()
        movie_id_set = set()

        self.num_chunks = 0
        self.num_ratings = 0.

        print("loading %s ..."%path)
        reader = self.reopen()
        for chunk in tqdm(reader,desc="DataSet init: iterating chunks"):
            self.num_chunks += 1
            user_id_set.update(chunk.userId)
            movie_id_set.update(chunk.movieId)

            self.num_ratings += float(len(chunk))

        self.user_new_index = new_index(user_id_set)
        self.movie_new_index = new_index(movie_id_set)

        print("N",self.N)
        print("M",self.M)
        print("N_compressed",self.N_compressed)
        print("M_compressed",self.M_compressed)

    def chunk(self, chunk_id):
        reader = self.reopen()
        for curr_id, df in tqdm(enumerate(reader),desc="finding chunk_id={}".format(chunk_id)):
            if curr_id == chunk_id:
                ret = []
                for index, row in tqdm(df.iterrows(),desc="getting ratings in chunk"):
                    i = self.user_new_index[row.userId]
                    j = self.movie_new_index[row.movieId]
                    Rij = row.rating
                    ret.append(((i,j),Rij))
                return ret
        raise Exception("DataSet did not find chunk_id={}".format(chunk_id))

    @property
    def N(self):
        return len(self.user_new_index)

    @property
    def M(self):
        return len(self.movie_new_index)

    @property
    def N_compressed(self):
        return self.num_ratings/self.M

    @property
    def M_compressed(self):
        return self.num_ratings/self.N

    def reopen(self):
        return pd.read_csv(self.path,iterator=True,chunksize=self.chunk_len)

def flat_to_R(dataset): # FIXME DELME

    # making straightforward movieId -> column id access possible
    R = scipy.sparse.dok_matrix((N,M))

    for chunk in tqdm(reader,desc="getting the ratings chunk by chunk"):
        #for index, row in tqdm(list(chunk.iterrows()),desc="getting ratings in chunk"):
        for index, row in tqdm(chunk.iterrows(),desc="getting ratings in chunk"):
        #for index, row in chunk.iterrows():
            i = user_new_index[row.userId]
            j = movie_new_index[row.movieId]
            R[i,j] = row.rating
    return R

def load(name):
    path = paths[name]
    dataset = DataSet(path,chunk_len=config.chunk_len)
    return dataset

def main():
    dataset = DataSet(paths[sys.argv[1]],chunk_len=int(sys.argv[2]))

if __name__=="__main__":
    main()
