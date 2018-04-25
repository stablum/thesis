#!/usr/bin/env python

import os
import sys
import pandas as pd
import scipy
import scipy.sparse
import numpy as np
import tqdm as _tqdm
_tqdm.monitor_interval = 0

import config
import utils


_tqdm.tqdm.pandas()
tqdm = lambda *args,**kwargs : _tqdm.tqdm(*args,**kwargs,mininterval=5)
search_in_dirs = ['.','..']
paths = {}
#paths['small'] = os.path.join("ml-latest-small","head2000.csv")
paths['small'] = os.path.join("ml-latest-small","ratings.csv")
paths['22m'] = os.path.join("ml-latest","ratings.csv.shuf")
paths['1m'] = os.path.join("ml-1m","ratings.dat.csv.shuf")

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

    def new_empty_lil(self):
        return scipy.sparse.lil_matrix((self.N,self.M),dtype='float32')

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

    @property
    def path(self):
        assert '_path' in dir(self), "use setter of 'path' before getting it"
        return self._path

    @path.setter
    def path(self,val):
        # absolute path, to prevent problems for subsequent chdirs
        global search_in_dirs

        for d in search_in_dirs:
            filename = os.path.join(d,val)
            abs_filename = os.path.abspath(filename)
            if os.path.isfile(abs_filename):
                self._path = abs_filename
                return
        raise Exception("could not find file {} in dirs {}".format(self.path,search_in_dirs))

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

    def read_entire(self):
        raise Exception("Please implement this method in subclass")

    @property
    def all_ratings(self):
        rr = [
            curr[1]
            for curr
            in self.read_entire()
        ]
        return rr

    @utils.cached_property
    def mean_and_std(self):
        """
        retrieving both statics at once because by caching them I have
        to retrieve all the ratings only once
        """
        rr = self.all_ratings
        mean = np.mean(rr).astype('float32')
        std = np.std(rr).astype('float32')
        return (mean,std)

    @utils.cached_property
    def mean(self):
        mean,std = self.mean_and_std
        return mean

    @utils.cached_property
    def std(self):
        mean,std = self.mean_and_std
        return std

class DataSetIndividualRatings(DataSet):
    def read_entire(self):
        """
        warning: this method loads the entire dataset into memory
        """

        reader = self.reopen()
        ret = []
        for df in tqdm(reader,desc="reading entire dataset into memory"):
            for index, row in tqdm(df.iterrows(),desc="getting ratings in chunk"):
                i = self.user_new_index[row.userId]
                j = self.movie_new_index[row.movieId]
                Rij = row.rating
                ret.append(((i,j),Rij))
        return ret

class DataSetRrows(DataSet):

    @utils.cached_property
    def R(self):
        """
        sparse ratings matrix
        """
        reader = self.reopen()
        # making straightforward movieId -> column id access possible
        lil = self.new_empty_lil()

        for chunk in tqdm(reader,desc="getting the ratings chunk by chunk"):
            #for index, row in tqdm(list(chunk.iterrows()),desc="getting ratings in chunk"):
            for index, row in tqdm(chunk.iterrows(),desc="getting ratings in chunk"):
            #for index, row in chunk.iterrows():
                i = self.user_new_index[row.userId]
                j = self.movie_new_index[row.movieId]
                if row.rating == 1:
                    lil[i,j] = 1.00001 #because zeroed entries are unobserved, and 1 will be converted to 0
                else:
                    lil[i,j] = row.rating
        csr = lil.tocsr()
        return csr

    def read_entire(self):
        ret = []
        for i in tqdm(range(self.N),desc="converting sparse matrix R to list of sparse rows"):
            ret.append((i,self.R[i,:]))
        return ret

dataset_type = globals()[config.dataset_type]
def load(name):
    path = paths[name]
    dataset = dataset_type(path,chunk_len=config.chunk_len)
    return dataset

def main():
    dataset = dataset_type(paths[sys.argv[1]],chunk_len=int(sys.argv[2]))

if __name__=="__main__":
    main()
