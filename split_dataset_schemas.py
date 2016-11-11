import numpy as np
import random
import config
import utils

class PermList(object):
    """
    this class prevents creating another permutated list
    by hijacking a sequential indexing.
    """

    def __init__(self, l, perm):
        assert len(l) == len(perm)
        self.l = l
        self.perm = perm

    def __len__(self):
        return len(self.l)

    def __getitem__(self, prev_index):
        actual_index = self.perm[prev_index]
        return self.l[actual_index]

class Splitter(object):
    """
    Represents a dataset split into a training set and a testing set.
    Moreover, the subclasses implement the preparation of a new training
    set for each epoch (possibly, trough random permutation of the sole training
    set)
    """
    def __init__(self,dataset):
        self.dataset = dataset

    def prepare_new_training_set(self):
        raise Exception("prepare_new_training_set needs to be implemented!")

class MemoryRandomCompleteEpochs(Splitter):
    """
    the dataset is going to be entirely stored in memory.
    Each training set is randomized (permutation) at each epoch, and it's given
    as complete (not a subset of training set).
    """
    def __init__(self,dataset):
        super().__init__(dataset)
        self._validation_set = self.entire[:self.splitpoint]
        self._training_set = self.entire[self.splitpoint:]

    @utils.cached_property
    def entire(self):
        ret = self.dataset.read_entire()
        random.shuffle(ret)
        return ret

    @property
    def splitpoint(self):
        ret = int(len(self.entire) * config.validation_set_fraction)
        return ret

    @property
    def validation_set(self):
        return self._validation_set

    def prepare_new_training_set(self):
        perm = np.random.permutation(len(self._training_set))
        self._training_set_perm = PermList(self._training_set,perm)

    @property
    def training_set(self):
        return self._training_set_perm

class Chunky(Splitter):
    def __init__(self,dataset):
        super().__init__(dataset)
        self._validation_set = None

    @property
    def validation_set(self):
        """
        lazy getter
        """
        if self._validation_set is None:
            self._validation_set = self.dataset.chunk(0)
        return self._validation_set

class ChunkyRandom(Chunky):

    def __init__(self,dataset):
        self._training_set = None
        super().__init__(dataset)

    def prepare_new_training_set(self):
        chunk_id = np.random.randint(1,dataset.num_chunks)
        chunk_training = self.dataset.chunk(chunk_id)

        perm = np.random.permutation(len(chunk_training))
        self._training_set = PermList(chunk_training,perm)

    @property
    def training_set(self):
        if self.training_set is None:
            self.prepare_new_training_set(self)
        return self._training_set

class ChunkyRandomCompleteEpochs(Chunky):
    def __init__(self,dataset):
        super().__init__(dataset)
        self.training_chunks_permutation = None

    def prepare_new_training_set(self):
        self.training_chunks_permutation = list(np.random.permutation(self.dataset.num_chunks))
        self.training_chunks_permutation.remove(0) # 0 is the testing set
        self.training_datapoints_permutations = {} # indexed by chunk_id

    @property
    def training_set(self):

        if self.training_chunks_permutation is None:
            self.new_training_set()

        def _training_set_generator():
            for chunk_id in self.training_chunks_permutation:
                chunk = self.dataset.chunk(chunk_id)
                if chunk_id not in self.training_datapoints_permutations.keys():
                    self.training_datapoints_permutations[chunk_id] = np.random.permutation(len(chunk))
                for datapoint in PermList(chunk,self.training_datapoints_permutations[chunk_id]):
                    yield datapoint

        return _training_set_generator()

def main():
    import movielens
    print("testing splitting the movielens dataset. Let's see if it's done properly\n\n")
    dataset = movielens.load('1m')
    tr,va= chunky_random_complete_epochs()
    print("\nTRAINING SET....\n\n")
    i=0
    for datapoint in tr(dataset):
        i = i + 1
        print("training set datapoint {}".format(i))

    print("\ndone. VALIDATION SET....\n\n")

    i=0
    for datapoint in va(dataset):
        i = i + 1
        print("validation set datapoint {}".format(i))
    print("\nALL DONE.")

if __name__=="__main__":
    main()
