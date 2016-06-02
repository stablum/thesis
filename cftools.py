import numpy as np

def split_sets(R):
    Ritems = R.items()
    sel = np.random.permutation(len(Ritems))
    midpoint = int(len(Ritems)/2)
    training_set = []
    testing_set = []
    for curr in sel[:midpoint]:
        training_set.append(Ritems[curr])
    for curr in sel[midpoint+1:]:
        testing_set.append(Ritems[curr])
    return training_set, testing_set

def rating_error(Rij,U,i,V,j):
    ret = Rij - np.dot(U[:,i].T, V[:,j])
    return ret

def rmse(subset,U,V):
    errors = []
    for curr in subset:
        (i,j),Rij = curr
        eij = rating_error(Rij,U,i,V,i)
        errors.append(eij**2)
    return np.sqrt(np.mean(errors))

