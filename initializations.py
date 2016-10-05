import numpy as np

def uniform(shape):
    ret = np.random.random(shape).astype('float32')
    return ret

def normal(shape):
    ret = np.random.normal(0,1,shape).astype('float32')
    return ret

