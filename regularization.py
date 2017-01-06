import lasagne

regularizers = {
    'L1' : lasagne.regularization.l1,
    'L2' : lasagne.regularization.l2,
}

def get(regularization_type):
    if regularization_type in regularizers.keys():
        return regularizers[regularization_type]
    else:
        raise Exception('unknown regularization_type {}'.format(regularization_type))


