#!/usr/bin/env python3

import numpy as np
import lasagne
import theano
import theano.sparse
from theano import tensor as T
import random
import lasagne_sparse
import scipy
import scipy.sparse
from mnist import MNIST
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

n_epochs = 1000
n_points = 60000
n_hid=100
latent_dim=30#3
lr = 0.01#0.0001
noise=0.0001
point_len = 784
sparsity = 0.05#0.20
hid_g = lasagne.nonlinearities.elu
cherries = [300,400,500,600,700,800,900,1000,1100,1200,1300]
def load_dataset():
    mndata = MNIST('/home/francesco/reimplementations/python-mnist/data/')
    mndata.load_training()
    mndata.load_testing()
    return mndata.train_images

def create_sparse_training_set():

    training_set_lists = load_dataset()
    #sparse_training_set = scipy.sparse.csr_matrix((n_points,point_len))
    #lil_training_set = scipy.sparse.lil_matrix((n_points,point_len))
    tmp_training_set = scipy.sparse.lil_matrix((n_points,point_len))
    n_sparse_features = int(point_len * sparsity)
    print("n_sparse_features",n_sparse_features)
    original_training_set = []
    for i,row in enumerate(tqdm(training_set_lists,desc='creating sparse training set')):
        feature_indexes = np.random.choice(range(point_len),n_sparse_features,False)
        reranged_row = 2.*(-0.5 + np.array(row)/255.)
        original_training_set.append(reranged_row)
        for j in feature_indexes:
            tmp_training_set[i,j] = reranged_row[j]

        if i in cherries:
            plot_prediction(np.array(reranged_row),i,'orig')
            plot_prediction(tmp_training_set[i,:].todense(),i,'sparse')
    print("nnz",tmp_training_set.nnz)
    print("converting to csr...")
    csr_training_set = tmp_training_set.tocsr()
    print("done.")
    return csr_training_set,original_training_set

def main():
    sparse_training_set,original_training_set = create_sparse_training_set()

    #input_var = T.dvector('input_var')
    input_sparse_var = theano.sparse.csr_matrix(name='a_sparse_input_var')
    print("input_sparse_var",
          input_sparse_var,
          input_sparse_var.name,
          input_sparse_var.type,
          input_sparse_var.type.ndim
          )

    l_in = lasagne.layers.InputLayer((1,point_len,),input_var=input_sparse_var,name="input_layer")
    print("l_in_out",lasagne.layers.get_output(l_in))
    print("l_in.shape",l_in.shape)

    l_hid_enc = lasagne_sparse.SparseInputDenseLayer(
        l_in, num_units=n_hid,num_leading_axes=0,
        nonlinearity=hid_g,
        name="hidden_enc_layer"
    )
    print("l_hid_enc_out",lasagne.layers.get_output(l_hid_enc))

    l_latent = lasagne.layers.DenseLayer(
        l_hid_enc, num_units=latent_dim,num_leading_axes=0,
        nonlinearity=hid_g,
        name="latent_layer"
    )
    l_hid_dec = lasagne.layers.DenseLayer(
        l_latent, num_units=n_hid,num_leading_axes=0,
        nonlinearity=hid_g,
        name="hidden_dec_layer"
    )
    l_out = lasagne.layers.DenseLayer(
        l_hid_dec, num_units=point_len,num_leading_axes=0,
        nonlinearity=hid_g,
        name="out_layer"
    )
    params = lasagne.layers.get_all_params(l_out, trainable=True)

    prediction = lasagne.layers.get_output(l_out)
    prediction = prediction.dimshuffle('x',0)
    prediction = theano.tensor.specify_shape(prediction, (1, point_len))

    print("prediction",prediction.type,prediction.type.ndim)
    input_dense_var = theano.sparse.dense_from_sparse(input_sparse_var)
    mask_plus = input_dense_var > 0.0001
    mask_minus = input_dense_var < -0.0001
    mask = mask_plus + mask_minus
    loss_diff = input_sparse_var - prediction
    loss_sq = (loss_diff) ** 2
    loss = (loss_sq * mask).mean()
    excluded_loss = (loss_sq * (1-mask)).mean()
    #loss = (loss_sq).mean()

    updates = lasagne.updates.adam(
        loss, params, learning_rate=lr)

    train_fn = theano.function([input_sparse_var], [loss,mask,loss_sq,excluded_loss,prediction], updates=updates)

    perm = list(range(n_points))
    #print("params BEFORE",[p.get_value() for p in params])
    for epoch in tqdm(range(n_epochs),desc="epochs"):
        random.shuffle(perm)
        total_loss = 0
        total_excluded_loss = 0
        total_reconstruction_error = 0
        for i in tqdm(perm,desc="datapoints"):
            x = sparse_training_set[[i],:]
            _loss,_mask,_loss_sq,_excluded_loss,_prediction = train_fn(x)
            if i in cherries:
                plot_prediction(_prediction,i,epoch)
            #print(_loss_sq.mean(),_mask.sum(),_loss,_excluded_loss.mean())
            total_loss += _loss
            total_excluded_loss += _excluded_loss
            original_row = original_training_set[i]
            reconstruction_error = np.mean((original_row - _prediction) ** 2)
            total_reconstruction_error += reconstruction_error
        #print("params",[p.get_value() for p in params])
        print("\n\ntotal loss",total_loss)
        print("total excluded loss",total_excluded_loss)
        print("total reconstruction error",total_reconstruction_error)
        print("average reconstruction error",total_reconstruction_error/n_points)
        print("\n\n")

def plot_prediction(_prediction,cherry,epoch):
    raster = np.reshape(_prediction,(28,28))
    clipped = np.clip(raster,-1,1.)
    plt.imshow(clipped,interpolation='none')
    plt.colorbar()
    plt.clim(-1.0, 1.0)
    plt.savefig("/tmp/lr_{}_latent_dim_{}_cherry_{}_epoch_{}.png".format(lr,latent_dim,cherry,epoch))
    plt.cla(); plt.clf()

if __name__ == "__main__":
    main()

