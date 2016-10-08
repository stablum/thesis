import numpy as np
import os
from tqdm import tqdm
import sys
import time
import random
import theano
import theano.tensor as T

# local imports

import config
import update_algorithms

def create_training_set_matrix(training_set):
    print("creating training set matrix..")
    return np.array([
        [_i,_j,_Rij]
        for (_i,_j),_Rij
        in tqdm(training_set)
    ])

def create_training_set_apart(training_set):
    i_l = []
    j_l = []
    Rij_l = []
    print("creating training set vectors..")
    for (_i,_j),_Rij in tqdm(training_set):
        i_l.append([_i])
        j_l.append([_j])
        Rij_l.append([_Rij])
    return \
        np.array(i_l).astype('int32'), \
        np.array(j_l).astype('int32'), \
        np.array(Rij_l)

def UV_np(dataset,initialization=config.initialization,K=config.K):
    """
    R: ratings matrix
    """

    U_values = initialization((K,dataset.N))
    V_values = initialization((K,dataset.M))
    return U_values,V_values

def UV_vectors_np(dataset):
    U_values,V_values = UV_np(dataset)
    U = []
    V = []
    for i in tqdm(range(dataset.N),desc="ui numpy vectors"):
        ui = U_values[:,i].astype('float32')
        U.append(ui)

    for j in tqdm(range(dataset.M),desc="vj numpy vectors"):
        vj = V_values[:,j].astype('float32')
        V.append(vj)

    return U,V

def UV_vectors(dataset):
    U_values,V_values = UV_np(dataset)
    U = []
    V = []
    for i in tqdm(range(U_values.shape[1]),desc="ui shared vectors"):
        ui = theano.shared(U_values[:,i])
        U.append(ui)

    for j in tqdm(range(V_values.shape[1]),desc="vj shared vectors"):
        vj = theano.shared(V_values[:,j])
        V.append(vj)

    return U,V

def UV(dataset):
    U_values,V_values = UV_np(dataset)
    U = theano.shared(U_values)
    V = theano.shared(V_values)
    return U,V

def wrong(x):
    return ((not np.isfinite(x)) or np.isnan(x) or x>10000. or x<-10000.)

def rating_error(Rij_mb,ui_mb,vj_mb,prediction_function):
    #print("rating error Rij={}, i={}, j={}".format(Rij,i,j),flush=True)
    predictions_mb = prediction_function(ui_mb,vj_mb)
    ret = Rij_mb - predictions_mb
    return ret

def split_minibatch(subset,U,V,title):
    ui_mb_l = []
    vj_mb_l = []
    Rij_mb_l = []
    for curr in tqdm(subset,desc=title):
        (i,j),Rij = curr
        ui_mb_l.append(U[i])
        vj_mb_l.append(V[j])
        Rij_mb_l.append(Rij)
        if len(ui_mb_l) >= config.minibatch_size:
            ui_mb = np.vstack(ui_mb_l)
            vj_mb = np.vstack(vj_mb_l)
            Rij_mb = np.vstack(Rij_mb_l)
            ui_mb_l = []
            vj_mb_l = []
            Rij_mb_l = []
            yield Rij_mb,ui_mb,vj_mb

def rmse(subset,U,V,prediction_function):
    errors = []
    for Rij_mb, ui_mb, vj_mb in split_minibatch(subset,U,V,"rmse"):
        eij_mb = rating_error(Rij_mb,ui_mb,vj_mb,prediction_function)
        errors.append(eij_mb**2)
    errors_np = np.vstack(errors)
    return np.sqrt(np.mean(errors_np))

def predictions(subset,U,V,prediction_function):
    l = []
    for Rij_mb, ui_mb, vj_mb in split_minibatch(subset,U,V,"predictions"):
        prediction = prediction_function(ui_mb,vj_mb)
        l.append(prediction)
    ret = np.vstack(l)
    return ret

def test_value(theano_var, _test_value):
    if type(_test_value) is tuple:
        _test_value = np.random.random(_test_value).astype('float32')
    theano_var.tag.test_value = _test_value

class Log(object):
    _file = None

    def __init__(self):
        prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        time_str = time.strftime('%Y%m%d_%H%M%S')
        log_filename = prefix + "_" + time_str + ".log"
        dirname = "logs"
        try:
            os.mkdir(dirname)
        except Exception as e:
            # cannot create dir (maybe it already exists?)
            # loudly ignore the exception
            print("cannot create dir %s: %s"%(dirname,str(e)))

        full_path = os.path.join(dirname,log_filename)
        self._file = open(full_path, 'a')
        self("logging to %s"%full_path)
        self("initial learning rate: %f"%config.lr_begin)
        self("lr annealing T: %f"%config.lr_annealing_T)
        self("update algorithm: %s"%config.update_algorithm)
        self("adam_beta1: %f adam_beta2: %f"%(config.adam_beta1,config.adam_beta2))
        self("K: %d"%config.K)
        self("n_epochs: %d"%config.n_epochs)

    def __call__(self,msg):

        time_str = time.strftime('%Y:%m:%d %H:%M:%S')
        msg = time_str + " " + msg
        print(msg, flush=True)
        self._file.write(msg+"\n")
        self._file.flush()

    def statistics(self, _lr, epoch_nr, splitter, U, V, prediction_function):
        training_rmse = rmse(splitter.training_set,U,V,prediction_function)
        testing_rmse = rmse(splitter.validation_set,U,V,prediction_function)
        _predictions = predictions(splitter.training_set,U,V,prediction_function)
        meanstd = lambda l: "%f %f"%(np.mean(l),np.std(l))
        U_stats = meanstd(U)
        V_stats = meanstd(V)
        p_stats = meanstd(_predictions)
        self("epoch %d"%epoch_nr)
        self("learning rate: %f"%_lr)
        if config.update_algorithm == 'adam':
            U_adam_m_stats = meanstd([ curr.m for curr in U])
            U_adam_v_stats = meanstd([ curr.v for curr in U])
            U_adam_t_stats = meanstd([ curr.t for curr in U])
            self("U adam 'm' mean and std: %s"%U_adam_m_stats)
            self("U adam 'v' mean and std: %s"%U_adam_v_stats)
            self("U adam 't' mean and std: %s"%U_adam_t_stats)
            V_adam_m_stats = meanstd([ curr.m for curr in V])
            V_adam_v_stats = meanstd([ curr.v for curr in V])
            V_adam_t_stats = meanstd([ curr.t for curr in V])
            self("V adam 'm' mean and std: %s"%V_adam_m_stats)
            self("V adam 'v' mean and std: %s"%V_adam_v_stats)
            self("V adam 't' mean and std: %s"%V_adam_t_stats)
        self("training RMSE: %s"%training_rmse)
        self("testing RMSE: %s"%testing_rmse)
        self("U mean and std: %s"%U_stats)
        self("V mean and std: %s"%V_stats)
        self("predictions mean and std: %s"%p_stats)

class epochsloop(object):

    def __init__(self,dataset,U,V,prediction_function):
        self.dataset = dataset
        np.set_printoptions(precision=4, suppress=True)
        self._log = Log()
        self.splitter = config.split_dataset_schema(self.dataset)
        self.validation_set = self.splitter.validation_set
        self.U = U
        self.V = V
        self.prediction_function = prediction_function

    def __iter__(self):
        self._iter = iter(tqdm(list(range(config.n_epochs)),desc="epochs")) # internal "hidden" iterator
        return self

    def __next__(self):
        self.splitter.prepare_new_training_set()
        epoch_nr = next(self._iter) # will raise StopIteration when done

        _lr = update_algorithms.calculate_lr(epoch_nr)

        if type(self.U) is T.sharedvar.TensorSharedVariable:
            _U = self.U.get_value()
            _V = self.V.get_value()
        else:
            _U = self.U
            _V = self.V
        self._log.statistics(
            _lr,
            epoch_nr,
            self.splitter,
            _U,
            _V,
            self.prediction_function
        )
        return self.splitter.training_set,_lr

def mainloop(process_datapoint,dataset,U,V,prediction_function):
    for training_set,_lr in epochsloop(dataset,U,V,prediction_function):
        # WARNING: _lr is not updated in theano expressions
        for curr in tqdm(training_set,desc="training"):
            (i,j),Rij = curr
            process_datapoint(i,j,Rij,_lr)
