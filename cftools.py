import numpy as np
import argparse
import os
from tqdm import tqdm
tqdm.monitor_interval = 0

import sys
import time
import random
import theano
import theano.tensor as T
import scipy.sparse
import socket
import glob
# local imports

import config
import update_algorithms
import persistency

backup_files = [
    sys.argv[0],
    "engage.sh",
    "cftools.py",
    "update_algorithms.py",
    "movielens.py",
    "augmented_types.py",
    "activation_functions.py",
    "config.py",
    "model_build.py",
    "kl.py",
    "job.sh",
    "utils.py",
    "regularization.py",
    "lasagne_sparse.py",
    "initializations.py",
    "split_dataset_schemas.py",
    "numutils.py",
    "persistency.py",
]

tqdm_mininterval=5

def create_training_set_matrix(training_set):
    print("creating training set matrix..")
    return np.array([
        [_i,_j,_Rij]
        for (_i,_j),_Rij
        in tqdm(training_set,mininterval=tqdm_mininterval)
    ])

def create_training_set_apart(training_set):
    i_l = []
    j_l = []
    Rij_l = []
    print("creating training set vectors..")
    for (_i,_j),_Rij in tqdm(training_set,mininterval=tqdm_mininterval):
        i_l.append([_i])
        j_l.append([_j])
        Rij_l.append([_Rij])
    return \
        np.array(i_l).astype('int32'), \
        np.array(j_l).astype('int32'), \
        np.array(Rij_l)

def UV_np(dataset,initialization=config.initialization,latent_len=config.K):
    """
    R: ratings matrix
    """

    U_values = initialization((latent_len,dataset.N))
    V_values = initialization((latent_len,dataset.M))
    return U_values,V_values

def UV_vectors_np(dataset,expand_dims=False,latent_len=config.K):
    U_values,V_values = UV_np(dataset,latent_len=latent_len)
    U = []
    V = []
    for i in tqdm(range(dataset.N),desc="ui numpy vectors",mininterval=tqdm_mininterval):
        ui = U_values[:,i].astype('float32')
        if expand_dims is True:
            ui = np.expand_dims(ui,0)
        U.append(ui)

    for j in tqdm(range(dataset.M),desc="vj numpy vectors",mininterval=tqdm_mininterval):
        vj = V_values[:,j].astype('float32')
        if expand_dims is True:
            vj = np.expand_dims(vj,0)
        V.append(vj)

    return U,V

def UV_vectors(dataset):
    U_values,V_values = UV_np(dataset)
    U = []
    V = []
    for i in tqdm(range(U_values.shape[1]),desc="ui shared vectors",mininterval=tqdm_mininterval):
        ui = theano.shared(U_values[:,i])
        U.append(ui)

    for j in tqdm(range(V_values.shape[1]),desc="vj shared vectors",mininterval=tqdm_mininterval):
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

def split_minibatch_rrows(subset,title):
    Ri_mb_l = []
    for curr in tqdm(subset,desc=title,mininterval=tqdm_mininterval):
        i,Ri = curr
        Ri_mb_l.append(Ri)
        if len(Ri_mb_l) >= config.minibatch_size:
            Ri_mb = scipy.sparse.vstack(Ri_mb_l)
            Ri_mb_l = []
            yield Ri_mb

def split_minibatch_UV(subset,U,V,title):
    ui_mb_l = []
    vj_mb_l = []
    Rij_mb_l = []
    for curr in tqdm(subset,desc=title,mininterval=tqdm_mininterval):
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

def rmse_rrows(subset_in, subset_out, prediction_function):
    errors = []
    sum_mask = 0
    Ri_mb_ins = split_minibatch_rrows(subset_in,"rmse (in)")
    Ri_mb_outs = split_minibatch_rrows(subset_out,"rmse (out)")
    for Ri_mb_in, Ri_mb_out in zip(Ri_mb_ins,Ri_mb_outs):
        mask = (Ri_mb_out > 0.0000000000000000001).todense().astype('float32')
        predictions, = prediction_function(Ri_mb_in)
        Ri_mb_out_masked = np.multiply(Ri_mb_out.todense(),mask)
        predictions_masked = np.multiply(predictions,mask)
        ei_mb = Ri_mb_out_masked - predictions_masked
        ei_sq = np.power(ei_mb,2)
        error = np.sum(ei_sq)
        sum_mask += np.sum(mask)
        errors.append(error)

    # the average is calculated on the individual ratings, not on user/item rows
    errors_sum = np.sum(errors)
    errors_avg = errors_sum/sum_mask
    ret = np.sqrt(errors_avg)
    return ret

def predictions_rrows(subset,prediction_function):
    l = []
    for Ri_mb in split_minibatch_rrows(subset,"predictions"):
        prediction = prediction_function(Ri_mb)
        l.append(prediction)
    ret = np.vstack(l)
    return ret

def rmse(subset,U,V,prediction_function):
    errors = []
    for Rij_mb, ui_mb, vj_mb in split_minibatch_UV(subset,U,V,"rmse"):
        eij_mb = rating_error(Rij_mb,ui_mb,vj_mb,prediction_function)
        errors.append(eij_mb**2)
    errors_np = np.vstack(errors)
    return np.sqrt(np.mean(errors_np))

def predictions(subset,U,V,prediction_function):
    l = []
    for Rij_mb, ui_mb, vj_mb in split_minibatch_UV(subset,U,V,"predictions"):
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

    def __init__(self,dirname="logs",resume=False,log_params={}):
        prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        time_str = time.strftime('%Y%m%d_%H%M%S')
        if resume is True:
            pattern = os.path.join(dirname,prefix+"_*.log")
            logfiles = glob.glob(pattern)
            if len(logfiles) != 1:
                raise Exception("there must be only one logfile matching the pattern {}, but there are".format(
                    pattern,
                    len(logfiles)
                ))
            full_path = logfiles[0] # take the only logfile that matches the pattern
            print("found logfile to resume: {}".format(full_path))
        else: # create logs directory (if does not already exist) and new log file
            log_filename = prefix + "_" + time_str + ".log"
            try:
                os.mkdir(dirname)
            except Exception as e:
                # cannot create dir (maybe it already exists?)
                # loudly ignore the exception
                print("cannot create dir %s: %s"%(dirname,str(e)))

            full_path = os.path.join(dirname,log_filename)

        self._file = open(full_path, 'a')
        self("logging to %s"%full_path)
        self("initial learning rate: %.10f"%config.lr_begin)
        self("lr annealing T: %f"%config.lr_annealing_T)
        self("update algorithm: %s"%config.update_algorithm)
        self("adam_beta1: %f adam_beta2: %f"%(config.adam_beta1,config.adam_beta2))
        self("K: %d"%config.K)
        self("TK: %d"%config.TK)
        self("hid_dim: %d"%config.hid_dim)
        self("n_hid_layers: %d"%config.n_hid_layers)
        self("chan_out_dim: %d"%config.chan_out_dim)
        self("stochastic_prediction: {}".format(config.stochastic_prediction))
        self("movielens_which: %s"%config.movielens_which)
        self("optimizer: %s"%config.optimizer)
        self("n_epochs: %d"%config.n_epochs)
        self("minibatch_size: %d"%config.minibatch_size)
        self("validation_set_fraction: {}".format(config.validation_set_fraction))
        self("g_in:",config.g_in)
        self("g_latent:",config.g_latent)
        self("g_rij:",config.g_rij)
        self("g_hid:",config.g_hid)
        self("g_transform:",config.g_transform)
        self("g_flow",config.g_flow)
        self("regularization_lambda:",config.regularization_lambda)
        self("dropout_p:",config.dropout_p)
        self("input_dropout_p:",config.input_dropout_p)
        self("batch_normalization:",config.batch_normalization)
        self("regression_error_coef:",config.regression_error_coef)
        self("regression_type:",config.regression_type)
        self("regularization_latent_kl:",config.regularization_latent_kl)
        self("node_hostname:",socket.gethostname())
        self("preprocessing_type:",config.preprocessing_type)
        self("regularization_type:",config.regularization_type)
        self("free_nats:",config.free_nats)
        self("flow_type:",config.flow_type)
        self("enforce_invertibility",config.enforce_invertibility)
        for k,v in log_params.items():
            self(k+":",v)

    def __call__(self,*args):
        msg = " ".join(map(str,args))
        time_str = time.strftime('%Y:%m:%d %H:%M:%S')
        msg = time_str + " " + msg
        print(msg, flush=True)
        self._file.write(msg+"\n")
        self._file.flush()

    def statistics(self, epoch_nr, splitter, U, V, prediction_function):
        if None not in (U,V):
            training_rmse = rmse(splitter.training_set,U,V,prediction_function)
            testing_rmse = rmse(splitter.validation_set,U,V,prediction_function)
            _predictions = predictions(splitter.training_set,U,V,prediction_function)
        else:
            training_rmse = rmse_rrows(
                splitter.training_set,
                splitter.training_set,
                prediction_function
            )
            testing_rmse = rmse_rrows(
                splitter.training_set,
                splitter.validation_set,
                prediction_function
            )
            _predictions = predictions_rrows(splitter.training_set,prediction_function)
        def meanstd(l,axis=0):
            m = np.mean(l,axis=axis)
            s = np.std(l,axis=axis)
            if len(m.shape) == 0:
                indexes = None
            elif len(m.shape) == 2:
                indexes = range(m.shape[1-axis])
            else:
                indexes = range(len(m))
            if axis==0:
                a = np.vstack([indexes,m,s])
                return str(a)
            else:
                return "{} {}".format(m,s)
        p_stats = meanstd(_predictions,axis=None)
        self("epoch %d"%epoch_nr)
        _lr = update_algorithms.calculate_lr(epoch_nr)
        self("learning rate: %12.12f"%_lr)
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
        if None not in (U,V):
            U_stats = meanstd(U)
            V_stats = meanstd(V)
            self("U mean and std: %s"%U_stats)
            self("V mean and std: %s"%V_stats)
        self("predictions mean and std: %s"%p_stats)

class Epochsloop(object):

    def parse_cmd_args(self):
        # called by __init__

        print("argv:",sys.argv)
        print("cwd:",os.getcwd())

        self.cmd_args_parser = argparse.ArgumentParser(description='argument parser')
        self.cmd_args_parser.add_argument('--resume-state',action='store_true')

        self.cmd_args = self.cmd_args_parser.parse_args(sys.argv[1:])
        if self.cmd_args.resume_state:
            # then should be already inside the harvest directory
            last_dir = os.path.split(os.getcwd())[-1]
            if 'harvest' not in last_dir:
                raise Exception("program should be run from within a harvest dir if --resume-state is enabled (cwd={})".format(os.getcwd()))
            __lr,state_epoch = persistency.load("state",self.model)
            # actually the read learning rate __lr will be ignored, as the learning
            # rate is calculated from the epoch.

            # the saved state belongs to that epoch.
            # the next computations belong to the subsequent epoch, hence the +1
            self.epoch_offset = state_epoch + 1
            self._log = Log(dirname='.',resume=True,log_params=self.log_params)
        else:
            self.make_and_cd_experiment_dir()
            self._log = Log(dirname='.',resume=False,log_params=self.log_params)

    def __init__(self,dataset,U,V,prediction_function,log_params,model):
        assert model is not None
        self.model = model
        self.dataset = dataset
        self.splitter = config.split_dataset_schema(self.dataset)

        # order of instructions is important
        # the following has to happen *before* loading the persistency state
        model.n_datapoints = self.n_datapoints

        self.epoch_offset = 1 # eventually overridden by self.parse_cmd_args()
        self.log_params = log_params
        np.set_printoptions(precision=5, suppress=True)

        # includes loading the persistency state
        self.parse_cmd_args()

        self.U = U
        self.V = V
        self.prediction_function = prediction_function
        self.epoch_nr = None

    def log(self, *args, **kwargs):
        return self._log(*args,**kwargs)

    @property
    def validation_set(self):
        return self.splitter.validation_set

    def make_and_cd_experiment_dir(self):
        scriptname_component= os.path.splitext(os.path.basename(sys.argv[0]))[0]
        time_str = time.strftime('%Y%m%d_%H%M%S')
        dirname = "harvest_"+ scriptname_component + "_" + time_str
        try:
            os.mkdir(dirname)
        except Exception as e:
            # cannot create dir (maybe it already exists?)
            # loudly ignore the exception
            print("cannot create dir %s: %s"%(dirname,str(e)))

        # backup copy of source files
        for curr in backup_files:
            os.system("cp %s %s -vf"%(curr,dirname+"/"))

        os.chdir(dirname)

    @property
    def n_datapoints(self):
        # facade to hide the splitter
        return self.splitter.n_datapoints

    def __iter__(self):
        # internal "hidden" iterator
        _range = range(self.epoch_offset,config.n_epochs)
        self._iter = iter(tqdm(list(_range),desc="epochs"))
        return self

    def __next__(self):
        self.splitter.prepare_new_training_set()
        self.epoch_nr = next(self._iter) # will raise StopIteration when done

        if type(self.U) is T.sharedvar.TensorSharedVariable:
            _U = self.U.get_value()
            _V = self.V.get_value()
        else:
            _U = self.U
            _V = self.V
        self._log.statistics(
            self.epoch_nr,
            self.splitter,
            _U,
            _V,
            self.prediction_function
        )
        return self.splitter.training_set,self.epoch_nr

class Looper(object):
    @property
    def n_datapoints(self):
        # facade to hide the Epochsloop
        return self._epochsloop.n_datapoints

def LooperRij(Looper):
    def __init__(
            self,
            process_datapoint,
            dataset,
            U,
            V,
            prediction_function,
            log_params={},
            model=None
    ):
        self._process_datapoint = process_datapoint
        self._epochsloop = Epochsloop(dataset,U,V,prediction_function,log_params,model)

    def start(self):
        for training_set, in self._epochsloop:
            # WARNING: _lr is not updated in theano expressions
            for curr in tqdm(training_set,desc="training",mininterval=tqdm_mininterval):
                (i,j),Rij = curr
                self._process_datapoint(i,j,Rij,_lr)

class LooperRrows(Looper):
    def __init__(
            self,
            process_rrow,
            dataset,
            prediction_function,
            epoch_hook=lambda *args,**kwargs: None,
            log_params={},
            model=None
    ):
        U = None
        V = None
        self._epochsloop = Epochsloop(dataset,U,V,prediction_function,log_params,model)
        self._log = lambda *args, **kwargs: self._epochsloop.log(*args,**kwargs)
        self._process_rrow = process_rrow
        self._epoch_hook = epoch_hook

    def start(self):
        for training_set,_epoch_nr in self._epochsloop:
            for curr in tqdm(training_set,desc="training",mininterval=tqdm_mininterval):
                i,Ri = curr

                if Ri.getnnz() == 0:
                    # nope
                    continue

                self._process_rrow(i,Ri,_epoch_nr)

            self._epoch_hook(
                log=self._log,
                epochsloop=self._epochsloop,
                epoch_nr=_epoch_nr
            )

def preprocess(data,dataset):

    if config.preprocessing_type == "0to1":
        ret = (data - 1.) / (config.max_rating - 1.)
    elif config.preprocessing_type == "zscore":
        mean, std = dataset.mean_and_std
        ret = (data - mean) / std
    else: # vanilla
        ret = data

    return ret

def unpreprocess(data,dataset):

    if config.preprocessing_type == "0to1":
        ret = (data * (config.max_rating - 1. )) + 1.
    elif config.preprocessing_type == "zscore":
        mean, std = dataset.mean_and_std
        ret = (data * std) + mean
    else: # vanilla
        ret = data

    return ret
