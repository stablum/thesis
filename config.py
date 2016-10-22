import theano
import ipdb; ipdb.set_trace
import initializations
import split_dataset_schemas

theano_mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'

#theano_mode = 'DebugMode'
#theano.config.optimizer = 'None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'raise'

theano.mode = theano_mode

n_epochs=1000

K=15
hid_dim=1000
n_hid_layers=2
chan_out_dim=K
stochastic_prediction=False#True
regularization_lambda=0.#0.01
dropout_p=0.#0.5
lr_begin=0.005#0.001#0.005
lr_annealing_T=n_epochs
max_rating=5.

update_algorithm = 'adam_symbolic'
#update_algorithm = 'sgd'

adam_beta1 = 0.9
adam_beta2 = 0.999

#split_dataset_schema = split_dataset_schemas.ChunkyRandomCompleteEpochs
split_dataset_schema = split_dataset_schemas.MemoryRandomCompleteEpochs
validation_set_fraction=0.05

initialization = initializations.normal

g_rij = "sigmoid"
g_in = "elu"

chunk_len =64*1024
minibatch_size = 64

debug=False
if debug:
    movielens_which='small'
    optimizer = "debug"
else:
    movielens_which='1m'
    optimizer = "gpu_omp"

if optimizer == "debug":
    theano.config.exception_verbosity="high"
    theano.config.optimizer='None'
    theano.config.on_unused_input='ignore'
    theano.config.floatX='float32'

elif optimizer == "cpu":
    theano.config.optimizer='fast_run'
    theano.config.floatX='float32'

elif optimizer == "gpu":
    theano.config.optimizer='fast_run'
    theano.config.openmp=False
    theano.config.openmp_elemwise_minsize=8
    #theano.config.device='gpu'
    theano.config.floatX='float32'
    theano.config.assert_no_cpu_op='raise'

elif optimizer == "gpu_omp":
    theano.config.optimizer='fast_run'
    theano.config.openmp=True
    theano.config.openmp_elemwise_minsize=4
    #theano.config.device='gpu'
    theano.config.floatX='float32'
    theano.config.assert_no_cpu_op='raise'
