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

K=60#10#3
hid_dim=500
lr_begin=0.005#0.001#0.005
lr_annealing_T=n_epochs
max_rating=5.

update_algorithm = 'adam_symbolic'
#update_algorithm = 'sgd'

adam_beta1 = 0.9
adam_beta2 = 0.999

#split_dataset_schema = split_dataset_schemas.ChunkyRandomCompleteEpochs
split_dataset_schema = split_dataset_schemas.MemoryRandomCompleteEpochs

initialization = initializations.normal

chunk_len =64*1024
minibatch_size = 64
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
