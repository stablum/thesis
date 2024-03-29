import theano
import socket
import ipdb; ipdb.set_trace
import initializations
import split_dataset_schemas
import movielens

seed=12345
n_epochs=1000

K=250
TK=1 # transformation's K (number of nested transformation steps)
enforce_invertibility=True
flow_type="planar"
g_flow="tanh"
hid_dim=1000
n_hid_layers=2
chan_out_dim=K
stochastic_prediction=False#True
regularization_lambda=100
regularization_type="L2" # in L1/L2
regularization_latent_kl=1.0#1.0#1.0#0#0.5
kl_annealing_epsilon=0.05
kl_annealing_T=100
input_dropout_p=0.5
dropout_p=0.0
batch_normalization=False
lr_begin=2e-6#2e-7#2e-2 #2e-4#2e-6 # 1e-5 # 1e-6 # 0.5 # 5e-3
lr_annealing_T=10
max_rating=5.

free_nats=1.0 * K # 2*K
soft_free_nats=True

ratings_training_set_subsample_size = 5000

regression_error_coef=0.5#1.#100.#1.#2.

#update_algorithm = 'rprop_masked'
update_algorithm = 'adam_masked'
#update_algorithm = 'sgd'

adam_beta1 = 0.9
adam_beta2 = 0.999

#split_dataset_schema = split_dataset_schemas.ChunkyRandomCompleteEpochs
split_dataset_schema = split_dataset_schemas.MemoryRandomCompleteEpochsSparseRows
dataset_type = 'DataSetIndividualRatings'
validation_set_fraction=0.05

initialization = initializations.normal

g_rij = "pseudo_linear"
g_in = "elu"
g_hid = "elu"
g_latent = "pseudo_linear"
g_transform = "sigmoid"

preprocessing_type = "vanilla" # in 0to1/vanilla/zscore
spherical_likelihood = True

chunk_len =64*1024
minibatch_size = 64 #1 #2 # 16 # 64

regression_type = "item" # in user/item/user+item

nanguard=False
verbose=False
norm_clip_threshold=10

if socket.gethostname() in ['playertrackingmobile']:
    # locally
    debug=True
else:
    # super/grid computing
    debug=False

if debug:
    movielens_which='small'
    optimizer = "debug"
else:
    movielens_which='1m'
    optimizer = "debug"
    #optimizer = "gpu_omp"

if optimizer == "debug":
    theano.config.exception_verbosity="high"
    theano.config.optimizer='None'
    theano.config.on_unused_input='ignore'
    theano.config.floatX='float32'
    theano.config.warn_float64='raise'
    theano.mode="DebugMode"

elif optimizer == "cpu":
    theano.config.optimizer='fast_run'
    theano.mode = "FAST_RUN"
    theano.config.floatX='float32'
    theano.config.allow_gc=False

elif optimizer == "gpu":
    theano.config.optimizer='fast_run'
    theano.mode = "FAST_RUN"
    theano.config.openmp=False
    theano.config.openmp_elemwise_minsize=8
    #theano.config.device='gpu'
    theano.config.floatX='float32'
    theano.config.assert_no_cpu_op='raise'
    theano.config.allow_gc=False

elif optimizer == "gpu_omp":
    theano.config.optimizer='fast_run'
    theano.mode = "FAST_RUN"
    theano.config.openmp=True
    theano.config.openmp_elemwise_minsize=4
    #theano.config.device='gpu'
    theano.config.floatX='float32'
    theano.config.assert_no_cpu_op='raise'
    theano.config.allow_gc=False
