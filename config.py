import theano
import ipdb; ipdb.set_trace
import initializations

theano_mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'

#theano_mode = 'DebugMode'
#theano.config.optimizer = 'None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'raise'

theano.mode = theano_mode

n_epochs=2000

K=10#3
lr_begin=0.005#0.001#0.005
lr_annealing_T=n_epochs
max_rating=5.

update_algorithm = 'adam_symbolic'
#update_algorithm = 'sgd'

adam_beta1 = 0.9
adam_beta2 = 0.99

initialization = initializations.normal

