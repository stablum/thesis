import lasagne
import config
import theano
from theano import tensor as T
import copy

def make_hid_part(
        l_in,
        hid_dim,
        name,
        g_hid
    ):
    layers = []
    l_curr = l_in
    for i in range(config.n_hid_layers):
        l_curr = lasagne.layers.DenseLayer(
            l_curr,
            hid_dim,
            nonlinearity=g_hid,
            name=name+"_hid_"+str(i)
        )
        layers.append(l_curr)
        if config.dropout_p > 0:
            l_curr = lasagne.layers.DropoutLayer(
                l_curr,
                p=config.dropout_p,
                rescale=True,
                name=name+"_drop_hid_"+str(i)
            )
            layers.append(l_curr)
    return layers

def make_net(
        input_var,
        in_dim,
        hid_dim,
        out_dim,
        name,
        g_hid,
        g_out,
        stochastic_output=False,
        return_only_sample=None
    ):

    if return_only_sample is None:
        return_only_sample=stochastic_output

    l_in = lasagne.layers.InputLayer(
        (config.minibatch_size,in_dim),
        input_var=input_var,
        name=name+"_input"
    )
    layers = [l_in]

    layers += make_hid_part(l_in,hid_dim,name,g_hid)
    l_curr = layers[-1]
    l_out_mu = lasagne.layers.DenseLayer(l_curr,out_dim,nonlinearity=g_out,name=name+"_out")
    layers.append(l_out_mu)

    if stochastic_output is True:
        # adding additional diagonal of covariance matrix output
        # the usual output l_out_mu is considered as being the mu parameter
        l_out_log_sigma = lasagne.layers.DenseLayer(
            l_curr,
            out_dim,
            nonlinearity=lasagne.nonlinearities.linear,
            name=name+"_out"
        )
        layers.append(l_out_log_sigma)
        l_merge_distr = lasagne.layers.ConcatLayer([l_out_mu,l_out_log_sigma],name="concat")
        l_sampling = SamplingLayer(l_merge_distr,dim=config.K,name="sampling")
        layers.append(l_sampling)
        if return_only_sample is True:
            net_output_det = lasagne.layers.get_output(l_out_log_sigma,deterministic=True)
            net_output_lea = lasagne.layers.get_output(l_out_log_sigma,deterministic=False)
        else:
            # output of the network is sample and a (mu,sigma) distribution (tuple)
            outputting_layers = [
                l_sampling,
                l_out_mu,
                l_out_log_sigma,
                l_merge_distr
            ]
            net_output_det = outputting_layers# lasagne.layers.get_output(outputting_layers, deterministic=True)
            net_output_lea = None# lasagne.layers.get_output(outputting_layers, deterministic=False)
            #net_output_det = (sample_det, net_output_mu_det, net_output_log_sigma_det, net_output_distr_det)
            #net_output_lea = (sample_lea, net_output_mu_lea, net_output_log_sigma_lea, net_output_distr_lea)

    net_params = lasagne.layers.get_all_params(layers)

    regularizer_term = lasagne.regularization.regularize_network_params(
        l_out_mu,
        lasagne.regularization.l2
    )
    return net_output_det, net_output_lea, net_params, regularizer_term, l_sampling

class SamplingLayer(lasagne.layers.Layer):

    def __init__(self, *args, **kwargs):
        self.dim = kwargs.pop('dim',None)
        super(SamplingLayer, self).__init__(*args, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        if type(inputs) in (tuple,list):
            assert len(inputs) == 2
            mu = inputs[0]
            log_sigma = inputs[1]
            sample = reparameterization_trick([mu,log_sigma],"samplinglayersplit")
        else:
            assert self.dim is not None
            sample = reparameterization_trick(inputs,"samplinglayermerged",dim=self.dim)

        return sample

    def get_output_shape_for(self, input_shape):
        if self.dim is not None:
            output_shape = (input_shape[0], self.dim)
        else:
            output_shape = input_shape
        return output_shape

def split_distr(in_var,dim):
    mu = in_var[:,0:dim]
    log_sigma = in_var[:,dim:dim*2]
    return mu, log_sigma

def reparameterization_trick(in_var,name,dim=None):
    epsilon = T.shared_randomstreams.RandomStreams().normal(
        ( config.minibatch_size, dim),
        avg=0.0,
        std=1.0
    )
    epsilon.name = 'epsilon_'+name

    if type(in_var) in (list,tuple):
        mu = in_var[0]
        log_sigma = in_var[1]
    else:
        assert dim is not None
        mu, log_sigma = split_distr(in_var,dim)
        mu.name = name+'_splitmu'
        log_sigma.name = name+"_splitlogsigma"
    sigma = T.exp(log_sigma)
    sigma.name = name+"_sigma"
    sample = mu + (epsilon * (sigma**0.5))
    sample.name = name+'_sample'
    return sample
