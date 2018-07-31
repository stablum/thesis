import lasagne
import numpy as np
from lasagne.layers.dense import DenseLayer
import theano
from theano import tensor as T
import copy
import random
import regularization
from theano.compile.nanguardmode import NanGuardMode

import utils
import update_algorithms
import config

def tp(x):
    if getattr(config, "verbose", False):
        ret = theano.printing.Print('print '+str(x.name))(x)
    ret = x
    return ret

def norm_clip_gradient(grad):
    # http://nmarkou.blogspot.com/2017/07/deep-learning-why-you-should-use.html
    norm = grad.norm(2)
    threshold = getattr(config,"norm_clip_threshold",10)
    if threshold not in (0,None):
        coeff = theano.ifelse.ifelse(T.lt(norm,threshold), 1.0, threshold/norm)
        ret = coeff * grad
    else:
        ret = grad
    return ret

class Abstract(object):

    @utils.cached_property
    def epoch_nr(self):
        ret = T.fscalar()
        ret.name="epoch_nr"
        return ret

    @utils.cached_property
    def lr(self):
        ret = update_algorithms.calculate_lr_symbol(self.epoch_nr)
        ret.name="lr"
        return ret

    @utils.cached_property
    def update(self):
        ret = update_algorithms.get_func()
        return ret

    @property
    def log(self):
        if '_log' not in dir(self):
            self._log = lambda *args: print(*args)#None
        return self._log

    @log.setter
    def log(self,val):
        self._log = val

    @property
    def n_datapoints(self):
        assert '_n_datapoints' in dir(self)
        assert self._n_datapoints is not None

        return self._n_datapoints

    @n_datapoints.setter
    def n_datapoints(self,val):
        self._n_datapoints = val

    def wrap(self,layer):

        if config.batch_normalization is True:
            layer = lasagne.layers.BatchNormLayer(
                layer,
                name="batch_norm_"+layer.name
            )

        if config.dropout_p > 0:
            layer = lasagne.layers.DropoutLayer(
                layer,
                p=config.dropout_p,
                rescale=False,
                name="drop_"+layer.name
            )
        return layer

    @property
    def all_layers(self):
        return lasagne.layers.get_all_layers(self.l_out)

    @property
    def all_layers_outputs(self):
        ret = []
        for l in self.all_layers:
            o = lasagne.layers.get_output(l,deterministic=True)
            o.name="output_of_layer_"+l.name
            ret.append(o)
        return ret

    @utils.cached_property
    def input_dim(self):
        if config.regression_type == "user":
            return self.dataset.M
        elif config.regression_type == "item":
            return self.dataset.N
        elif config.regression_type == "user+item":
            return self.dataset.M + self.dataset.N
        else:
            raise Exception("config.regression_type not valid")

    @property
    def params_for_persistency(self):
        params_values = lasagne.layers.get_all_param_values(self.all_layers)
        return params_values

    @params_for_persistency.setter
    def params_for_persistency(self,params):
        lasagne.layers.set_all_param_values(self.all_layers, params)

    @property
    def params_updates_values(self):
        ret = []
        for k in list(self.params_updates.keys()):
            ret.append(k.get_value())
        return ret

    @params_updates_values.setter
    def params_updates_values(self,vals):
        for new_value,k in zip(vals,list(self.params_updates.keys())):
            k.set_value(new_value)

    @property
    def params_updates(self):
        if '_params_updates' not in dir(self):
            self.log("creating parameter updates...")
            clipped_gradients = list(map(norm_clip_gradient,self.grads_params))
            self._params_updates = self.update (
                clipped_gradients,
                self.params,
                self.all_masks,
                learning_rate=self.lr * config.minibatch_size
            )
        return self._params_updates

    @params_updates.setter
    def params_updates(self,val):
        assert val is not None
        self._params_updates = val

def make_function( *args, **kwargs ):
    if config.nanguard == True:
        kwargs['mode'] = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
    fn = theano.function(*args,**kwargs)
    return fn

def make_hid_part(
        l_in,
        hid_dim,
        name,
        g_hid
    ):
    import config
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
    import config

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

    regularization_function = regularization.get(config.regularization_type)

    regularizer_term = lasagne.regularization.regularize_network_params(
        l_out_mu,
        regularization_function
    )

    return net_output_det, net_output_lea, net_params, regularizer_term, l_sampling

def create_autoregressive_masks_W_V(shape):
    K = shape[0]
    D = shape[1]
    MW = np.zeros((K,D))
    MV = np.zeros((D,K))

    m = [1,D-1]
    while len(m) < K:
        m_k = random.randint(1,D-1)
        if m_k not in m:
            m.append(m_k)
    random.shuffle(m)

    for k in range(K):
        for d in range(D):
            if m[k] >= d:
                MW[k,d] = 1
            if d > m[k]:
                MV[d,k] = 1

    return m,lasagne.utils.floatX(MW),lasagne.utils.floatX(MV)

class MaskedDenseLayer(lasagne.layers.dense.DenseLayer):
    def __init__(self,*args,**kwargs):
        mask = kwargs.pop('mask')
        super(MaskedDenseLayer, self).__init__(*args, **kwargs)
        self.mask = self.add_param(
            mask,
            mask.shape,
            trainable=False
        )
    def get_output_for(self, input, **kwargs):
        m = T.mul(self.W,self.mask.T)
        m.name = "m_"+self.name
        activation = T.dot(input, m)
        activation.name = "activation_"+self.name
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
            activation.name = "activation_dimshuffle_"+self.name
        ret = self.nonlinearity(activation)
        ret.name = "ret_"+self.name
        return ret

class MaskLayer(lasagne.layers.Layer):
    def __init__(self,*args,**kwargs):
        self.dim = kwargs.pop('dim')
        self.mask = kwargs.pop('mask') # should be 0/1 numpy array
        super(MaskLayer,self).__init__(*args, **kwargs)

    def get_output_for(self,input, **kwargs):
        ret,_ = theano.scan(
            fn=lambda z: T.mul(z, self.mask),
            sequences=[input]
        )
        ret.name = "ret_"+self.name
        return ret

    def get_output_shape_for(self, input_shape):
        if self.dim is not None:
            output_shape = (input_shape[0], self.dim)
        else:
            output_shape = input_shape
        return output_shape

class ProducedDenseLayer(lasagne.layers.MergeLayer):
    def __init__(self, *args,**kwargs):
        self.dim = kwargs.pop('dim')
        self.nonlinearity = kwargs.pop('nonlinearity')

        super(ProducedDenseLayer,self).__init__(*args,**kwargs)

    def get_output_for(self, inputs, **kwargs):
        input,ws,bs = inputs
        ws_reshaped,_ = theano.scan(
            fn = lambda curr : curr.reshape((self.dim,self.dim),ndim=2),
            sequences = [ws]
        )
        dots,_ = theano.scan(
            fn = lambda z,w : T.dot(z,w),
            sequences = [input, ws_reshaped]
        )
        activations,_ = theano.scan(
            fn = lambda d,b : d+b,
            sequences = [dots,bs]
        )
        ret = self.nonlinearity(activations)
        return ret

    def get_output_shape_for(self, input_shape):
        if self.dim is not None:
            output_shape = (input_shape[0][0], self.dim)
        else:
            output_shape = input_shape
        return output_shape

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
            sample.name="sample(1)_"+self.name
        else:
            assert self.dim is not None
            sample = reparameterization_trick(inputs,"samplinglayermerged_"+self.name,dim=self.dim)
            sample.name="sample(2)_"+self.name

        return sample

    def get_output_shape_for(self, input_shape):
        if self.dim is not None:
            output_shape = (input_shape[0], self.dim)
        else:
            output_shape = input_shape
        return output_shape

class ILTTEnforceInvertibilityLayer(lasagne.layers.base.MergeLayer):
    def __init__(self,*args,**kwargs):
        self.dim = kwargs.pop('dim',None)
        super(ILTTEnforceInvertibilityLayer, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[0][:1] + (self.dim,)

    def get_output_for(self, inputs, **kwargs):
        w,u = inputs
        wu,_updates = theano.scan(
            fn=T.dot,
            sequences=[w,u]
        )
        wu=wu.dimshuffle(0,'x')
        wu.name="wu"
        m = lambda x: -1 + lasagne.nonlinearities.softplus(x)
        ww,_updates = theano.scan(
            fn=T.dot,
            sequences=[w,w]
        )
        ww = ww.dimshuffle(0,'x')
        ww.name="ww"
        epsilon = 1e-7
        ww = ww + epsilon
        ww = tp(ww)
        w_norm,_updates=theano.scan(
            fn= lambda x,y: x/y,
            sequences = [w,ww]
        )
        w_norm.name = "w_norm"
        mwu = m(wu)
        mwu.name= "mwu"
        sub = mwu - wu
        sub.name="sub"
        u_hat = u + sub * w_norm
        u_hat.name="u_hat"
        return u_hat

class ILTTLayer(lasagne.layers.base.MergeLayer):
    def __init__(self,*args,**kwargs):
        self.dim = kwargs.pop('dim',None)
        self.nonlinearity = kwargs.pop('nonlinearity',None)

        super(ILTTLayer, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[0][:1] + (self.dim,)

    def get_output_for(self, inputs, **kwargs):
        input,w,b,u_hat = inputs
        activation = T.dot(input, w.T)
        activation.name="activation_"+self.name
        if b is not None:
            activation = activation +  b
            activation.name = "activation+b_"+self.name
        h = self.nonlinearity(activation)
        h.name = "h_"+self.name
        dot = T.dot(h,u_hat)
        dot.name = "dot_"+self.name
        ret = input + dot
        ret.name = "ret_"+self.name
        ret = tp(ret)
        return ret

def split_distr(in_var,dim):
    mu = in_var[:,0:dim]
    log_sigma = in_var[:,dim:dim*2]
    return mu, log_sigma

def reparameterization_trick(in_var,name,dim=None):
    import config
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
    sample = mu + (epsilon * sigma)
    sample.name = name+'_sample'
    return sample

def scalar(sometensor):
    return sometensor.reshape((),ndim=0)
