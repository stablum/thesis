import lasagne
import config

def make_net(input_var,in_dim,hid_dim,out_dim,name,g_hid,g_out):
    layers = []
    l_in = lasagne.layers.InputLayer((config.minibatch_size,in_dim),input_var=input_var,name=name+"_input")
    layers.append(l_in)
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
    l_out = lasagne.layers.DenseLayer(l_curr,out_dim,nonlinearity=g_out,name=name+"_out")
    layers.append(l_out)
    net_output_det = lasagne.layers.get_output(l_out,deterministic=True)
    net_output_lea = lasagne.layers.get_output(l_out,deterministic=False)
    net_params = lasagne.layers.get_all_params(layers)

    regularizer_term = lasagne.regularization.regularize_network_params(
        l_out,
        lasagne.regularization.l2
    )
    return net_output_det, net_output_lea, net_params, regularizer_term

