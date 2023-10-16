import functools
import numpy as npy

import matplotlib.pyplot as plt

import jax
import jax.numpy as np

from jax import jit
from jax import Array
from jax.example_libraries import optimizers as jax_opt


# RNN helpers (test)

def esn_params(input_scaling, esn_size, spectral_radius, a_dt, bias_scaling=0.8, seed=1235):
    input_size = 1
    output_size = 1
    # bias_scaling = 0.8
    prng = npy.random.default_rng(seed)

    def esn_ini(shape, spectral_radius):
        w = prng.normal(size=shape)
        current_spectral_radius = max(abs(npy.linalg.eig(w)[0]))
        w *= spectral_radius / current_spectral_radius
        return w
    params = dict(win=prng.normal(size=(esn_size, input_size))*input_scaling, w=esn_ini((esn_size, esn_size), spectral_radius), d=esn_ini((esn_size, esn_size), spectral_radius),
                  bias=prng.normal(size=(esn_size,))*bias_scaling, wout=prng.normal(size=(output_size, esn_size)), bias_out=prng.normal(size=(output_size,))*bias_scaling,
                  a_dt=a_dt*np.ones(esn_size), x_ini=0.1*prng.normal(size=(2, esn_size)))

    return jax.tree_map(lambda x: np.array(x), params)

def esn3d_params(input_scaling, esn_size, data_size, spectral_radius, a_dt, mlp_size_hidden, mlp_in_out, bias_scaling = 0.8, seed = 1235):
    # state based computation

    input_size = data_size
    output_size = data_size
    prng = npy.random.default_rng(seed) 
    def esn_ini(shape, spectral_radius, prng):
        w = prng.normal(size=shape)
        current_spectral_radius = max(abs(npy.linalg.eig(w)[0]))
        w *= spectral_radius / current_spectral_radius
        return w
    params = dict(
         bias=prng.normal(size=(esn_size,))*bias_scaling, bias_out=prng.normal(size=(output_size,))*bias_scaling,
         a_dt = a_dt*np.ones(esn_size))

    # params['wout0'] = prng.normal(size=(output_size, mlp_size_hidden[0]))
    # params['wout1'] = prng.normal(size=(output_size, mlp_size_hidden[0]))
    params['wout0'] = prng.normal(size=(output_size, esn_size))
    params['wout1'] = prng.normal(size=(output_size, esn_size))
    params['w0'] = esn_ini((esn_size,esn_size), spectral_radius, prng)
    params['w1'] = esn_ini((esn_size,esn_size), spectral_radius, prng)
    params['win0'] = prng.normal(size=(esn_size,input_size))*input_scaling
    params['win1'] = prng.normal(size=(esn_size,input_size))*input_scaling


    params['x_ini0']=  npy.array([-.5,0.01,0.01])#0.1*prng.normal(size=(esn_size))# B_bias[0]#npy.array(C_bias) @ (params['x_ini'][0,0] - npy.array(B_bias[0])) + npy.array(B_bias[0])
    params['x_ini1']= npy.array([-.5,0.01,0.01])#0.1*prng.normal(size=(esn_size))#B_bias[1]#C_bias @ (params['x_ini'][1,0] - B_bias[1]) + B_bias[1]
    params['bias_out0'] = prng.normal(size=(output_size,))*bias_scaling
    params['bias_out1'] = prng.normal(size=(output_size,))*bias_scaling


    # cool initialization for being in the right subspace
    params['bias0'] = prng.normal(size=(esn_size,))*bias_scaling
    params['bias1'] = prng.normal(size=(esn_size,))*bias_scaling
    params['win1'][0,:] = 0
    params['win0'][0,:] = 0

    # MLP based computation
    def initialize_mlp(sizes, prng):
        """ Initialize the weights of all layers of a linear layer network """
        # Initialize a single layer with Gaussian weights -  helper function
        def initialize_layer(m, n, prng, scale=1e-2):
            return scale * prng.normal(size=(n, m)), scale * prng.normal(size=(n,))
        return [initialize_layer(m, n, prng) for m, n in zip(sizes[:-1], sizes[1:])]
    
    for pos in ["state", "in", "out"]:
        if pos =="state":
            mlp_size = [esn_size] + mlp_size_hidden + [esn_size]
        elif pos =="in":
            if mlp_in_out[0] == []:
                break
            else:
                mlp_size = [input_size] + mlp_in_out[0] + [esn_size]
        elif pos == "out":
            if mlp_in_out[1] == []:
                break
            else:
                mlp_size = [esn_size] + mlp_in_out[1] + [output_size]
        for i in range(2):
            l_para = initialize_mlp(mlp_size, prng)
            for idx_layer, para in enumerate(l_para):
                params[f'mlp_{pos}{i}_l{idx_layer}_w'] = para[0]
                params[f'mlp_{pos}{i}_l{idx_layer}_b'] = para[1]   

    return jax.tree_map(lambda x: np.array(x), params)

def forward_esn(params, C_bottleneck, ut, idx, x_init=None, encoding=True, biased=False):
    # u_clock can be used both for clock and for input (for initialization)
    if x_init is None:
        x_init = params["x_ini"][idx]

    def apply_fun_scan(params, encoding, xy, ut):

        x, y = xy

        if encoding:
            if biased == True:
                x = np.dot(C_bottleneck, (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
                    np.dot(params['win'], ut) + np.dot(params['w'], x) + params['bias']))
            else:
                x = (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
                    np.dot(params['win'], ut) + np.dot(params['w'], x) + params['bias'])
        else:
            x = np.dot(C_bottleneck, (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
                np.dot(params['win'], np.dot(params['wout'], x) + params['bias_out']) +
                np.dot(params['w'], x) + params['bias']
            ))

        y = np.dot(
            params['wout'], x) + params['bias_out']

        xy = (x, y)

        return xy, np.concatenate((y, x))

    f = functools.partial(apply_fun_scan, params)
    f = functools.partial(f, encoding)
    xy = (x_init, np.zeros(params['bias_out'].shape[0]))
    _, YX = jax.lax.scan(f, xy, ut)
    return YX

def forward_esn_interp(params, C_manifold, ut, x_init, t_interp):
    if x_init is None:
        x_init = params["x_ini"][0]

    def apply_fun_scan(params, xyc, ut):

        x, y, count = xyc
        ratio = t_interp[count]

        C_fb = (1-ratio)*C_manifold[0] + ratio*C_manifold[1]

        x = np.dot(C_fb, (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
            np.dot(params['win'], np.dot(params['wout'], x) + params['bias_out']) +
            np.dot(params['w'], x) + params['bias']
        ))

        y = np.dot(
            params['wout'], x) + params['bias_out']

        xyc = (x, y, count+1)

        return xyc, np.concatenate((y, x))

    f = functools.partial(apply_fun_scan, params)
    xyc = (x_init, np.zeros(params['bias_out'].shape[0]), 0)
    _, YX = jax.lax.scan(f, xyc, ut)
    return YX

# the one with mixing
def forward_esn3d(params, C_bias, B_bias, ut, idx, x_init = None, encoding = True, biased = False, sep = True, y_init = None, noise = 0.):
    """
    Compute the forward pass for each example individually.
    :param params: parameters of the ESN
    :param C_bottleneck: conceptor matrix
    :param ut: input (T, K)
    :param idx: index of the ESN
    :param x_init: initial state of the reservoir
    :param encoding: whether to encode or decode
    :param biased: whether to use conceptor to bias the state
    :param sep: whether to mix the weights or not
    :param y_init: initial output of the reservoir
    :param inp_mix: whether to mix the input with the state in the MLP or not
    """


    # u_clock can be used both for clock and for input (for initialization)
    if x_init is None:
        interp_para = idx
        x_init = (1-interp_para)*params['x_ini0'] + interp_para*params['x_ini1']
    if y_init is None:
        y_init = ut[1]
    def apply_fun_scan(params, encoding, C_bias, B_bias, xy, ut):

        x, y = xy 
        if sep:
            interp_para = idx
        else:
            interp_para = (x[0]+0.5)
        w_eff = (1-interp_para)*params['w0'] + interp_para*params['w1']
        win_eff = (1-interp_para)*params['win0'] + interp_para*params['win1']
        wout_eff = (1-interp_para)*params['wout0'] + interp_para*params['wout1'] 
        bias_eff = (1-interp_para)*params['bias0'] + interp_para*params['bias1'] 
        bias_out_eff = (1-interp_para)*params['bias_out0'] + interp_para*params['bias_out1'] 
        # w_exp_eff = (1-interp_para)*params['expension0'] + interp_para*params['expension1']


        
        def encode(params, ut, x, y, interp_para):
            x_in = mlp_eff_in_out(params, interp_para, ut, "in", win_eff, bias_eff)[0]          
            x_tanh = np.dot(w_eff, x) + x_in
            x_tanh, x_exp = mlp_eff(params, interp_para, x_tanh)
            x = (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
                    x_tanh)
            return x, x_exp

        def decode(params, ut, x, y, interp_para):
            x_in = mlp_eff_in_out(params, interp_para, y, "in", win_eff, bias_eff)[0]
            x_tanh = np.dot(w_eff, x) + x_in
            x_tanh, x_exp = mlp_eff(params, interp_para, x_tanh)
            x = (1-params["a_dt"])*x + params["a_dt"]*np.tanh(
                x_tanh)
            return x, x_exp


        

        x, x_exp = jax.lax.cond(encoding, encode, decode,
            params, ut, x, y, interp_para
        )
        # x = np.tanh(x) # still required?

        # add a little bit of noise to the state and the input
        # noise along x-axis is directly cancelled
        _, ut = jax.lax.cond(
            encoding,
            lambda x: (x[0] + noise*npy.random.randn(*x[0].shape), x[1] + noise*npy.random.randn(*x[1].shape)),
            lambda x: x,
            (x, ut)
        )
        # project
        x = np.dot(C_bias, x-B_bias) + B_bias #+ np.clip(x-B_bias, -0.01, 0.01)
        # x = B_bias
        # x = np.dot(C_bias, x)
        # x = jax.lax.cond(
        #     False,
        #     lambda x: x,
        #     lambda x: np.dot(C_bias, x-B_bias) + B_bias,
        #     x
        # )
        
        y = mlp_eff_in_out(params, interp_para, x, "out", wout_eff, bias_out_eff)[0]

        xy = (x,y)

        return xy, (y,x,x_exp)

    f = functools.partial(apply_fun_scan, params)
    f = functools.partial(f, encoding) 
    f = functools.partial(f, C_bias)
    f = functools.partial(f, B_bias)
    xy = (x_init, y_init)
    _, YX = jax.lax.scan(f, xy, ut)
    return YX

def mlp(params, in_array, pos):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    i = 0
    while True:
        if f'mlp_{pos}_l{i+1}_w' not in params:
            break
        w = params[f'mlp_{pos}_l{i}_w']
        b = params[f'mlp_{pos}_l{i}_b']
        activations = jax.nn.relu(w @ activations + b)
        i += 1

    w = params[f'mlp_{pos}_l{i}_w']
    b = params[f'mlp_{pos}_l{i}_b']
    activations = w @ activations + b
    return activations

def mlp_eff_in_out(params, interp, in_array, pos, w_in_out, b_in_out):
    # check if a key in params exist
    # activations, x_exp = jax.lax.cond(f'mlp_{pos}0_l0_w' in params,
    #     lambda params, interp, in_array, pos: mlp_eff(params, interp, in_array, pos),
    #     lambda params, interp, in_array, pos, w_in_out, b_in_out: (np.dot(w_in_out, in_array) + b_in_out, None),
    #     (params, interp, in_array, pos, w_in_out, b_in_out)
    # )
    # without lax.cond
    if f'mlp_{pos}0_l0_w' in params:
        activations, x_exp = mlp_eff(params, interp, in_array, pos)
        # skip layer
        activations = activations + np.dot(w_in_out, in_array) + b_in_out
    else:
        activations = np.dot(w_in_out, in_array) + b_in_out
        x_exp = None

    return activations, x_exp
    
def mlp_eff(params, interp, in_array, pos="state"):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    i = 0
    while True:

        w0 = params[f'mlp_{pos}0_l{i}_w']
        w1 = params[f'mlp_{pos}1_l{i}_w']
        b0 = params[f'mlp_{pos}0_l{i}_b']
        b1 = params[f'mlp_{pos}1_l{i}_b']
        w_eff = (1-interp) * w0 + interp * w1
        b_eff = (1-interp) * b0 + interp * b1
        activations = jax.nn.relu(w_eff @ activations + b_eff) 
        if i == 0:
            x_exp = activations
        i += 1
        if f'mlp_{pos}0_l{i+1}_w' not in params:
            break

    w0 = params[f'mlp_{pos}0_l{i}_w']
    w1 = params[f'mlp_{pos}1_l{i}_w']
    b0 = params[f'mlp_{pos}0_l{i}_b']
    b1 = params[f'mlp_{pos}1_l{i}_b']
    w_eff = (1-interp) * w0 + interp * w1
    b_eff = (1-interp) * b0 + interp * b1

    activations = w_eff @ activations + b_eff
    return activations, x_exp

# def mlp_eff(params, interp, in_array):
#     """ Compute the forward pass for each example individually """
#     activations = in_array

#     # Loop over the ReLU hidden layers
#     i = 0
#     while True:

#         w0 = params[f'mlp_0_l{i}_w']
#         w1 = params[f'mlp_1_l{i}_w']
#         b0 = params[f'mlp_0_l{i}_b']
#         b1 = params[f'mlp_1_l{i}_b']
#         w_eff = (1-interp) * w0 + interp * w1
#         b_eff = (1-interp) * b0 + interp * b1
#         activations = jax.nn.relu(w_eff @ activations + b_eff) 
#         if i == 0:
#             x_exp = activations
#         i += 1
#         if f'mlp_0_l{i+1}_w' not in params:
#             break

#     w0 = params[f'mlp_0_l{i}_w']
#     w1 = params[f'mlp_1_l{i}_w']
#     b0 = params[f'mlp_0_l{i}_b']
#     b1 = params[f'mlp_1_l{i}_b']
#     w_eff = (1-interp) * w0 + interp * w1
#     b_eff = (1-interp) * b0 + interp * b1

#     activations = w_eff @ activations + b_eff
#     return activations, x_exp

# def mlp_eff(params, interp, in_array):
#     """ Compute the forward pass for each example individually with interpolation of parameters"""
#     activations = in_array

#     # Loop over the tanh hidden layers
#     for i, ((w0, b0), (w1, b1)) in enumerate(zip(params["mlp_state0"], params["mlp_state1"])):
#         w_eff = (1-interp) * w0 + interp * w1
#         b_eff = (1-interp) * b0 + interp * b1
#         activations = np.tanh(w_eff @ activations + b_eff)
#         if i == 0:
#             x_exp = activations
    
#     return activations, x_exp

def wout_ridge_regression(xt: Array, ut: Array, yt_hat: Array,
                          alpha: float) -> Array:
    """
        Compute updated weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        :return W: weight matrix of size (N, N)
        """
    S = xt.copy()
    if alpha is None or alpha == 0.:
        # linear regression
        # npy.dot(npy.linalg.pinv(S), yt_hat).T
        w_out = npy.dot(npy.linalg.pinv(S), yt_hat).T
    else:
        # ridge regression
        R = npy.dot(S.T, S)
        D = yt_hat
        P = npy.dot(S.T, D)
        w_out = np.dot(
            npy.linalg.inv(R + alpha * npy.eye(R.shape[0])), P).T
    return w_out


def initialize_wout(params, ut, yt, reg_wout=10, washout=0):
    idx = np.array([0, 1])

    # Record input driven dyna
    YX = jax.vmap(forward_esn, (None, None, 0, 0, None, None, None))(
        params, None, ut, idx, None, True, False)

    X = YX[:, :, ut.shape[2]:]
    y_esn = YX[:, :, :ut.shape[2]]

    X_flat = X.reshape(-1, X.shape[-1])
    X_flat_bias = np.hstack((X_flat, np.ones((X_flat.shape[0], 1))))
    Y_flat = yt.reshape(-1, yt.shape[-1])
    Wout_b = wout_ridge_regression(X_flat_bias, 0, Y_flat, reg_wout)

    Wout = Wout_b[:, :-1]
    b = Wout_b[:, -1]
    params['wout'] = Wout
    params['bias_out'] = b

    return params, X_flat, Y_flat

def initialize_wout3d(params, C_bias, B_bias, ut, yt, reg_wout=10, washout=0):
    idx = np.array([0, 1])

    # Record input driven dyna
    Y, X, X_exp = jax.vmap(forward_esn3d, (None,None, 0, 0, 0, None, None, None))(
        params, C_bias, B_bias, ut, idx, None, True, False)

    X_flat = X_exp.reshape(-1, X_exp.shape[-1])
    X_flat_bias = np.hstack((X_flat, np.ones((X_flat.shape[0], 1))))
    Y_flat = yt.reshape(-1, yt.shape[-1])
    Wout_b = wout_ridge_regression(X_flat_bias, 0, Y_flat, reg_wout)

    Wout = Wout_b[:, :-1]
    b = Wout_b[:, -1]
    params['wout'] = Wout
    params['bias_out'] = b

    return params, X_flat, Y_flat

def loss_fn(params, u_input, y_reconstruction, aperture, conceptor_loss_amp=0, washout=0):
    idx = np.array([0, 1])
    from jax import random
    # forward pass
    key = random.PRNGKey(0)
    x_init = None if washout == 0 else random.uniform(
        key, shape=(params['w'].shape[0],))
    YX = jax.vmap(forward_esn, (None, None, 0, 0, None, None, None))(
        params, None, u_input, idx, x_init, True, False)

    X = YX[:, :, u_input.shape[2]:]
    y_esn = YX[:, :, :u_input.shape[2]]
    error_per_sample = np.sum(
        (y_esn[:, washout:, :] - y_reconstruction[:, washout:, :]) ** 2, axis=2)
    error_per_sample = np.mean(error_per_sample, axis=1)

    # get conceptor
    R = jax.vmap(
        lambda X: np.dot(X.T, X)/X.shape[0], (0)
    )(X)

    C = np.array(
        [np.dot(r, np.linalg.inv(r + aperture ** (-2) * np.eye(r.shape[0]))) for r in R])

    M = jax.vmap(
        lambda x: np.mean(x, axis=0), (0))(X)

    Er_c = np.linalg.norm(C[0]-C[1])
    Er_mean = np.linalg.norm(M[0]-M[1])

    return np.mean(error_per_sample) + conceptor_loss_amp * (Er_c + 0*Er_mean/10) + 0.01*np.linalg.norm(params['wout']**2) + 0.01*np.linalg.norm(params['w']**2), (Er_c, Er_mean, error_per_sample, X)

def loss_fn3d(params, u_input, y_reconstruction, encoding, C_bias, B_bias, noise, p_forcing = True):
    idx = np.array([0, 1])
    y_esn, X, X_exp = jax.vmap(forward_esn3d, (None,None,0,0,0,None,None,None, None, None, None))(
        params, C_bias, B_bias, u_input, idx, None, encoding, False, True, None, noise)

    # YX = YX_df[0]
    # df = YX_df[1]
    # X = YX[:, :, u_input.shape[2]:]
    # y_esn = YX[:, :, :u_input.shape[2]]
    error_per_sample = np.sum((y_esn - y_reconstruction) ** 2, axis = 2)
    error_per_sample = np.mean(error_per_sample, axis = 1)
    

    # error_jcb = np.mean(df**2, axis =3)
    # error_jcb = np.mean(error_jcb, axis = 2)
    # error_jcb = np.mean(error_jcb, axis = 1)
    # error_jcb = np.mean(error_jcb, axis = 0)

    # run in decoding mode
    def forcing(params, u_input, y_reconstruction, idx, C_bias, B_bias, noise, y_esn):
        # stop grad
        y_esn = jax.lax.stop_gradient(y_esn)
        encoding = False
        y_esn_dec, X_dec, _ = jax.vmap(forward_esn3d, (None,None,0,0,0,None,None,None, None, None, None))(
        params, C_bias, B_bias, u_input, idx, None, encoding, False, True, None, noise)

        # compute loss between hidden states
        #error_state_per_sample = np.sum((X_dec - X) ** 2, axis = 2) + np.sum((y_esn_dec - y_esn) ** 2, axis = 2)
        error_state_per_sample = np.sum((y_esn_dec - y_esn) ** 2, axis = 2)
        error_state_per_sample = np.where(error_state_per_sample < 0.005**2, error_state_per_sample, np.zeros_like(error_state_per_sample))

        error_state_per_sample = np.mean(error_state_per_sample, axis = 1)
        Er_state = np.mean(error_state_per_sample, axis = 0)
        return Er_state
    Er_state = jax.lax.cond(
        p_forcing,
        lambda params, u_input, y_reconstruction, idx, C_bias, B_bias, noise, y_esn: forcing(params, u_input, y_reconstruction, idx, C_bias, B_bias, noise, y_esn),
        lambda params, u_input, y_reconstruction, idx, C_bias, B_bias, noise, y_esn: 0.,
        params, u_input, y_reconstruction, idx, C_bias, B_bias, noise, y_esn
    )

    Er_c = 0#np.linalg.norm(C[0]-C[1]) 
    Er_mean = 0#np.linalg.norm(M[0]-M[1])
    # Er_conceptor
    # Er_c = 0
    # Er_c = jax.vmap(lambda x, c, b: np.mean(np.linalg.norm((x - b) @ c - (x-b),axis =1)), (0, None, 0))(X, C_bias, B_bias)
    return np.mean(error_per_sample)+np.mean(Er_c) + Er_state/10000, (Er_c, Er_state, error_per_sample, X, y_esn)

@functools.partial(jax.jit, static_argnums=(4, 5, 7, 8, 9))
def update(params, u_input, y_reconstruction, opt_state, opt_update, get_params, epoch_idx, aperture, conceptor_loss_amp=0, washout=0):

    params = get_params(opt_state)

    (loss, tuple_encoding), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params, u_input, y_reconstruction, aperture, conceptor_loss_amp, washout)
    er_c, er_mean, er_y, X = tuple_encoding

    nrm = jax.tree_map(
        lambda g: np.linalg.norm(g), grads
    )

    def clipping(g):
        g = jax.lax.cond(
            np.linalg.norm(g) > 500,
            lambda g: g/np.linalg.norm(g),
            lambda g: g,
            g
        )
        return g
    grads = jax.tree_map(
        lambda g: clipping(g), grads
    )
    grads_norm, value_tree = jax.tree_util.tree_flatten(nrm)

    # SGD update
    opt_state = opt_update(epoch_idx, grads, opt_state)
    return opt_state, loss, er_c, er_mean, er_y, X, grads_norm

@functools.partial(jax.jit, static_argnums=(3, 4, 6))
def update3d(u_input, y_reconstruction, opt_state, opt_update, get_params, epoch_idx, encoding, C_bias, B_bias, noise, p_forcing):
    params = get_params(opt_state)

    (loss, tuple_encoding), grads = jax.value_and_grad(loss_fn3d, has_aux = True)(params, u_input, y_reconstruction, encoding, C_bias, B_bias, noise, p_forcing)
    er_c, er_mean, er_y, X, y_esn = tuple_encoding

    # clip gradient
    nrm = jax.tree_map(
        lambda g: np.linalg.norm(g), grads
        )
    def clipping(g):
        g = jax.lax.cond(
            np.linalg.norm(g) > 500,
            lambda g: g/np.linalg.norm(g),
            lambda g: g,
            g
        )
        # if the gradient is nan then return 0
        g = jax.lax.cond(
            np.isnan(np.linalg.norm(g)),
            lambda g: np.zeros(g.shape),
            lambda g: g,
            g
        )
        return g
    
    grads = jax.tree_map(
        lambda g: clipping(g), grads  
        )
    grads_norm, value_tree = jax.tree_util.tree_flatten(nrm)


    # SGD update
    opt_state = opt_update(epoch_idx, grads, opt_state)
    return opt_state, loss, er_c, er_mean, er_y, X, y_esn, grads_norm

def visualize_interpolation(params, C, ut_train, log_folder, filename):

    len_seqs = 200

    plt.figure()
    # compute how the system interpolate
    for lamda in [0, 0.5, 1]:
        # t_interp = 100
        t_interp = np.ones(len_seqs)*lamda
        ut_interp = np.zeros(len_seqs)
        YX_interpolation = forward_esn_interp(
            params, C, ut_interp, None, t_interp=t_interp)

        X_interp = YX_interpolation[:, ut_train.shape[2]:]
        y_esn_interp = YX_interpolation[:, :ut_train.shape[2]]

        plt.plot(y_esn_interp)

    plt.savefig(f'{log_folder}/plots/interpolation_{filename}.png')
    plt.close()

def visualize_decoding(params, idx, C_bias, B_bias, ut_train, log_folder, filename):

    
    # compute the system autonomously
    y_esn_auto, X_auto, X_exp = jax.vmap(forward_esn3d,(None, None,0,0,0,None,None,None,None,None))(
        params, C_bias, B_bias, ut_train, idx, None, False, True, True, None)
    plt.figure()
    plt.plot(y_esn_auto[0, :, 0], y_esn_auto[0, :, 1], 'r', label = 'robot')
    plt.plot(y_esn_auto[1, :, 0], y_esn_auto[1, :, 1], 'g', label = 'human')
    plt.legend()        
    plt.savefig(f'{log_folder}/plots/auto_{filename}.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_auto[0,:,0], X_auto[0,:,1], X_auto[0,:,2], 'r', label = 'robot')
    ax.plot(X_auto[1,:,0], X_auto[1,:,1], X_auto[1,:,2], 'g', label = 'human')
    plt.legend()
    plt.savefig(f'{log_folder}/plots/auto3d_{filename}.png')
    plt.close()

    plt.figure()
    for i in range(3):
        plt.plot(X_auto[0,:,i], label = f'robot_{i}')
        plt.plot(X_auto[1,:,i], label = f'human_{i}')
    plt.legend()
    plt.savefig(f'{log_folder}/plots/auto_2d{filename}.png')
    plt.close() 

def visualize_data(X, y_esn, log_folder, filename):
    plt.figure()
    plt.plot(y_esn[0, :, 0], y_esn[0, :, 1], 'r', label = 'robot')
    plt.plot(y_esn[1, :, 0], y_esn[1, :, 1], 'g', label = 'human')
    plt.legend()
    plt.savefig(f'{log_folder}/plots/input_{filename}.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0,:,0], X[0,:,1], X[0,:,2], 'r', label = 'robot')
    ax.plot(X[1,:,0], X[1,:,1], X[1,:,2], 'g', label = 'human')
    plt.legend()
    plt.savefig(f'{log_folder}/plots/input3d_{filename}.png')
    plt.close()

    plt.figure()
    for i in range(3):
        plt.plot(X[0,:,i], label = f'robot_{i}')
        plt.plot(X[1,:,i], label = f'human_{i}')
    plt.legend()
    plt.savefig(f'{log_folder}/plots/input_2d{filename}.png')
    plt.close() 


# Conceptor helper

def affine_conceptor(x_shift):
    """
    Return the affine conceptor projecting on the yz plane with a shift of +-x_shift
    
    """
    S = np.array([[0, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])

    U = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    b1 = np.array([-x_shift, 0, 0])
    b2 = np.array([x_shift, -0, 0])


    # normal
    B_bias = np.stack([b1, b2], axis=0)

    C_bias = U @ S @ U.T
    return C_bias, B_bias

# create a jupyter cell
#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, color = 'black')

plt.show()
# %%
