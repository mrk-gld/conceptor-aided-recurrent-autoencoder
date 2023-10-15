import jax
import jax.numpy as np
from jax import random


import numpy as npy

# Conceptor helpers

## Generate affine conceptor
def affine_conceptor(x_shift):
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

# RNN helpers

def esn3d_params(input_scaling, esn_size, data_size, spectral_radius, a_dt, mlp_size_hidden, bias_scaling = 0.8, seed = 1235):
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

    params['wout0'] = prng.normal(size=(output_size, mlp_size_hidden[0]))
    params['wout1'] = prng.normal(size=(output_size, mlp_size_hidden[0]))
    params['w0'] = esn_ini((esn_size,esn_size), spectral_radius, prng)
    params['w1'] = esn_ini((esn_size,esn_size), spectral_radius, prng)
    params['win0'] = prng.normal(size=(esn_size,input_size))*input_scaling
    params['win1'] = prng.normal(size=(esn_size,input_size))*input_scaling


    params['x_ini0']=  0.1*prng.normal(size=(esn_size))# B_bias[0]#npy.array(C_bias) @ (params['x_ini'][0,0] - npy.array(B_bias[0])) + npy.array(B_bias[0])
    params['x_ini1']= 0.1*prng.normal(size=(esn_size))#B_bias[1]#C_bias @ (params['x_ini'][1,0] - B_bias[1]) + B_bias[1]
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
    
    mlp_size = [esn_size] + mlp_size_hidden + [esn_size]
    params['mlp_state0'] = initialize_mlp(mlp_size, prng)
    params['mlp_state1'] = initialize_mlp(mlp_size, prng)

    return jax.tree_map(lambda x: np.array(x), params)
    

def mlp(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:]:
        activations = np.tanh(w @ activations + b) 

    return activations

def mlp_eff(params, interp, in_array):
    """ Compute the forward pass for each example individually with interpolation of parameters"""
    activations = in_array

    # Loop over the ReLU hidden layers
    for (w0, b0), (w1, b1) in zip(params["mlp_state0"], params["mlp_state1"]):
        w_eff = (1-interp) * w0 + interp * w1
        b_eff = (1-interp) * b0 + interp * b1
        activations = np.tanh(w_eff @ activations + b_eff) 

    return activations

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