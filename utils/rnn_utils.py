import functools
import numpy as npy
import optax

import matplotlib.pyplot as plt

import jax
import jax.numpy as np

from jax import random
from jax import jit
from jax import Array
from jax.example_libraries import optimizers as jax_opt


def rnn_params(rnn_size, input_size, output_size, input_scaling, spectral_radius, a_dt, bias_scaling=0.8, seed=1235):
    """
    Initializes the parameters for a simple RNN model.

    Args:
    rnn_size (int): The number of hidden units in the RNN.
    input_size (int): The number of input features.
    output_size (int): The number of output features.
    input_scaling (float): Scaling factor for the input weights.
    spectral_radius (float): Desired spectral radius of the recurrent weight matrix.
    a_dt (float): Time step size.
    bias_scaling (float, optional): Scaling factor for the bias terms. Defaults to 0.8.
    seed (int, optional): Seed for the random number generator. Defaults to 1235.

    Returns:
    dict: A dictionary containing the initialized parameters.
    """

    prng = npy.random.default_rng(seed)

    def rnn_ini(shape, spectral_radius):
        w = prng.normal(size=shape)
        current_spectral_radius = max(abs(npy.linalg.eig(w)[0]))
        w *= spectral_radius / current_spectral_radius
        return w

    params = dict(
        win=prng.normal(size=(rnn_size, input_size))*input_scaling,
        w=rnn_ini((rnn_size, rnn_size), spectral_radius),
        bias=prng.normal(size=(rnn_size,))*bias_scaling,
        wout=prng.normal(size=(output_size, rnn_size)),
        bias_out=prng.normal(size=(output_size,))*bias_scaling,
        a_dt=a_dt*np.ones(rnn_size),
        x_ini=0.1*prng.normal(size=(rnn_size))
    )

    return jax.tree_map(lambda x: np.array(x), params)

@functools.partial(jax.jit, static_argnums=(4))
def forward_rnn(params, conceptor, ut, x_init=None, autoregressive=False):
    """
    Forward pass of a recurrent neural network (RNN) with optional autoregressive mode.

    Args:
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - conceptor (ndarray): matrix used to constrain the RNN dynamics.
    - ut (ndarray): input to the RNN.
    - idx (int): index of the RNN to use (in case multiple RNNs are stored in params).
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autoregressive (bool, optional): whether to use autoregressive output. Defaults to False.

    Returns:
    - YX (ndarray): output of the RNN.
    """
    if x_init is None:
        x_init = params["x_ini"]

    if conceptor is None:
        conceptor = np.eye(x_init.shape[0])

    def apply_fun_scan(params, xy, ut, autoregressive=False):
        x, y = xy

        ut = ut if not autoregressive else np.dot(
            params['wout'], x) + params['bias_out']

        x = conceptor @ ((1-params["a_dt"])*x + params["a_dt"]*np.tanh(
            params['w'] @ x + params['win'] @ ut + params['bias']))

        y = params['wout'] @ x + params['bias_out']

        xy = (x, y)

        return xy, np.concatenate((y, x))

    f = functools.partial(apply_fun_scan, params,
                          autoregressive=autoregressive)
    xy = (x_init, np.zeros(params['bias_out'].shape[0]))
    _, YX = jax.lax.scan(f, xy, ut)
    return YX

@jit
def forward_rnn_interp(params, C_manifold, ut, x_init, t_interp):
    
    if x_init is None:
        x_init = params["x_ini"]

    def apply_fun_scan(params, xyc, ut):

        x, y, count = xyc
        ratio = t_interp[count]

        C_fb = (1-ratio)*C_manifold[0] + ratio*C_manifold[1]

        ut = params['wout'] @ x + params['bias_out']

        x = C_fb @ ((1-params["a_dt"])*x + params["a_dt"]*np.tanh(
            params['w'] @ x + params['win'] @ ut + params['bias']))

        y = params['wout'] @ x + params['bias_out']

        xyc = (x, y, count+1)

        return xyc, np.concatenate((y, x))

    f = functools.partial(apply_fun_scan, params)
    xyc = (x_init, np.zeros(params['bias_out'].shape[0]), 0)
    _, YX = jax.lax.scan(f, xyc, ut)
    return YX


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


def initialize_wout(params, ut, yt, reg_wout=10):

    # Record input driven dyna
    yx_vmap = jax.vmap(forward_rnn, in_axes=(None, None, 0, None, None))
    xy = yx_vmap(params,None,ut,params["x_ini"],False)

    X = xy[:, :, ut.shape[2]:]

    X_flat = X.reshape(-1, X.shape[-1])
    X_flat_bias = np.hstack((X_flat, np.ones((X_flat.shape[0], 1))))
    Y_flat = yt.reshape(-1, yt.shape[-1])
    Wout_b = wout_ridge_regression(X_flat_bias, 0, Y_flat, reg_wout)

    Wout = Wout_b[:, :-1]
    b = Wout_b[:, -1]
    params['wout'] = Wout
    params['bias_out'] = b

    return params, X_flat, Y_flat


def compute_conceptor(X, aperture):
    """
    Computes the conceptor matrix for a given input matrix X and an aperture value.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (n_samples, n_features).
    aperture (float): Aperture value used to compute the conceptor matrix.

    Returns:
    numpy.ndarray: Conceptor matrix of shape (n_features, n_features).
    """
    R = np.dot(X.T, X) / X.shape[0]
    C = np.dot(R, np.linalg.inv(R + aperture ** (-2) * np.eye(R.shape[0])))
    return C


def loss_fn(params, u_input, y_reconstruction, aperture, beta_1=0, beta_2=0, washout=0):

    key = random.PRNGKey(0)

    x_init = None if washout == 0 else random.uniform(
        key, shape=(params['w'].shape[0],))

    forward_vmap = jax.vmap(forward_rnn, (None, None, 0, None, None))
    yx = forward_vmap(params,None,u_input,x_init,False)

    X = yx[:, :, u_input.shape[2]:]
    y_rnn = yx[:, :, :u_input.shape[2]]

    error_per_sample = np.sum(
        (y_rnn[:, washout:, :] - y_reconstruction[:, washout:, :]) ** 2, axis=2)
    error_per_sample = np.mean(error_per_sample, axis=1)

    # get conceptor
    C = jax.vmap(lambda x: compute_conceptor(x, aperture))(X)

    M = jax.vmap(
        lambda x: np.mean(x, axis=0), (0))(X)

    err_mse = np.mean(error_per_sample)
    err_c = np.linalg.norm(C[0]-C[1])
    err_c_mean = np.linalg.norm(M[0]-M[1])
    ridge = np.linalg.norm(params['wout']**2) + np.linalg.norm(params['w']**2)
    loss = err_mse + beta_1 * err_c + beta_2 * err_c_mean #+ 0.01 * ridge

    return loss, (err_c, err_c_mean, error_per_sample, X)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def update(params, u_input, y_reconstruction, opt_state, opt_update, aperture, beta_1=0, beta_2=0, washout=0):
    """
    Update the parameters of a recurrent neural network using the given input and output data.

    Args:
    - params: the current parameters of the RNN
    - u_input: the input data to the RNN
    - y_reconstruction: the output data from the RNN
    - opt_state: the current state of the optimizer
    - opt_update: the update function for the optimizer
    - aperture: the size of the conceptor aperture used in the loss function
    - beta_1: the beta_1 parameter for the conceptor distance loss (default 0)
    - beta_2: the beta_2 parameter for the mean state loss (default 0)
    - washout: the number of time steps to use for washout (default 0)

    Returns:
    - opt_state: the updated state of the optimizer
    - loss: the current loss value
    - err_c: the conceptor error value
    - err_c_mean: the current mean error value for the states
    - err_mse: the current mean squared error value
    - X: the current output of the RNN
    - grads_norm: the norm of the gradients used in the update
    """
    (loss, ret), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params,
                               u_input,
                               y_reconstruction,
                               aperture,
                               beta_1,
                               beta_2,
                               washout)

    err_c, err_c_mean, err_mse, X = ret

    # compute gradient norms
    nrm = jax.tree_map(
        lambda g: np.linalg.norm(g), grads
    )
    grads_norm, _ = jax.tree_util.tree_flatten(nrm)

    # gradient update

    def update_param(grads, opt_state, params):
        updates, opt_state = opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params)

    opt_state, params = update_param(grads, opt_state, params)

    return params, opt_state, loss, err_c, err_c_mean, err_mse, X, grads_norm
