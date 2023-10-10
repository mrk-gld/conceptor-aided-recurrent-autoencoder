import functools
import numpy as npy

import matplotlib.pyplot as plt

import jax
import jax.numpy as np

from jax import jit
from jax import Array
from jax.example_libraries import optimizers as jax_opt


def esn_params(esn_size, input_size, output_size, input_scaling, spectral_radius, a_dt, bias_scaling=0.8, seed=1235):
    
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


@functools.partial(jax.jit, static_argnums=(4, 5, 9, 10))
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
