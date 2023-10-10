import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rnn_utils import forward_esn_interp

def setup_logging_directory(logdir, name):
    """Create a new logging directory with the given name, or use the next available index."""
    if not os.path.exists(f"{logdir}/{name}"):
        log_folder = f"{logdir}/{name}"
    else:
        last_idx = 1
        while os.path.exists(f"{logdir}/{name}_{last_idx}"):
            last_idx += 1
        log_folder = f"{logdir}/{name}_{last_idx}"
    return log_folder

def visualize_sine_interpolation(params, C, ut_train, log_folder, filename):

    len_seqs = 200

    plt.figure()
    # compute how the system interpolate
    for lamda in [0, 0.5, 1]:
        # t_interp = 100
        t_interp = jnp.ones(len_seqs)*lamda
        ut_interp = jnp.zeros(len_seqs)
        YX_interpolation = forward_esn_interp(
            params, C, ut_interp, None, t_interp=t_interp)

        X_interp = YX_interpolation[:, ut_train.shape[2]:]
        y_esn_interp = YX_interpolation[:, :ut_train.shape[2]]

        plt.plot(y_esn_interp)

    plt.savefig(f'{log_folder}/plots/interpolation_{filename}.png')
    plt.close()
    pass


def visualize_mocap_interpolation(params, c_matrix, ut_train, log_folder, filename):
    """
    Visualize the interpolation of the motion capture data using the Echo State Network.

    Args:
        params (dict): Dictionary containing the parameters of the Echo State Network.
        c_matrix (ndarray): The matrix of the input weights of the Echo State Network.
        ut_train (ndarray): The training input data of the Echo State Network.
        log_folder (str): The path of the logging directory.
        filename (str): The name of the file to save the plot.

    Returns:
        None
    """
    len_seqs = 200
    # compute how the system interpolates
    states = []
    lamdas = [0, 0.25, 0.5, 0.75, 1]
    for lamda in lamdas:
        t_interp = jnp.ones(len_seqs) * lamda
        ut_interp = jnp.zeros(len_seqs)
        yx_interpolation = forward_esn_interp(
            params, c_matrix, ut_interp, None, t_interp=t_interp)

        x_interpolation = yx_interpolation[:, ut_train.shape[2]:]
        states.append(x_interpolation)

    states_conca = jnp.concatenate(states, axis=0)
    pca = PCA(n_components=2)
    pca.fit(states_conca)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for lamda, state in zip(lamdas, states):
        data_pca = pca.transform(state)
        ax.plot(data_pca[:, 0], data_pca[:, 1], lamda)

    plt.tight_layout()
    plt.savefig(f'{log_folder}/plots/interpolation_{filename}.png')
    plt.close()
