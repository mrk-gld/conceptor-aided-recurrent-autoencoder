import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

from utils.rnn_utils import forward_rnn_interp as forward_rnn_interp_rnn
from utils.lstm_utils import forward_rnn_interp as forward_rnn_interp_lstm


def setup_logging_directory(logdir, name):
    """
    Create a new logging directory with the given name, or use the next available index.

    Returns:
    - directory of the created log folder
    """
    if not os.path.exists(f"{logdir}/{name}"):
        log_folder = f"{logdir}/{name}"
    else:
        last_idx = 1
        while os.path.exists(f"{logdir}/{name}_{last_idx}"):
            last_idx += 1
        log_folder = f"{logdir}/{name}_{last_idx}"
    return log_folder


def visualize_sine_interpolation(params, conceptors, log_folder, fname, len_seqs=300, ntype='rnn'):
    """
    Visualizes the interpolation of a sine wave using a recurrent neural network.

    Args:
    - params: dictionary containing the parameters of the RNN
    - conceptors: dictionary containing the conceptors of the RNN
    - log_folder: string representing the path to the log folder
    - filename: string representing the name of the file to save the plot
    - len_seqs: integer representing the length of the sequence to interpolate

    Returns:
    - None
    """
    forward_rnn_interp = forward_rnn_interp_rnn if ntype == 'rnn' else forward_rnn_interp_lstm
    fig, axs = plt.subplots(5, sharex=True, sharey=True)
    # compute how the system interpolate
    for idx, lamda in enumerate([0, 0.25, 0.5, 0.75, 1]):
        # t_interp = 100
        lambda_t = jnp.ones(len_seqs)*lamda
        x_interp, y_interp = forward_rnn_interp(
            params, conceptors, None, lambda_t)

        axs[idx].plot(y_interp, label=r"$\lambda=${}".format(lamda))
        axs[idx].legend(frameon=False)

    axs[idx].set_ylabel("y(k)")
    axs[idx].set_xlabel("k")
    plt.savefig(f'{log_folder}/plots/interpolation_{fname}.png')
    plt.close()


def visualize_mocap_interpolation(params, conceptors, log_folder, filename):
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
        lambda_t = jnp.ones(len_seqs) * lamda
        x_interp, y_interp = forward_rnn_interp(
            params, conceptors, None, lambda_t=lambda_t)

        states.append(x_interp)

    states_conca = jnp.concatenate(states, axis=0)
    pca = PCA(n_components=2)
    pca.fit(states_conca)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for lamda, state in zip(lamdas, states):
        data_pca = pca.transform(state)
        ax.plot(data_pca[:, 0], data_pca[:, 1], lamda,label=r"$\lambda$={}".format(lamda))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{log_folder}/plots/interpolation_{filename}.png')
    plt.close()
    