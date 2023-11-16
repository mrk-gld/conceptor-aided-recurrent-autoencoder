import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.rnn_utils import forward_rnn_interp as forward_rnn_interp_rnn
from utils.lstm_utils import forward_rnn_interp as forward_rnn_interp_lstm

from utils.rnn_utils import forward_rnn_interp

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
        x_interp, y_interp = forward_rnn_interp(
            params, conceptors, None, ratio=lamda, length=len_seqs)

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
        x_interp, y_interp = forward_rnn_interp(
            params, conceptors, None, ratio=lamda,length=600)

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
    
    
def compute_JS_divergence_and_acf(timeseries1, timeseries2, timeseries_interp, lag=10, bins=50):
    
    def compute_autocorrelation(x1):
        acf_full = []
        x1 = np.array(x1)
        for q in range(x1.shape[1]):
            acf = []
            for i in range(1,lag):
                acf.append(np.corrcoef(x1[:-i,q], x1[i:,q])[0,1])
            acf_full.append(np.abs(acf))
        return np.array(acf_full)
    
    def jsd(P,Q):
        def kl_divergence(A,B):
            return np.sum(A * np.log2(A / B))
        
        M = 0.5 * (P + Q)
        return 0.5 * (kl_divergence(P, M) + kl_divergence(Q, M))
    
    if timeseries1.ndim == 1:
        timeseries1 = timeseries1.reshape(-1,1)
    if timeseries2.ndim == 1:
        timeseries2 = timeseries2.reshape(-1,1)
    if timeseries_interp.ndim == 1:
        timeseries_interp = timeseries_interp.reshape(-1,1)
        
    # resize all timeseries to the same length
    min_len = np.min([timeseries1.shape[0], timeseries2.shape[0], timeseries_interp.shape[0]])
    timeseries1 = timeseries1[-min_len:,:]
    timeseries2 = timeseries2[-min_len:,:]
    timeseries_interp = timeseries_interp[-min_len:,:]
    
    max_val = np.max([timeseries1.max(), timeseries2.max()])
    min_val = np.min([timeseries1.min(), timeseries2.min()])
    
    bins = np.linspace(min_val, max_val, bins)
    
    jsd_1_curr = []
    jsd_2_curr = []        
    
    for p in range(timeseries1.shape[1]):
        prob1 = np.histogram(timeseries1[:,p], bins=bins, density=True)[0] + 1e-18
        prob2 = np.histogram(timeseries2[:,p], bins=bins, density=True)[0] + 1e-18
        prob_interp = np.histogram(timeseries_interp[:,p], bins=bins, density=True)[0] + 1e-18

        prob1 = prob1/np.sum(prob1)
        prob2 = prob2/np.sum(prob2)
        prob_interp = prob_interp/np.sum(prob_interp)
        
        jsd_divergence_1 = jsd(prob1, prob_interp)
        jsd_divergence_2 = jsd(prob2, prob_interp)
        
        jsd_1_curr.append(jsd_divergence_1)
        jsd_2_curr.append(jsd_divergence_2)
    
    jsd_1_curr = np.array(jsd_1_curr)
    jsd_2_curr = np.array(jsd_2_curr)
    
    jsd_1_curr[~np.isfinite(jsd_1_curr)] = 1
    jsd_2_curr[~np.isfinite(jsd_2_curr)] = 1
        
    jsd_1 = np.mean(jsd_1_curr)
    jsd_2 = np.mean(jsd_2_curr)
    
    acf_1 = compute_autocorrelation(timeseries1)
    acf_2 = compute_autocorrelation(timeseries2)
    acf_interp = compute_autocorrelation(timeseries_interp)
    
    # compute the difference in autocorrelation
    acf_diff_1 = np.abs(acf_1 - acf_interp)
    acf_diff_2 = np.abs(acf_2 - acf_interp)
    acf_1 = np.mean(acf_diff_1)
    acf_2 = np.mean(acf_diff_2)
    
    return jsd_1, jsd_2, acf_1, acf_2
