import os
import jax.numpy as jnp

from numpy import Inf, NaN
import numpy as np

# from utils.rnn_utils import forward_rnn_interp
from utils.lstm_utils import forward_rnn_interp
from utils.mocap_utils import get_mocap_data

from utils.video_processing import convert_motion_data_to_video
from utils.skeleton import *
import base64


folder = "motion_data_numpy/"
dataset_names = ["walk_15", "run_55"]
_, _, mean, std = get_mocap_data(folder, dataset_names=dataset_names)


folder = "./logs/mocap_conceptor_loss_1/"
epoch = "051"
folder = "./logs/lstm_mocap_xav_comp_b08_lk01"
epoch = "1901"
ckpt_path = os.path.join(folder, "ckpt")
params_file = f"params_{epoch}.npz.npy"
conceptor_file = f"conceptor_{epoch}.npz"

params = np.load(os.path.join(ckpt_path, params_file), allow_pickle=True)[()]
params = dict(params)
rnn_size = params["wout"].shape[1]
conceptors = np.load(os.path.join(ckpt_path, conceptor_file), allow_pickle=True)
conceptors = dict(conceptors)
c_matrix = [conceptors[f"C_{i+1}"] for i in range(len(conceptors))]

len_seqs = 400
lamdas = [0, 0.25, 0.5, 0.75, 1]

states_lamda = []
y_interp_lamda = []
for lamda in lamdas:
    # TODO: use lstm version of this
    x_interp, y_interp = forward_rnn_interp(params, c_matrix, None, length=len_seqs)
    # lambda_t = jnp.ones(len_seqs) * lamda
    # x_interp, y_interp = forward_rnn_interp(params, c_matrix, None, lambda_t=lambda_t)

    states_lamda.append(x_interp)
    y_interp_lamda.append(y_interp)

# load initial conditions
data_ini_walk = np.load("motion_data_numpy/data_ini_walk.npy")
data_ini_walk = [[data_ini_walk[0], data_ini_walk[1]], [data_ini_walk[2]]]

c_joints = np.load("motion_data_numpy/c_joint.npy", allow_pickle=True).item()
norm_ij = np.load("motion_data_numpy/norm_ij.npy")

files = convert_motion_data_to_video(
    folder,
    np.array(y_interp_lamda),
    mean,
    std,
    [f"epoch_{epoch}_lambda_{l}" for l in lamdas],
)
