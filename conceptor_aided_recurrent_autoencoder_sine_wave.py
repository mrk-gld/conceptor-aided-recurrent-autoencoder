import os
from tqdm import tqdm
from absl import flags, app

import jax
import jax.numpy as np
from jax.example_libraries import optimizers as jax_opt

from numpy import Inf, NaN

from rnn_utils import update
from rnn_utils import esn_params
from rnn_utils import initialize_wout
from rnn_utils import visualize_interpolation

from torch.utils.tensorboard import SummaryWriter

# define flags
FLAGS = flags.FLAGS


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

# linear interpolation between sine waves of different frequencies (parametrisation of similar to Wyffels et al. (2014))


def sine_wave(t_pattern, s):
    signal = []
    for i in range(t_pattern):
        signal.append(np.sin(0.075*s*i))
    return np.array(signal)


flags.DEFINE_string("name", "sine_wave_interp",
                    "name of this training run/experiment")
flags.DEFINE_string("logdir", "./logs", "path to the log directory")
flags.DEFINE_integer("num_epochs", 3050, "number of training epochs")
flags.DEFINE_integer("steps_per_eval", 100,
                     "number of training steps per evaluation")
flags.DEFINE_integer("washout", 0, "washout period")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 42, "seed for random number generators")
flags.DEFINE_float("conceptor_loss_amp", 1/50, "conceptor loss amplitude")


def main(_):

    t_pattern = 300
    data = jax.vmap(sine_wave, in_axes=(None, 0))(
        t_pattern, np.linspace(0.5, 3, 10))
    data = np.expand_dims(data, axis=2)

    data_sample = np.array([0, -1])
    ut_train = data[data_sample, 0:data.shape[1]-1, :]
    yt_train = data[data_sample, 1:data.shape[1], :]

    log_folder = setup_logging_directory(FLAGS.logdir, FLAGS.name)
    tb_writer = SummaryWriter(log_dir=log_folder)
    os.makedirs(os.path.join(log_folder, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'plots'), exist_ok=True)

    params_ini = esn_params(1., 512, 1., 0.1, 0.8, seed=21)

    params_esn, _, _ = initialize_wout(
        params_ini.copy(), ut_train, yt_train, reg_wout=10)

    opt_init, opt_update, get_params = jax_opt.adam(FLAGS.learning_rate)
    opt_state = opt_init(params_esn.copy())

    for epoch_idx in tqdm(range(FLAGS.num_epochs)):

        opt_state, loss, er_c, er_mean, er_y, X, grads_norm = update(params_esn,
                                                                     ut_train,
                                                                     yt_train,
                                                                     opt_state,
                                                                     opt_update,
                                                                     get_params,
                                                                     epoch_idx,
                                                                     aperture=10.,
                                                                     washout=FLAGS.washout,
                                                                     conceptor_loss_amp=FLAGS.conceptor_loss_amp)

        # log losses to tensorboard
        tb_writer.add_scalar("loss", loss.item(), epoch_idx)
        tb_writer.add_scalar("loss_c", er_c.item(), epoch_idx)
        tb_writer.add_scalar("loss_c_mean", er_mean.item(), epoch_idx)
        tb_writer.add_scalar("loss_rec", np.mean(er_y).item(), epoch_idx)
        tb_writer.add_scalar("grads_norm", grads_norm[0].item(), epoch_idx)

        if epoch_idx % FLAGS.steps_per_eval == 0:
            R = jax.vmap(
                lambda X: np.dot(X.T, X)/X.shape[0], (0))(X[:, FLAGS.washout:, :])
            U, S, V = jax.vmap(
                lambda R: np.linalg.svd(R, full_matrices=False, hermitian=True), (0))(R)
            C = jax.vmap(
                lambda U, S: U*(S/(S+0.001*np.ones(S.shape)))@U.T, (0, 0))(U, S)

            params = get_params(opt_state)
            visualize_interpolation(params, C,
                                    ut_train,
                                    log_folder,
                                    f"{epoch_idx:03}")

            # save params
            np.savez(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", **{
                key: params[key].__array__() for key in params.keys()
            })

            conceptor = {"C_1": C[0], "C_2": C[1]}
            # save params
            np.savez(f"{log_folder}/ckpt/conceptor_{epoch_idx+1:03}.npz", **{
                key: np.array(conceptor[key]) for key in conceptor.keys()
            })


if __name__ == "__main__":
    app.run(main)