import os
from tqdm import tqdm
from absl import flags, app

import jax
import jax.numpy as np
from jax.example_libraries import optimizers as jax_opt

from numpy import Inf, NaN

from rnn_utils import update
# from rnn_utils import esn_params
# from rnn_utils import initialize_wout
from rnn_utils import visualize_interpolation
from rnn_utils import esn3d_params, initialize_wout3d, update3d, affine_conceptor, visualize_decoding_interp, visualize_data
import shutil

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


flags.DEFINE_string("name", "sine_wave_interp",
                    "name of this training run/experiment")
flags.DEFINE_string("logdir", "./logs", "path to the log directory")
flags.DEFINE_integer("num_epochs", 3050, "number of training epochs")
flags.DEFINE_integer("steps_per_eval", 100,
                     "number of training steps per evaluation")
flags.DEFINE_integer("washout", 0, "washout period")
flags.DEFINE_float("noise", 0., "noise injected in the state update")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 21, "seed for random number generators")
flags.DEFINE_float("conceptor_loss_amp", 1/50, "conceptor loss amplitude")
flags.DEFINE_string("mlp_size_hidden", "[512]", "size of the mlp hidden layers  [n1_in, n2_in, ...]")
flags.DEFINE_string("mlp_in_out", "[[],[]]", "size of the mlp hidden layers of input and oupt [[n1_in, n2_in, ...],[n1_out, n2_out, ...]]")
flags.DEFINE_string("data", "robot_human_data.npy", "data file")
flags.DEFINE_float("a_dt", 0.1, "leakage para of the esn")
flags.DEFINE_bool("p_forcing", True, "forcing probability")
flags.DEFINE_string("loading", "None", "whether to load a checkpoint")
flags.DEFINE_bool("sep", False, "enforce separation of weights")
flags.DEFINE_float("interp_range", 0.5, "range of interpolation")
flags.DEFINE_float("interp_range_mixing", 0.5, "range of mixing the parameters")

def main(_):




    data = np.load(f"robot_data/{FLAGS.data}")
    data_sample = np.array([0, -1])
    ut_train = data[data_sample, 0:data.shape[1]-1, :]
    yt_train = data[data_sample, 1:data.shape[1], :]

    log_folder = setup_logging_directory(FLAGS.logdir, FLAGS.name)
    tb_writer = SummaryWriter(log_dir=log_folder)
    os.makedirs(os.path.join(log_folder, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'scripts'), exist_ok=True)

    # store scripts and flags
    shutil.copy("conceptor_aided_recurrent_autoencoder_robot_human.py", os.path.join(log_folder, 'scripts'))
    shutil.copy("rnn_utils.py", os.path.join(log_folder, 'scripts'))
    with open(os.path.join(log_folder, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    if FLAGS.loading == "None":
        params_esn = esn3d_params(1., 3, 2, 1., FLAGS.a_dt, eval(FLAGS.mlp_size_hidden), eval(FLAGS.mlp_in_out), 0.8, seed=FLAGS.seed)
    else:
        params_esn = np.load(FLAGS.loading)
        params_esn = {key: params_esn[key] for key in params_esn.keys()}
    
    # get the affine conceptor
    C_bias, B_bias = affine_conceptor(FLAGS.interp_range)

    # params_esn, _, _ = initialize_wout3d(
    #     params_esn, C_bias, B_bias, ut_train, yt_train, reg_wout=10)

    opt_init, opt_update, get_params = jax_opt.adam(FLAGS.learning_rate)
    opt_state = opt_init(params_esn.copy())

    # check if gpu is used
    print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

    for epoch_idx in tqdm(range(FLAGS.num_epochs)):
        if epoch_idx > 100:
            p_forcing = FLAGS.p_forcing
        else:
            p_forcing = False


        opt_state, loss, er_c, er_mean, er_y, X, y_esn, grads_norm = update3d(ut_train,
                                                                     yt_train,
                                                                     opt_state,
                                                                     opt_update,
                                                                     get_params,
                                                                     epoch_idx,
                                                                     encoding=True,
                                                                     C_bias = C_bias,
                                                                     B_bias = B_bias,
                                                                     noise=FLAGS.noise,
                                                                     p_forcing=p_forcing,
                                                                     sep = FLAGS.sep,
                                                                     interp_range=FLAGS.interp_range,
                                                                     interp_range_mixing=FLAGS.interp_range_mixing,)

        # log losses to tensorboard
        tb_writer.add_scalar("loss", loss.item(), epoch_idx)
        tb_writer.add_scalar("loss_c", er_c.item(), epoch_idx)
        tb_writer.add_scalar("loss_c_mean", er_mean.item(), epoch_idx)
        tb_writer.add_scalar("loss_rec", np.mean(er_y).item(), epoch_idx)
        tb_writer.add_scalar("grads_norm", grads_norm[0].item(), epoch_idx)

        if epoch_idx % FLAGS.steps_per_eval == 0:
            # visualize input driven
            visualize_data(X,y_esn, log_folder, f"{epoch_idx:03}")

            idx = np.array([0, 1])
            params = get_params(opt_state)
            visualize_decoding_interp(params, idx, C_bias, B_bias,
                                    log_folder,
                                    f"{epoch_idx:03}", interp=True, vector_field=True, interp_range=FLAGS.interp_range)
            # visualize_interpolation3d(params, C)

            # save params
            np.savez(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", **{
                key: params[key].__array__() for key in params.keys()
            })
            # load params
            params = np.load(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz")
            params = {key: params[key] for key in params.keys()}



if __name__ == "__main__":
    app.run(main)
