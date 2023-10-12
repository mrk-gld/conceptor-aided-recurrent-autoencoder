import os
from tqdm import tqdm
from absl import flags, app
import optax
import jax
import jax.numpy as np
from jax.example_libraries import optimizers as jax_opt

from numpy import Inf, NaN

from utils.rnn_utils import update
from utils.rnn_utils import rnn_params
from utils.rnn_utils import initialize_wout
from utils.rnn_utils import compute_conceptor

from utils.utils import setup_logging_directory
from utils.utils import visualize_sine_interpolation

from torch.utils.tensorboard import SummaryWriter

# define flags
FLAGS = flags.FLAGS

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
flags.DEFINE_integer("rnn_size", 512, "number of hidden units")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 42, "seed for random number generators")
flags.DEFINE_float("beta_1", 0.02, "conceptor loss amplitude")
flags.DEFINE_float("beta_2", 0.01, "conceptor loss amplitude")
flags.DEFINE_float("aperture", 10, "aperture of the conceptor")


def main(_):

    t_pattern = 300
    datasets = jax.vmap(sine_wave, in_axes=(None, 0))(
        t_pattern, np.linspace(1, 3, 10))
    datasets = np.expand_dims(datasets, axis=2)

    data_sample = np.array([0, -1])
    ut_train = datasets[data_sample, 0:datasets.shape[1]-1, :]
    yt_train = datasets[data_sample, 1:datasets.shape[1], :]

    log_folder = setup_logging_directory(FLAGS.logdir, FLAGS.name)
    tb_writer = SummaryWriter(log_dir=log_folder)
    os.makedirs(os.path.join(log_folder, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'plots'), exist_ok=True)

    with open(os.path.join(log_folder, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    input_size = ut_train.shape[-1]
    output_size = yt_train.shape[-1]
    params_ini = rnn_params(512,input_size,output_size,1., 1., 0.1, 0.8, seed=21)

    params_rnn, _, _ = initialize_wout(
        params_ini.copy(), ut_train, yt_train, reg_wout=10)

    optimizer = optax.chain(
        optax.clip(FLAGS.clip_grad),
        optax.adam(learning_rate=FLAGS.learning_rate)
    )

    opt_state = optimizer.init(params_rnn)
    opt_update = optimizer.update

    for epoch_idx in tqdm(range(FLAGS.num_epochs)):

        params_rnn, opt_state, X, info = update(params_rnn,
                                                ut_train,
                                                yt_train,
                                                opt_state,
                                                opt_update,
                                                aperture=FLAGS.aperture,
                                                washout=FLAGS.washout,
                                                beta_1=FLAGS.beta_1,
                                                beta_2=FLAGS.beta_2
                                                )

        # log losses to tensorboard
        tb_writer.add_scalar("loss", info['loss'].item(), epoch_idx)
        tb_writer.add_scalar("loss_c", info['err_c'].item(), epoch_idx)
        tb_writer.add_scalar("loss_c_mean", info['err_c_mean'].item(), epoch_idx)
        tb_writer.add_scalar("loss_rec", np.mean(info['err_mse']).item(), epoch_idx)
        tb_writer.add_scalar("grads_norm", info['grads_norm'][0].item(), epoch_idx)

        if epoch_idx % FLAGS.steps_per_eval == 0:
            C = jax.vmap(lambda x: compute_conceptor(x, FLAGS.aperture,svd=True))(X[:,FLAGS.washout:,:])

            visualize_sine_interpolation(params_rnn, 
                                         C,
                                        log_folder,
                                        f"{epoch_idx:03}")

            # save params
            np.savez(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", **{
                key: params_rnn[key].__array__() for key in params_rnn.keys()
            })

            conceptor = {"C_1": C[0], "C_2": C[1]}
            # save params
            np.savez(f"{log_folder}/ckpt/conceptor_{epoch_idx+1:03}.npz", **{
                key: np.array(conceptor[key]) for key in conceptor.keys()
            })


if __name__ == "__main__":
    app.run(main)
