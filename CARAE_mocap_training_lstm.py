import os
from functools import partial
from tqdm import tqdm
from absl import flags, app
import optax
import flax.linen as nn
import jax
import jax.numpy as np

from torch.utils.tensorboard import SummaryWriter

from utils.lstm_utils import update, compute_conceptor

# from utils.rnn_utils import rnn_params
# from utils.rnn_utils import initialize_wout

from utils.utils import setup_logging_directory
from utils.utils import visualize_mocap_interpolation

from utils.mocap_utils import get_mocap_data

# define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "mocap_interp", "name of this training run/experiment")
flags.DEFINE_string("logdir", "./logs", "path to the log directory")
flags.DEFINE_integer("num_epochs", 3050, "number of training epochs")
flags.DEFINE_integer("steps_per_eval", 100, "number of training steps per evaluation")
flags.DEFINE_integer("washout", 0, "washout period")
flags.DEFINE_integer("rnn_size", 512, "number of hidden units")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 42, "seed for random number generators")
flags.DEFINE_float("beta_1", 0.02, "conceptor loss amplitude")
flags.DEFINE_float("beta_2", 0.1, "conceptor loss amplitude")
flags.DEFINE_float("aperture", 10, "aperture of the conceptor")


def main(_):
    folder = "motion_data_numpy/"
    dataset_names = ["walk_15", "run_55"]
    datasets, _, _, _ = get_mocap_data(folder, dataset_names)

    data_sample = np.array([0, -1])
    ut_train = datasets[data_sample, 0:datasets.shape[1] - 1, :]
    yt_train = datasets[data_sample, 1:datasets.shape[1], :]

    log_folder = setup_logging_directory(FLAGS.logdir, FLAGS.name)
    tb_writer = SummaryWriter(log_dir=log_folder)
    os.makedirs(os.path.join(log_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, "plots"), exist_ok=True)

    with open(os.path.join(log_folder, "flags.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    input_size = ut_train.shape[-1]
    output_size = yt_train.shape[-1]
    hidden_size = FLAGS.rnn_size

    # TODO: implement leak rate?
    # TODO: implement spectral radius scaling?
    # NOTE: bias_scaling for LSTM ignored (was 0.8)
    # NOTE: inp_scaling ignored (used 1.0 anyway)
    # rhoW = 1.0
    # inp_scaling = 1.0
    a_dt = 0.1
    bias_scaling = 0.8

    key = jax.random.PRNGKey(0)

    lstm = nn.LSTMCell(hidden_size)
    carry = lstm.initialize_carry(key, (input_size,))
    lstm_params = lstm.init(key, carry, np.zeros((input_size,)))
    params = dict(
        lstm=lstm_params,
        wout=jax.random.normal(key, shape=(output_size, hidden_size)),
        bias_out=jax.random.normal(key, shape=(output_size,)) * bias_scaling,
        a_dt=a_dt * np.ones(hidden_size),
        x_ini=0.1 * jax.random.normal(key, shape=(hidden_size,)),
    )

    # params_ini = rnn_params(FLAGS.rnn_size, input_size,
    #                         output_size, 1., 1., 0.1, 0.8, seed=21)
    # params_rnn, _, _ = initialize_wout(
    #     params_ini.copy(), ut_train, yt_train, reg_wout=10)

    optimizer = optax.chain(
        optax.clip(FLAGS.clip_grad), optax.adam(learning_rate=FLAGS.learning_rate)
    )

    opt_state = optimizer.init(params)
    opt_update = optimizer.update

    for epoch_idx in tqdm(range(FLAGS.num_epochs)):
        params, opt_state, X, info = update(
            params,
            ut_train,
            yt_train,
            opt_state,
            opt_update,
            aperture=FLAGS.aperture,
            washout=FLAGS.washout,
            beta_1=FLAGS.beta_1,
            beta_2=FLAGS.beta_2,
        )

        # log losses to tensorboard
        tb_writer.add_scalar("loss", info["loss"].item(), epoch_idx)
        tb_writer.add_scalar("loss_c", info["err_c"].item(), epoch_idx)
        tb_writer.add_scalar("loss_c_mean", info["err_c_mean"].item(), epoch_idx)
        tb_writer.add_scalar("loss_rec", np.mean(info["err_mse"]).item(), epoch_idx)
        tb_writer.add_scalar("grads_norm", info["grads_norm"][0].item(), epoch_idx)

        if epoch_idx % FLAGS.steps_per_eval == 0:
            f_partial = partial(compute_conceptor, aperture=FLAGS.aperture, svd=True)
            C = jax.vmap(f_partial)(X[:, FLAGS.washout:, :])

            visualize_mocap_interpolation(params, C, log_folder, f"{epoch_idx:03}", ntype='lstm')

            # save params
            np.save(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", params)
            # np.savez(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", **{
            #     key: params[key].__array__() for key in params.keys()
            # })
            conceptor = {"C_1": C[0], "C_2": C[1]}
            np.savez(
                f"{log_folder}/ckpt/conceptor_{epoch_idx+1:03}.npz",
                **{key: np.array(conceptor[key]) for key in conceptor.keys()},
            )


if __name__ == "__main__":
    app.run(main)
