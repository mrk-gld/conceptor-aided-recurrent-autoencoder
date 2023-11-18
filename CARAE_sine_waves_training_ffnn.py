import os
from functools import partial
from tqdm import tqdm
from absl import flags, app
import flax.linen as nn
import optax
import jax
import jax.numpy as np

from utils.ffnn_utils import update
from utils.ffnn_utils import SimpleDenseModel

from flax import linen as nn
# from utils.lstm_utils import rnn_params
# from utils.lstm_utils import initialize_wout
from utils.lstm_utils import compute_conceptor
from utils.ffnn_utils import forward_rnn_interp

from utils.utils import setup_logging_directory
from utils.utils import compute_JS_divergence_and_acf
from utils.ffnn_utils import visualize_sine_interpolation

from torch.utils.tensorboard import SummaryWriter

# define flags
FLAGS = flags.FLAGS


def sine_wave(t_pattern, s):
    """linear interpolation between sine waves of different frequencies
    (parametrisation of similar to Wyffels et al. (2014))"""
    signal = []
    for i in range(t_pattern):
        signal.append(np.sin(0.075 * s * i))
    return np.array(signal)


flags.DEFINE_string("name", "ffnn_sin_test", "name of this training run/experiment")
flags.DEFINE_string("logdir", "./logs", "path to the log directory")
flags.DEFINE_integer("num_epochs", 201, "number of training epochs")
flags.DEFINE_integer("steps_per_eval", 5, "number of training steps per evaluation")
flags.DEFINE_integer("washout", 0, "washout period")
flags.DEFINE_integer("rnn_size", 512, "number of hidden units")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 42, "seed for random number generators")
flags.DEFINE_float("beta_1", 0.02, "conceptor loss amplitude")
flags.DEFINE_float("beta_2", 0.002, "conceptor loss amplitude")
flags.DEFINE_float("aperture", 10, "aperture of the conceptor")

flags.DEFINE_bool("plot_interp", True, "plot interpolation between sine waves")
flags.DEFINE_bool("calc_metric", True, "calculate metric for interpolation")
flags.DEFINE_bool("save_param",False,"save parameters")

def main(_):
    t_pattern = 300
    datasets = jax.vmap(sine_wave, in_axes=(None, 0))(t_pattern, np.linspace(1, 3, 10))
    datasets = np.expand_dims(datasets, axis=2)

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
    hidden_size = 100
    input_size=20
    # generate input from ut_train that contains 2 dataset (shape (datasets, length=299,features=1)) and a moving window of size input_size
    dataset_input = []
    for dataset_idx in range(ut_train.shape[0]):
        X = []
        dataset = ut_train[dataset_idx]
        for i in range(dataset.shape[0] - input_size):
            X.append(dataset[i:i+input_size])
        X = np.array(X)
        dataset_input.append(X)
    ut_train = np.array(dataset_input).reshape(2,-1,input_size)
    yt_train = yt_train[:,input_size:,:]
    
    key = jax.random.PRNGKey(0)
    
    model = SimpleDenseModel()
    conceptor = np.eye(100)
    params = model.init(key, np.ones((1, input_size)), conceptor)
    
    bias_scaling = 0.1
    
    params = dict(
        ffnn=params,
        wout=jax.nn.initializers.xavier_normal()(key, (output_size, hidden_size)),
        bias_out=jax.nn.initializers.xavier_normal()(key, (output_size,1)).reshape(-1,) * bias_scaling,
    )

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
            C = jax.vmap(f_partial)(X)
            
            if FLAGS.calc_metric:
            
                lamda = 0.5
                len_seqs = 1000
        
                _, y_interp = forward_rnn_interp(params,
                                                    C,
                                                    ut_train,
                                                    ratio=lamda,
                                                    length=len_seqs,
                                                    spd_interp=None)
                
                js_div1, js_div2, acf1, acf2 = compute_JS_divergence_and_acf(ut_train[0],
                                                                            ut_train[1],
                                                                            y_interp,
                                                                            lag=30,
                                                                            bins=50
                                                                            )
                
                metric = 0.25 * (js_div1 + js_div2 + acf1 + acf2)
                tb_writer.add_scalar("metric", metric, epoch_idx)

            if FLAGS.plot_interp:
                visualize_sine_interpolation(params, C,ut_train, log_folder, f"{epoch_idx:03}", ntype='lstm')

            # save params
            # np.savez(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz", params)
            np.savez(
                f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npz",
                **{key: params[key].__array__() for key in params.keys()},
            )

            conceptor = {"C_1": C[0], "C_2": C[1]}
            # save conceptors
            np.savez(
                f"{log_folder}/ckpt/conceptor_{epoch_idx+1:03}.npz",
                **{key: np.array(conceptor[key]) for key in conceptor.keys()},
            )


if __name__ == "__main__":
    app.run(main)
