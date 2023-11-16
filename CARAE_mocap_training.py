import os
from tqdm import tqdm
from absl import flags, app
import optax

import jax
import jax.numpy as np

from torch.utils.tensorboard import SummaryWriter

from utils.rnn_utils import update
from utils.rnn_utils import rnn_params
from utils.rnn_utils import initialize_wout
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import forward_rnn_interp

from utils.utils import setup_logging_directory
from utils.utils import visualize_mocap_interpolation
from utils.utils import visualize_sine_interpolation
from utils.utils import compute_JS_divergence_and_acf


from utils.mocap_utils import get_mocap_data

# define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "mocap_interp",
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
flags.DEFINE_float("beta_2", 0.1, "conceptor loss amplitude")
flags.DEFINE_float("aperture", 10, "aperture of the conceptor")

flags.DEFINE_bool("plot_interp", True, "plot interpolation between mocap motions")
flags.DEFINE_bool("calc_metric", True, "calculate metric for interpolation")

def main(_):

    # load data
    folder = "motion_data_numpy/"
    dataset_names = ["walk_15", "run_55"]
    datasets, _, _, _ = get_mocap_data(folder, dataset_names)

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
    
    params_ini = rnn_params(FLAGS.rnn_size,
                            input_size,
                            output_size,
                            1., 1., 0.1, 0.8,
                            seed=FLAGS.seed)

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
            C = jax.vmap(lambda x: compute_conceptor(x, FLAGS.aperture, svd=True))(X[:,FLAGS.washout:,:])

            if FLAGS.calc_metric:
            
                lamda = 0.5
                len_seqs = 1000
        
                x_interp, y_interp = forward_rnn_interp(params_rnn,
                                                    C,
                                                    x_init=None,
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
                visualize_mocap_interpolation(params_rnn, 
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
