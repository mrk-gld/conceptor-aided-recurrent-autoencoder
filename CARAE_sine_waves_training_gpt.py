import os
from functools import partial
from tqdm import tqdm
from absl import flags, app
import jax
import jax.numpy as np

from utils.rnn_utils import compute_conceptor
# from utils.rnn_utils import forward_rnn_interp, initialize_wout, rnn_params, update

from utils.utils import setup_logging_directory
# from utils.utils import visualize_sine_interpolation, compute_JS_divergence_and_acf
from torch.utils.tensorboard import SummaryWriter

from utils.nano_gpt import GPTConfig, GPT

import matplotlib.pyplot as plt

# define flags
FLAGS = flags.FLAGS


@partial(jax.jit, static_argnames=('train',))
def forward(state, batch, *, train: bool):
    x, y = batch
    rngs = {'dropout': jax.random.fold_in(jax.random.PRNGKey(0), state.step)}
    variables = {'params': state.params}
    y_pred, loss = state.apply_fn(variables, x, train=train, targets=y, rngs=rngs)
    return y_pred, loss


@partial(jax.jit, static_argnames=('train', 'aperture', 'beta_1', 'beta_2', 'conceptor_layers'))
def forward_conceptor(state, batch, *, train: bool, aperture, beta_1, beta_2, conceptor_layers):
    x, y = batch
    rngs = {'dropout': jax.random.fold_in(jax.random.PRNGKey(0), state.step)}
    variables = {'params': state.params}
    y_pred, loss, info = state.apply_fn(
        variables, x, train=train, targets=y, rngs=rngs, conceptor_loss=True,
        aperture=aperture, beta_1=beta_1, beta_2=beta_2, conceptor_layers=conceptor_layers
    )
    return y_pred, loss, info


@partial(jax.jit, static_argnames=('train', 'conceptor_layers'))
def forward_conceptor_interpolation(state, batch, *, train: bool, conceptors, ratios,
                                    conceptor_layers):
    """
    conceptors: two conceptors of shape (2, t, features)
                where the first one is the conceptor of the first input (min. freq.)
                and the second one is the conceptor of the second input (max. freq.)
    ratio: ratio of the interpolation between the two conceptors
           if ratio = 0, the conceptor of the first input is used (min. freq.)
           if ratio = 1, the conceptor of the second input is used (max. freq.)
    """
    assert not train, "forward with conceptor interpolation only works in eval mode"
    x, y = batch
    rngs = {'dropout': jax.random.fold_in(jax.random.PRNGKey(0), state.step)}
    variables = {'params': state.params}
    y_pred, loss = state.apply_fn(
        variables, x, train=train, targets=y, rngs=rngs,
        conceptor_interpolation=True, conceptors=conceptors, ratios=ratios,
        conceptor_layers=conceptor_layers
    )
    return y_pred, loss


@partial(jax.jit, donate_argnames=('state',))
def train_step(state, batch):
    def loss_fn(params):
        state_ = state.replace(params=params)
        _, loss = forward(state_, batch, train=True)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state


@partial(jax.jit, donate_argnames=('state',), static_argnames=('aperture', 'beta_1', 'beta_2', 'conceptor_layers'))
def train_step_conceptor(state, batch, aperture, beta_1, beta_2, conceptor_layers):
    def loss_fn(params):
        state_ = state.replace(params=params)
        _, loss, ret = forward_conceptor(state_, batch, train=True, aperture=aperture,
                                         beta_1=beta_1, beta_2=beta_2,
                                         conceptor_layers=conceptor_layers)
        return loss, ret
    (loss, ret), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state, ret


def estimate_loss(state, train_loader, val_loader, eval_iters=10):
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k, batch in enumerate(train_loader if split == 'train' else val_loader):
            if k >= eval_iters:
                break
            _, loss = forward(state, batch, train=False)
            losses.append(loss)
        out[split] = np.array(losses).mean()
    return out


def sine_wave(t_pattern, s):
    """linear interpolation between sine waves of different frequencies
    (parametrisation of similar to Wyffels et al. (2014))"""
    signal = []
    for i in range(t_pattern):
        signal.append(np.sin(0.075 * s * i))
    return np.array(signal)


flags.DEFINE_string("name", "sine_wave_interp", "name of this training run/experiment")
flags.DEFINE_string("logdir", "./logs", "path to the log directory")
flags.DEFINE_integer("num_epochs", 5000, "number of training epochs")
# flags.DEFINE_integer("num_epochs", 3050, "number of training epochs")
flags.DEFINE_integer("steps_per_eval", 100, "number of training steps per evaluation")
flags.DEFINE_integer("washout", 0, "washout period")
flags.DEFINE_integer("rnn_size", 512, "number of hidden units")

flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("clip_grad", 1e-2, "gradient clipping norm value")

flags.DEFINE_integer("seed", 42, "seed for random number generators")
flags.DEFINE_float("beta_1", 0.02, "conceptor loss amplitude")
flags.DEFINE_float("beta_2", 0.01, "conceptor loss amplitude")
flags.DEFINE_float("aperture", 10, "aperture of the conceptor")

flags.DEFINE_bool("plot_interp", True, "plot interpolation between sine waves")
flags.DEFINE_bool("calc_metric", True, "calculate metric for interpolation")
flags.DEFINE_bool("save_param", False, "save parameters")

flags.DEFINE_integer("len_cueing", 20, "length of cueing period")
flags.DEFINE_integer("n_heads", 4, "number of attention heads")
flags.DEFINE_integer("n_layers", 3, "number of transformer layers")
# flags.DEFINE_integer("n_layers", 1, "number of transformer layers")
flags.DEFINE_integer("n_embd", 128, "embedding dimension")

flags.DEFINE_boolean("conceptor_loss", True, "use conceptor loss")
flags.DEFINE_list("conceptor_layers", [0], "layers for conceptor loss and interpolation")


def setup_model(t_pattern=300):
    # initialize GPT model
    config = GPTConfig(
        block_size=t_pattern,  # max sequence length (was 512)
        n_layer=FLAGS.n_layers,
        n_head=FLAGS.n_heads,  # was 8
        n_embd=FLAGS.n_embd,  # embedding dimension (was 512)
        dropout=0.,
        input_dim=1,
        # vocab_size=50257,  # vocab size (not used here)
    )
    model = GPT(config)
    state = model.create_state(
        learning_rate=6e-4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        decay_lr=True,
        warmup_iters=FLAGS.num_epochs // 10,  # was 2000
        lr_decay_iters=FLAGS.num_epochs,
        min_lr=1e-5,  # was 6e-5
        params=None,
    )
    return config, model, state


def main(_):
    conceptor_layers = tuple(map(int, FLAGS.conceptor_layers))

    # setup dataset: 2 training sine waves, 8 testing sine waves
    t_pattern = 300
    datasets = jax.vmap(sine_wave, in_axes=(None, 0))(t_pattern, np.linspace(1, 3, 10))
    datasets = np.expand_dims(datasets, axis=2)
    data_sample = np.array([0, -1])
    ut_train = datasets[data_sample, 0:datasets.shape[1] - 1, :]
    yt_train = datasets[data_sample, 1:datasets.shape[1], :]
    ut_test = datasets[1:-1, 0:datasets.shape[1] - 1, :]
    yt_test = datasets[1:-1, 1:datasets.shape[1], :]

    log_folder = setup_logging_directory(FLAGS.logdir, FLAGS.name)
    tb_writer = SummaryWriter(log_dir=log_folder)
    os.makedirs(os.path.join(log_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, "plots"), exist_ok=True)

    with open(os.path.join(log_folder, "flags.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    config, model, state = setup_model(t_pattern=t_pattern)

    for epoch_idx in (pbar := tqdm(range(FLAGS.num_epochs))):
        pbar.set_description("Training")
        shuffled_idxs = jax.random.permutation(jax.random.PRNGKey(epoch_idx), np.arange(2))
        batch = (ut_train[shuffled_idxs], yt_train[shuffled_idxs])

        ############################
        # training
        ############################

        if FLAGS.conceptor_loss:
            loss, state, info = train_step_conceptor(
                state, batch, aperture=FLAGS.aperture, beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_2,
                conceptor_layers=conceptor_layers
            )
            loss_c, loss_y, err_c, err_c_mean, X = info
            # C = compute_conceptor(X, FLAGS.aperture)
            tb_writer.add_scalar("loss", loss.item(), epoch_idx)
            tb_writer.add_scalar("loss_c", loss_c.item(), epoch_idx)
            tb_writer.add_scalar("loss_y", loss_y.item(), epoch_idx)
            tb_writer.add_scalar("loss_c_wo_mean", err_c.item(), epoch_idx)
            tb_writer.add_scalar("loss_c_mean", err_c_mean.item(), epoch_idx)
            # NOTE: get grad norms?

        else:
            loss, state = train_step(state, batch)
            tb_writer.add_scalar("loss", loss.item(), epoch_idx)

        ############################
        # evaluation
        ############################

        if epoch_idx % FLAGS.steps_per_eval == 0:
            # evaluate on train and test (loss only)
            conceptor_manifold = jax.vmap(lambda x: compute_conceptor(x, FLAGS.aperture))(X)
            if FLAGS.conceptor_loss:
                # compute train loss with the conceptor plugged in
                _, train_loss = forward_conceptor_interpolation(
                    state, batch, train=False, conceptors=conceptor_manifold,
                    ratios=np.array([0., 1.]), conceptor_layers=conceptor_layers
                )
                # compute test loss with the conceptor plugged in
                _, test_loss = forward_conceptor_interpolation(
                    state, (ut_test, yt_test), train=False, conceptors=conceptor_manifold,
                    ratios=np.linspace(0, 1, 10)[1:-1], conceptor_layers=conceptor_layers
                )
            else:
                _, train_loss = forward(state, batch, train=False)
                _, test_loss = forward(state, (ut_test, yt_test), train=False)

            tb_writer.add_scalar("train_loss", train_loss.item(), epoch_idx)
            tb_writer.add_scalar("test_loss", test_loss.item(), epoch_idx)

            # generative mode for unseen testing sine wave frequencies
            gen_key = jax.random.PRNGKey(epoch_idx)
            pred = model.generate(
                key=gen_key,
                params=state.params,
                x=ut_test[:, :FLAGS.len_cueing, :],
                max_new_tokens=config.block_size-FLAGS.len_cueing,
                conceptors=conceptor_manifold,
                ratios=np.linspace(0, 1, 10)[1:-1],
                conceptor_layers=conceptor_layers,
            )
            auto_loss_test = np.mean(
                (pred[:, FLAGS.len_cueing:-1, :] - ut_test[:, FLAGS.len_cueing:, :]) ** 2
            )
            tb_writer.add_scalar("auto_loss_test", auto_loss_test.item(), epoch_idx)

            # generative mode (on training)
            gen_key = jax.random.PRNGKey(epoch_idx)
            pred = model.generate(
                key=gen_key,
                params=state.params,
                x=ut_train[:, :FLAGS.len_cueing, :],
                max_new_tokens=config.block_size-FLAGS.len_cueing,
                conceptors=conceptor_manifold,
                ratios=np.array([0., 1.]),
                conceptor_layers=conceptor_layers,
            )
            auto_loss = np.mean(
                (pred[:, FLAGS.len_cueing:-1, :] - ut_train[:, FLAGS.len_cueing:, :]) ** 2
            )
            tb_writer.add_scalar("auto_loss", auto_loss.item(), epoch_idx)

            # plot generative mode output
            fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10, 5), dpi=300)
            for i in range(2):
                axs[i].plot(np.arange(t_pattern), pred[i, :, 0], label="generated")
                axs[i].plot(np.arange(t_pattern-1), ut_train[i, :, 0], label="cueing")
                axs[i].vlines(FLAGS.len_cueing, -1, 1, color="k", linestyle="--", label="cue end")
                axs[i].legend()
            fig.savefig(f"{log_folder}/plots/generative_{epoch_idx:03}.png")
            plt.close(fig)

            # save parameters
            np.save(f"{log_folder}/ckpt/params_{epoch_idx+1:03}.npy", state.params)

            # save conceptor
            C = jax.vmap(lambda x: compute_conceptor(x, FLAGS.aperture))(X)
            conceptor = {"C_1": C[0], "C_2": C[1]}
            np.savez(
                f"{log_folder}/ckpt/conceptor_{epoch_idx+1:03}.npz",
                **{key: np.array(conceptor[key]) for key in conceptor.keys()},
            )

            # update progress bar
            pbar.set_postfix_str(f"loss train={train_loss:.2f}, test={test_loss:.2f}")


if __name__ == "__main__":
    app.run(main)
