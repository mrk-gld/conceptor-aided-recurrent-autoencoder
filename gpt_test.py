from functools import partial
from utils.nano_gpt import GPTConfig, GPT
import jax
import jax.numpy as np
import torch.utils.data as data
from tqdm.auto import tqdm


class Dataset(data.Dataset):
    def __init__(self, key, num_samples, sequence_length, freq_range):
        super().__init__()
        X = []
        y = []
        for _ in range(num_samples):
            fmin, fmax = freq_range
            freq = jax.random.uniform(key, minval=fmin, maxval=fmax)
            # freq = np.random.uniform(*freq_range)
            phase = jax.random.uniform(key, minval=0, maxval=2*np.pi)
            # phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(0, sequence_length + 1)
            sine_wave = np.sin(2 * np.pi * freq * t + phase)
            # delayed_wave = np.roll(sine_wave, -1)
            X.append(sine_wave[:-1])
            y.append(sine_wave[1:])
        self.X = np.array(X)
        self.y = np.array(y)
        assert self.X.shape[0] == num_samples == self.y.shape[0]

    def __len__(self):
        return num_samples

    def __getitem__(self, idx):
        # return shape [seq_len, feature_dim=1]
        return self.X[idx][:, None], self.y[idx][:, None]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


@partial(jax.jit, static_argnames=('train',))
def forward(state, batch, *, train: bool):
    x, y = batch
    rngs = {'dropout': jax.random.fold_in(jax.random.PRNGKey(0), state.step)}
    y_pred, loss = state.apply_fn({'params': state.params}, x, train=train, targets=y, rngs=rngs)
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


def estimate_loss(train_loader, val_loader, eval_iters=10):
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


# setup dataloaders

batch_size = 128
shuffle = True
num_samples = 1000
sequence_length = 300
freq_range = (0.001, 0.1)
ds_key_tr, ds_key_val = jax.random.split(jax.random.PRNGKey(0), 2)
dataset = Dataset(ds_key_tr, num_samples, sequence_length, freq_range)
train_loader = data.DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate
)
val_ds = Dataset(ds_key_val, num_samples // 10, sequence_length, freq_range)
val_loader = data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate
)
print("x shape:", dataset[0][0].shape)
print("y shape:", dataset[0][1].shape)

# setup GPT model

config = GPTConfig(
    block_size=512,  # max sequence length
    n_layer=1,
    n_head=8,
    n_embd=512,  # embedding dimension (projection of input)
    dropout=0.,
    input_dim=1,
    # vocab_size=50257,  # vocab size (not used here)
)
model = GPT(config)

state = model.create_state(
    learning_rate=6e-4,
    weight_decay=1e-2,
    beta1=0.9,
    beta2=0.95,
    decay_lr=True,
    warmup_iters=200,  # 2000
    lr_decay_iters=600_000,
    min_lr=6e-5,
    params=None,
)
params = state.params


for epoch_idx in range(n_epochs := 10):
    losses = []
    for idx, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader))):
        pbar.set_description(f"Epoch {epoch_idx+1}")
        loss, state = train_step(state, (batch))
        losses.append(loss)
        pbar.set_postfix_str(f"loss={loss:.2f}, mean_loss={np.array(losses).mean():.2f}")
    # evaluate
    eval_iters = 2
    losses = estimate_loss(train_loader, val_loader, eval_iters=eval_iters)
    print('\teval: train loss: {:.2f}, val loss: {:.2f}'.format(losses['train'], losses['val']))


##############################################################################
# output from run:
##############################################################################
# x shape: (300, 1)
# y shape: (300, 1)
# Epoch 1: 100%|█████████████████████| 8/8 [00:30<00:00,  3.78s/it, loss=0.43, mean_loss=0.82]
#         eval: train loss: 0.61, val loss: 0.91
# Epoch 2: 100%|█████████████████████| 8/8 [00:26<00:00,  3.29s/it, loss=0.28, mean_loss=0.41]
#         eval: train loss: 0.39, val loss: 0.36
# Epoch 3: 100%|█████████████████████| 8/8 [00:30<00:00,  3.80s/it, loss=0.28, mean_loss=0.26]
#         eval: train loss: 0.17, val loss: 0.11
# Epoch 4: 100%|█████████████████████| 8/8 [00:30<00:00,  3.87s/it, loss=0.06, mean_loss=0.12]
#         eval: train loss: 0.10, val loss: 0.09
# Epoch 5: 100%|█████████████████████| 8/8 [00:30<00:00,  3.83s/it, loss=0.06, mean_loss=0.09]
#         eval: train loss: 0.04, val loss: 0.05
# Epoch 6: 100%|█████████████████████| 8/8 [00:29<00:00,  3.74s/it, loss=0.04, mean_loss=0.05]
#         eval: train loss: 0.04, val loss: 0.03 - NOTE: lowest val loss, starts to increase after
# Epoch 7: 100%|█████████████████████| 8/8 [00:30<00:00,  3.76s/it, loss=0.03, mean_loss=0.04]
#         eval: train loss: 0.04, val loss: 0.05
# Epoch 8: 100%|█████████████████████| 8/8 [00:30<00:00,  3.77s/it, loss=0.03, mean_loss=0.03]
#         eval: train loss: 0.03, val loss: 0.04
# Epoch 9: 100%|█████████████████████| 8/8 [00:30<00:00,  3.87s/it, loss=0.02, mean_loss=0.03]
#         eval: train loss: 0.02, val loss: 0.06
# Epoch 10: 100%|█████████████████████| 8/8 [00:29<00:00,  3.73s/it, loss=0.02, mean_loss=0.02]
#         eval: train loss: 0.02, val loss: 0.08
