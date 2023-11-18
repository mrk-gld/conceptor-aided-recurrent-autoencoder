# import jax
# import torch
from utils.transformer import TrainerModuleRegression
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax.random as random
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, num_samples, sequence_length, freq_range):
        super().__init__()
        X = []
        y = []
        for _ in range(num_samples):
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(0, 2 * np.pi)
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


num_samples = 1000
sequence_length = 300
freq_range = (0.001, 0.1)
dataset = Dataset(num_samples, sequence_length, freq_range)
train_loader = data.DataLoader(
    dataset, batch_size=100, shuffle=False, collate_fn=numpy_collate
)
val_ds = Dataset(num_samples // 10, sequence_length, freq_range)
val_loader = data.DataLoader(
    val_ds, batch_size=100, shuffle=False, collate_fn=numpy_collate
)


print("x shape:", dataset[0][0].shape)
print("y shape:", dataset[0][0].shape)

# fig, ax = plt.subplots(figsize=(8, 6))
# x, y = dataset[0]  # Get the first sample from the dataset
# T = 20
# ax.plot(x[:T], label='x')
# ax.plot(y[:T], label='y')
# ax.set_xlabel("Time")
# ax.set_ylabel("Amplitude")
# ax.legend()
# plt.show()

# fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# for i, ax in enumerate(axs.flat):
#     x, _ = dataset[i]
#     ax.plot(x)
#     ax.set_title(f"Sine Wave {i+1}")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Amplitude")
# plt.tight_layout()
# plt.show()


class Trainer(TrainerModuleRegression):
    def batch_to_input(self, batch):
        inp_data, _ = batch
        # inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
        return inp_data

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            inp_data, y_true = batch
            # inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
            rng, dropout_apply_rng = random.split(rng)
            y_pred = self.model.apply(
                {"params": params},
                inp_data,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            # loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            loss = optax.l2_loss(y_pred, y_true).mean()
            return loss, rng

        return calculate_loss


trainer = Trainer(
    model_name="example_transformer",
    exmp_batch=next(iter(train_loader)),
    max_iters=1_000,
    lr=1e-3,
    warmup=100,
    seed=42,
    checkpoint_path="./logs",
    model_dim=256,
    input_dim=1,
    num_heads=4,
    num_layers=1,
    dropout_prob=0.0,
    input_dropout_prob=0.0,
)

mean_loss = trainer.eval_model(train_loader)
print(mean_loss)

trainer.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=2
)

mean_loss = trainer.eval_model(train_loader)
print(mean_loss)
