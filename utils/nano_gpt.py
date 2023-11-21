"""
Source: https://github.com/cgarciae/nanoGPT-jax/blob/master/model.py
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Modifications
- removed pretraining code
- removed embedding projection
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.training import train_state


@dataclass
class GPTConfig:
    block_size: int = 1024
    # vocab_size: int = 50257  # RM word embeddings
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    input_dim: int = 1


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        # head_size = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(config.n_embd * 3)
        # output projection
        self.c_proj = nn.Dense(config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch, move head forward to be the batch dim
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(
            1, 2
        )  # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(
            1, 2
        )  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(
            1, 2
        )  # (B, nh, T, hs)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float("-inf"), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(
            B, T, C
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.c_fc = nn.Dense(4 * config.n_embd)
        self.c_proj = nn.Dense(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(config)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        # assert config.vocab_size is not None  # RM word embeddings
        assert config.block_size is not None

        # self.wte = nn.Embed(config.vocab_size, config.n_embd)  # RM word embeddings
        self.wte_raw = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.dense_out = nn.Dense(config.input_dim)

    def __call__(
        self, x: jax.Array, *, train: bool, targets: Optional[jax.Array] = None
    ):
        """
        x: inputs of shape (b, t, features)
        targets: targets of shape (b, t, features)
        """
        b, t, f = x.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32)[None]  # shape (1, t)

        # embeddings
        # tok_emb = self.wte(x)  # token embeddings of shape (b, t, n_embd)  # RM word embeddings
        tok_emb = self.wte_raw(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)

        # forward the GPT model itself
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)

        # logits = self.wte.attend(x)
        # TODO: check if this is correct - need to apply transposed wte_raw matrix?
        logits = self.dense_out(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            loss = optax.l2_loss(logits, targets).mean()
        else:
            loss = None

        return logits, loss

    def crop_block_size(self, params, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model

        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        def crop_weights(path: Tuple[str, ...], x):
            if path[-2:] == ("wpe", "embedding"):
                return x[:block_size]
            return x

        return freeze(path_aware_map(crop_weights, params))

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will
        experience weight decay for regularization and those that won't (biases, and layernorm/
        embedding weights). We are then returning the optax optimizer object.
        """

        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                weight_decay=decay,
            )

        def partition_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ("bias", "scale", "embedding"):
                return "no_decay"
            elif path[-1] in ("kernel",):
                return "decay"
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        partition_optimizers = {
            "decay": get_optimizer(weight_decay),
            "no_decay": get_optimizer(0.0),
        }
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        tx = optax.adamw(
            learning_rate=learning_rate,
            b1=betas[0],
            b2=betas[1],
            weight_decay=0.0,
        )

        return tx

    def generate(
        self, key, params, input_tokens, max_new_tokens, temperature=1.0, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)

        # tokens index -> tokens None
        def scan_f(tokens, i):
            # l: x y
            # t: a b - -
            # i: 0 1 2 3
            step_key = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size
            #   else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.apply({"params": params}, tokens, train=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            # sample from the distribution
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(
                    logits, min(top_k, logits.shape[-1])
                )
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(
                    top_tokens, token_idx[:, None], axis=-1
                ).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
                # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
            # append sampled index to the running sequence and continue
            tokens = tokens.at[:, i].set(next_token)

            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, indexes)

        return tokens

    def create_state(
        self,
        learning_rate,
        weight_decay,
        beta1,
        beta2,
        decay_lr=None,
        warmup_iters=None,
        lr_decay_iters=None,
        min_lr=None,
        params=None,
        **kwargs,
    ):
        if params is None:
            variables = self.init(
                # jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32), train=False
                jax.random.PRNGKey(0), jnp.empty((1, 1, 1), dtype=jnp.float32), train=False
            )
            params = variables["params"]
        if decay_lr:
            assert (
                warmup_iters is not None
                and lr_decay_iters is not None
                and min_lr is not None
            )
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters,
                decay_steps=lr_decay_iters,
                end_value=min_lr,
            )
        else:
            lr_schedule = learning_rate
        tx = self.configure_optimizers(
            params,
            weight_decay=weight_decay,
            learning_rate=lr_schedule,
            betas=(beta1, beta2),
        )
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)
