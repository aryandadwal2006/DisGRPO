import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax.experimental import multihost_utils

from disgrpo.grpo import clipped_policy_loss


class JaxGRPOTrainer:
    def __init__(self, params, apply_fn, learning_rate, clip_epsilon):
        params = multihost_utils.broadcast_one_to_all(
            params,
            is_source=jax.process_index() == 0,
        )
        self.apply_fn = apply_fn
        self.clip_epsilon = clip_epsilon
        self.optimizer = optax.adam(learning_rate)
        self.params = jax.device_put_replicated(params, jax.local_devices())
        self.opt_state = jax.device_put_replicated(self.optimizer.init(params), jax.local_devices())

        def step(params, opt_state, tokens, attention_mask, response_mask, old_logprobs, rewards, group_ids):
            def objective(current_params):
                return clipped_policy_loss(
                    self.apply_fn,
                    current_params,
                    tokens,
                    attention_mask,
                    response_mask,
                    old_logprobs,
                    rewards,
                    group_ids,
                    self.clip_epsilon,
                )

            (loss, mean_reward), gradients = jax.value_and_grad(objective, has_aux=True)(params)
            gradients = jax.lax.pmean(gradients, axis_name="data")
            loss = jax.lax.pmean(loss, axis_name="data")
            mean_reward = jax.lax.pmean(mean_reward, axis_name="data")
            updates, opt_state = self.optimizer.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, mean_reward

        self._step = jax.pmap(step, axis_name="data")

    def update(self, batch):
        rows, width = batch.tokens.shape
        devices = jax.local_device_count()
        if rows % devices:
            raise ValueError("Rollout rows must divide across local JAX devices")
        group_ids, counts = np.unique(batch.group_ids, return_counts=True)
        if len(group_ids) % devices or len(set(counts.tolist())) != 1:
            raise ValueError("Prompt groups must be complete and divide across local JAX devices")
        per_device = rows // devices
        if per_device % counts[0]:
            raise ValueError("Prompt groups cannot be split across JAX devices")

        def shard(array):
            return jnp.asarray(array.reshape(devices, per_device, *array.shape[1:]))

        self.params, self.opt_state, losses, rewards = self._step(
            self.params,
            self.opt_state,
            shard(batch.tokens),
            shard(batch.attention_mask),
            shard(batch.response_mask),
            shard(batch.old_logprobs),
            shard(batch.rewards),
            shard(batch.group_ids),
        )
        losses.block_until_ready()
        return {
            "loss": float(np.asarray(losses[0])),
            "reward": float(np.asarray(rewards[0])),
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        params = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value[0])),
            self.params,
        )
        with open(os.path.join(directory, "params.pkl"), "wb") as handle:
            pickle.dump(params, handle, protocol=5)
