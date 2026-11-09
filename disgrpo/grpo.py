import jax
import jax.numpy as jnp


def group_advantages(rewards, group_ids, epsilon=1e-4):
    size = rewards.shape[0]
    counts = jax.ops.segment_sum(jnp.ones_like(rewards), group_ids, num_segments=size)
    sums = jax.ops.segment_sum(rewards, group_ids, num_segments=size)
    squares = jax.ops.segment_sum(jnp.square(rewards), group_ids, num_segments=size)
    means = sums / jnp.maximum(counts, 1.0)
    variances = jnp.maximum(squares / jnp.maximum(counts, 1.0) - jnp.square(means), 0.0)
    standard_deviations = jnp.sqrt(variances)
    return (rewards - means[group_ids]) / (standard_deviations[group_ids] + epsilon)


def clipped_policy_loss(
    apply_fn,
    params,
    tokens,
    attention_mask,
    response_mask,
    old_logprobs,
    rewards,
    group_ids,
    clip_epsilon,
):
    logits = apply_fn(params, tokens, attention_mask)
    log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = tokens[:, 1:]
    new_logprobs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    mask = response_mask[:, 1:].astype(jnp.float32)
    old = old_logprobs[:, 1:]
    advantages = group_advantages(rewards, group_ids)[:, None]
    ratios = jnp.exp(jnp.clip(new_logprobs - old, -20.0, 20.0))
    unclipped = ratios * advantages
    clipped = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    loss = -jnp.sum(jnp.minimum(unclipped, clipped) * mask) / denominator
    return loss, jnp.mean(rewards)
