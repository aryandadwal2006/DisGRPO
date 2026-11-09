# DisGRPO
DisGRPO runs vLLM rollouts and JAX policy updates.
Ray starts rollout workers.
Each prompt produces a group of responses.
The reward function scores each response.
The buffer stores token IDs, masks, rewards, and group IDs.
The learner reads samples from shared memory.
GRPO normalizes rewards within each prompt group.
JAX computes the clipped policy loss.
JAX distributed synchronizes gradients.
NCCL handles GPU collectives.
Docker builds the runtime image.
Kubernetes starts the worker job.
Set MODEL, PROMPTS, POLICY_ADAPTER, and REWARD_MODULE before launch.
Run `python -m disgrpo.train`.
