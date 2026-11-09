import asyncio
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from disgrpo.buffer import SharedMemoryRingBuffer
from disgrpo.batch import RolloutBatch
from disgrpo.config import Config
from disgrpo.policy import load_policy
from disgrpo.rollout import VLLMRolloutWorker
from disgrpo.trainer import JaxGRPOTrainer


def initialize_jax():
    process_count = int(os.getenv("JAX_PROCESS_COUNT", "1"))
    if process_count > 1:
        local_device_ids = os.getenv("JAX_LOCAL_DEVICE_IDS")
        options = {}
        if local_device_ids:
            options["local_device_ids"] = [int(value) for value in local_device_ids.split(",")]
        jax.distributed.initialize(
            coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
            num_processes=process_count,
            process_id=int(os.environ["JAX_PROCESS_ID"]),
            **options,
        )


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            values = [json.loads(line) for line in handle if line.strip()]
            prompts = [value["prompt"] for value in values]
        else:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError("Prompt file is empty")
    return prompts


def main():
    config = Config.from_env()
    initialize_jax()
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_prompts = load_prompts(config.prompts_path)[process_index::process_count]
    if not local_prompts:
        raise ValueError("No prompts assigned to this JAX process")
    if config.prompts_per_step % jax.local_device_count():
        raise ValueError("PROMPTS_PER_STEP must divide across local JAX devices")

    ray_address = os.getenv("RAY_ADDRESS") or None
    ray.init(address=ray_address, ignore_reinit_error=True)
    buffer = SharedMemoryRingBuffer.create(config.buffer_capacity, config.buffer_slot_size)
    worker = None

    try:
        node_id = ray.get_runtime_context().get_node_id()
        worker = VLLMRolloutWorker.options(
            num_cpus=2,
            num_gpus=config.tensor_parallel_size,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(config, buffer.name)
        params, apply_fn = load_policy(config.policy_adapter, config)
        trainer = JaxGRPOTrainer(
            params,
            apply_fn,
            config.learning_rate,
            config.clip_epsilon,
        )

        cursor = 0
        for step in range(config.train_steps):
            prompts = [
                local_prompts[(cursor + index) % len(local_prompts)]
                for index in range(config.prompts_per_step)
            ]
            cursor += config.prompts_per_step
            ray.get(worker.generate.remote(prompts))
            slot = asyncio.run(buffer.get())
            batch = RolloutBatch.from_payload(slot.payload)
            try:
                metrics = trainer.update(batch)
            finally:
                del batch
                asyncio.run(slot.release())
            print(
                f"process={process_index} step={step + 1} "
                f"loss={metrics['loss']:.6f} reward={metrics['reward']:.6f}",
                flush=True,
            )

        if process_index == 0:
            trainer.save(config.output_dir)
    finally:
        if worker is not None:
            ray.kill(worker)
        buffer.close()
        buffer.unlink()
        ray.shutdown()
        if process_count > 1:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
