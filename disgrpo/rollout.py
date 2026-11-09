import asyncio

import numpy as np
import ray

from disgrpo.batch import RolloutBatch
from disgrpo.buffer import SharedMemoryRingBuffer


def build_batch(outputs, config, reward_fn):
    records = []
    for group_id, request in enumerate(outputs):
        prompt_ids = list(request.prompt_token_ids or [])[-config.max_prompt_length:]
        for completion in request.outputs:
            response_ids = list(completion.token_ids[:config.max_completion_length])
            if not response_ids:
                continue
            if completion.logprobs is None:
                raise RuntimeError("vLLM must return token log probabilities")
            old_logprobs = [
                float(step[token_id].logprob)
                for step, token_id in zip(completion.logprobs, response_ids)
            ]
            reward = float(reward_fn(request.prompt, completion.text))
            records.append((prompt_ids, response_ids, old_logprobs, reward, group_id))

    if not records:
        raise RuntimeError("Rollout produced no completion tokens")

    width = max(len(prompt) + len(response) for prompt, response, _, _, _ in records)
    tokens = np.zeros((len(records), width), dtype=np.int32)
    attention_mask = np.zeros((len(records), width), dtype=np.uint8)
    response_mask = np.zeros((len(records), width), dtype=np.uint8)
    old_logprobs = np.zeros((len(records), width), dtype=np.float32)
    rewards = np.zeros((len(records),), dtype=np.float32)
    group_ids = np.zeros((len(records),), dtype=np.int32)

    for row, (prompt, response, logprobs, reward, group_id) in enumerate(records):
        prompt_length = len(prompt)
        response_length = len(response)
        total_length = prompt_length + response_length
        tokens[row, :prompt_length] = prompt
        tokens[row, prompt_length:total_length] = response
        attention_mask[row, :total_length] = 1
        response_mask[row, prompt_length:total_length] = 1
        old_logprobs[row, prompt_length:total_length] = logprobs
        rewards[row] = reward
        group_ids[row] = group_id

    return RolloutBatch(
        tokens=tokens,
        attention_mask=attention_mask,
        response_mask=response_mask,
        old_logprobs=old_logprobs,
        rewards=rewards,
        group_ids=group_ids,
    )


@ray.remote
class VLLMRolloutWorker:
    def __init__(self, config, buffer_name):
        from vllm import LLM, SamplingParams
        from disgrpo.policy import load_policy

        del load_policy
        self.config = config
        self.buffer = SharedMemoryRingBuffer.attach(buffer_name)
        module_name, function_name = config.reward_module.split(":", 1)
        import importlib

        self.reward_fn = getattr(importlib.import_module(module_name), function_name)
        self.engine = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        self.sampling = SamplingParams(
            n=config.generations,
            temperature=config.temperature,
            max_tokens=config.max_completion_length,
            logprobs=1,
        )

    def generate(self, prompts):
        outputs = self.engine.generate(prompts, self.sampling, use_tqdm=False)
        batch = build_batch(outputs, self.config, self.reward_fn)
        asyncio.run(self.buffer.put(batch.encode()))
        return len(batch.rewards)
