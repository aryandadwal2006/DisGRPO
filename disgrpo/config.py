import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model: str
    prompts_path: str
    policy_adapter: str
    reward_module: str
    output_dir: str
    generations: int
    prompts_per_step: int
    train_steps: int
    max_model_len: int
    max_prompt_length: int
    max_completion_length: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    temperature: float
    learning_rate: float
    clip_epsilon: float
    buffer_capacity: int
    buffer_slot_size: int

    @classmethod
    def from_env(cls):
        required = ("MODEL", "PROMPTS", "POLICY_ADAPTER", "REWARD_MODULE")
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise ValueError("Missing environment variables: " + ", ".join(missing))
        config = cls(
            model=os.environ["MODEL"],
            prompts_path=os.environ["PROMPTS"],
            policy_adapter=os.environ["POLICY_ADAPTER"],
            reward_module=os.environ["REWARD_MODULE"],
            output_dir=os.getenv("OUTPUT_DIR", "checkpoints"),
            generations=int(os.getenv("NUM_GENERATIONS", "4")),
            prompts_per_step=int(os.getenv("PROMPTS_PER_STEP", "8")),
            train_steps=int(os.getenv("TRAIN_STEPS", "100")),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "3072")),
            max_completion_length=int(os.getenv("MAX_COMPLETION_LENGTH", "1024")),
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.45")),
            temperature=float(os.getenv("TEMPERATURE", "0.8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "1e-6")),
            clip_epsilon=float(os.getenv("CLIP_EPSILON", "0.2")),
            buffer_capacity=int(os.getenv("BUFFER_CAPACITY", "4")),
            buffer_slot_size=int(os.getenv("BUFFER_SLOT_SIZE", str(256 * 1024 * 1024))),
        )
        if config.generations < 2:
            raise ValueError("NUM_GENERATIONS must be at least 2")
        if config.prompts_per_step < 1:
            raise ValueError("PROMPTS_PER_STEP must be positive")
        if config.train_steps < 1:
            raise ValueError("TRAIN_STEPS must be positive")
        if config.max_prompt_length + config.max_completion_length > config.max_model_len:
            raise ValueError("Prompt and completion limits exceed MAX_MODEL_LEN")
        return config
