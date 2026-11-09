import struct
from dataclasses import dataclass

import numpy as np


_HEADER = struct.Struct("!II")


@dataclass
class RolloutBatch:
    tokens: np.ndarray
    attention_mask: np.ndarray
    response_mask: np.ndarray
    old_logprobs: np.ndarray
    rewards: np.ndarray
    group_ids: np.ndarray

    def __post_init__(self):
        rows, width = self.tokens.shape
        if self.attention_mask.shape != (rows, width):
            raise ValueError("Invalid attention mask shape")
        if self.response_mask.shape != (rows, width):
            raise ValueError("Invalid response mask shape")
        if self.old_logprobs.shape != (rows, width):
            raise ValueError("Invalid log probability shape")
        if self.rewards.shape != (rows,) or self.group_ids.shape != (rows,):
            raise ValueError("Invalid rollout metadata shape")

    def encode(self):
        arrays = (
            np.ascontiguousarray(self.tokens, dtype=np.int32),
            np.ascontiguousarray(self.attention_mask, dtype=np.uint8),
            np.ascontiguousarray(self.response_mask, dtype=np.uint8),
            np.ascontiguousarray(self.old_logprobs, dtype=np.float32),
            np.ascontiguousarray(self.rewards, dtype=np.float32),
            np.ascontiguousarray(self.group_ids, dtype=np.int32),
        )
        rows, width = arrays[0].shape
        return b"".join(
            (_HEADER.pack(rows, width), *(array.tobytes() for array in arrays))
        )

    @classmethod
    def from_payload(cls, payload):
        view = memoryview(payload)
        rows, width = _HEADER.unpack_from(view)
        offset = _HEADER.size

        def take(dtype, count):
            nonlocal offset
            result = np.frombuffer(view, dtype=dtype, count=count, offset=offset)
            offset += result.nbytes
            return result

        tokens = take(np.int32, rows * width).reshape(rows, width)
        attention_mask = take(np.uint8, rows * width).reshape(rows, width)
        response_mask = take(np.uint8, rows * width).reshape(rows, width)
        old_logprobs = take(np.float32, rows * width).reshape(rows, width)
        rewards = take(np.float32, rows)
        group_ids = take(np.int32, rows)
        if offset != view.nbytes:
            raise ValueError("Rollout payload size mismatch")
        return cls(tokens, attention_mask, response_mask, old_logprobs, rewards, group_ids)
