import asyncio
import fcntl
import os
import struct
import time
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory


_HEADER = struct.Struct("!QQQQQQ")
_LENGTH = struct.Struct("!I")


class BufferSlot:
    def __init__(self, buffer, index, payload):
        self._buffer = buffer
        self.index = index
        self.payload = payload
        self._released = False

    async def release(self):
        if self._released:
            return
        self.payload.release()
        await asyncio.to_thread(self._buffer._release, self.index)
        self._released = True


class SharedMemoryRingBuffer:
    def __init__(self, shm, owner=False):
        self.shm = shm
        self.name = shm.name
        self.owner = owner
        self.lock_path = f"/tmp/disgrpo-{self.name}.lock"
        if shm.size < _HEADER.size:
            raise ValueError("Shared memory segment is too small")
        self.capacity, self.slot_size, _, _, _, _ = _HEADER.unpack_from(shm.buf)

    @classmethod
    def create(cls, capacity, slot_size):
        if capacity < 1 or slot_size < _LENGTH.size + 1:
            raise ValueError("Invalid ring buffer dimensions")
        size = _HEADER.size + capacity * slot_size
        shm = SharedMemory(create=True, size=size)
        _HEADER.pack_into(shm.buf, 0, capacity, slot_size, 0, 0, 0, 0)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name):
        return cls(SharedMemory(name=name, create=False))

    @contextmanager
    def _locked(self):
        with open(self.lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    async def put(self, payload):
        await asyncio.to_thread(self._put, payload)

    def _put(self, payload):
        source = memoryview(payload)
        if source.nbytes > self.slot_size - _LENGTH.size:
            source.release()
            raise ValueError("Payload exceeds ring buffer slot size")
        try:
            while True:
                with self._locked():
                    capacity, slot_size, head, tail, count, active = _HEADER.unpack_from(self.shm.buf)
                    if count < capacity:
                        offset = _HEADER.size + tail * slot_size
                        _LENGTH.pack_into(self.shm.buf, offset, source.nbytes)
                        self.shm.buf[offset + _LENGTH.size:offset + _LENGTH.size + source.nbytes] = source
                        tail = (tail + 1) % capacity
                        _HEADER.pack_into(self.shm.buf, 0, capacity, slot_size, head, tail, count + 1, active)
                        return
                time.sleep(0.001)
        finally:
            source.release()

    async def get(self):
        return await asyncio.to_thread(self._get)

    def _get(self):
        while True:
            with self._locked():
                capacity, slot_size, head, tail, count, active = _HEADER.unpack_from(self.shm.buf)
                if count and not active:
                    offset = _HEADER.size + head * slot_size
                    length = _LENGTH.unpack_from(self.shm.buf, offset)[0]
                    if length > slot_size - _LENGTH.size:
                        raise ValueError("Invalid slot length")
                    payload = self.shm.buf[offset + _LENGTH.size:offset + _LENGTH.size + length]
                    _HEADER.pack_into(self.shm.buf, 0, capacity, slot_size, head, tail, count, 1)
                    return BufferSlot(self, head, payload)
            time.sleep(0.001)

    def _release(self, index):
        with self._locked():
            capacity, slot_size, head, tail, count, active = _HEADER.unpack_from(self.shm.buf)
            if not active or count == 0 or index != head:
                raise RuntimeError("Ring buffer slot is not active")
            head = (head + 1) % capacity
            _HEADER.pack_into(self.shm.buf, 0, capacity, slot_size, head, tail, count - 1, 0)

    def close(self):
        self.shm.close()

    def unlink(self):
        if self.owner:
            self.shm.unlink()
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass
