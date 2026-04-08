# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared memory buffer for encoder cache (EC) transfer between
producer (encoder) and consumer (PD) instances.

Memory layout:
    [Global Header 64B] [Slot Directory max_slots*256B] [Data Region]

Each slot tracks one encoder cache tensor identified by mm_hash.
Slot state machine: FREE(0) -> WRITING(1) -> READY(2) -> READING(3) -> FREE(0)
"""

import struct
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory

import torch

from vllm.distributed.device_communicators.shm_broadcast import memory_fence
from vllm.logger import init_logger

logger = init_logger(__name__)

# Global header constants
_HEADER_SIZE = 64
_MAGIC = 0xEC_CAFE_DEAD_BEEF
_VERSION = 1

# Slot constants
_SLOT_SIZE = 256
_MAX_HASH_LEN = 128
_MAX_DIMS = 8

# Struct formats for header and slot
# Header: magic(Q) version(I) max_slots(I) data_offset(Q) data_size(Q)
_HEADER_FMT = struct.Struct("<QIIQQ")
# Slot: state(I) hash(128s) data_offset(Q) data_size(Q)
#       ndim(I) shape(8I) dtype(I)
_SLOT_FMT = struct.Struct(f"<I{_MAX_HASH_LEN}sQQI{_MAX_DIMS}II")

# torch dtype <-> integer mapping
_DTYPE_TO_INT: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.float32: 1,
    torch.bfloat16: 2,
    torch.int8: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float64: 6,
    torch.uint8: 7,
}
_INT_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_INT.items()}


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    READING = 3


class ECSharedMemoryBuffer:
    """Manages a POSIX shared memory region for EC tensor transfer."""

    def __init__(self, name: str, size: int, create: bool,
                 max_slots: int = 64):
        """
        Args:
            name: POSIX shared memory name.
            size: Total buffer size in bytes.
            create: True for producer (creates SHM), False for consumer.
            max_slots: Maximum number of concurrent EC entries.
        """
        self._name = name
        self._size = size
        self._max_slots = max_slots
        self._create = create

        slot_dir_size = max_slots * _SLOT_SIZE
        self._data_offset = _HEADER_SIZE + slot_dir_size
        self._data_size = size - self._data_offset
        if self._data_size <= 0:
            raise ValueError(
                f"Buffer size {size} is too small for {max_slots} slots. "
                f"Need at least {self._data_offset + 1} bytes."
            )

        if create:
            # Clean up stale SHM from previous runs
            try:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                logger.warning("Cleaned up stale SHM '%s'", name)
            except FileNotFoundError:
                pass
            self._shm = SharedMemory(name=name, create=True, size=size)
            self._write_header()
            self._init_slots()
            logger.info(
                "Created EC SHM buffer '%s': %d bytes, %d slots, "
                "data region %d bytes",
                name, size, max_slots, self._data_size,
            )
        else:
            self._shm = SharedMemory(name=name, create=False)
            self._validate_header()
            logger.info("Opened EC SHM buffer '%s'", name)

        self._buf = self._shm.buf
        # Bump allocator offset within the data region
        self._next_data_offset = 0

    def _write_header(self):
        data = _HEADER_FMT.pack(
            _MAGIC, _VERSION, self._max_slots,
            self._data_offset, self._data_size,
        )
        self._shm.buf[:_HEADER_FMT.size] = data

    def _validate_header(self):
        data = bytes(self._shm.buf[:_HEADER_FMT.size])
        magic, version, max_slots, data_offset, data_size = (
            _HEADER_FMT.unpack(data)
        )
        if magic != _MAGIC:
            raise RuntimeError(
                f"SHM magic mismatch: expected {_MAGIC:#x}, got {magic:#x}. "
                "Possibly stale shared memory."
            )
        if version != _VERSION:
            raise RuntimeError(
                f"SHM version mismatch: expected {_VERSION}, got {version}."
            )
        self._max_slots = max_slots
        self._data_offset = data_offset
        self._data_size = data_size

    def _init_slots(self):
        """Initialize all slots to FREE with zeroed fields."""
        empty_hash = b"\x00" * _MAX_HASH_LEN
        empty_shape = (0,) * _MAX_DIMS
        slot_data = _SLOT_FMT.pack(
            SlotState.FREE, empty_hash, 0, 0, 0, *empty_shape, 0,
        )
        for i in range(self._max_slots):
            offset = _HEADER_SIZE + i * _SLOT_SIZE
            self._shm.buf[offset:offset + _SLOT_FMT.size] = slot_data

    def _slot_offset(self, idx: int) -> int:
        return _HEADER_SIZE + idx * _SLOT_SIZE

    def _read_slot(self, idx: int) -> tuple:
        offset = self._slot_offset(idx)
        data = bytes(self._buf[offset:offset + _SLOT_FMT.size])
        return _SLOT_FMT.unpack(data)

    def _write_slot_state(self, idx: int, state: SlotState):
        offset = self._slot_offset(idx)
        struct.pack_into("<I", self._buf, offset, state)

    def _write_slot(self, idx: int, state: SlotState, mm_hash: str,
                    data_offset: int, data_size: int, shape: tuple,
                    dtype: torch.dtype):
        offset = self._slot_offset(idx)
        hash_bytes = mm_hash.encode("utf-8")[:_MAX_HASH_LEN]
        hash_bytes = hash_bytes.ljust(_MAX_HASH_LEN, b"\x00")
        ndim = len(shape)
        padded_shape = shape + (0,) * (_MAX_DIMS - ndim)
        dtype_int = _DTYPE_TO_INT.get(dtype, 0)
        slot_data = _SLOT_FMT.pack(
            state, hash_bytes, data_offset, data_size,
            ndim, *padded_shape, dtype_int,
        )
        self._buf[offset:offset + _SLOT_FMT.size] = slot_data

    def _find_slot_by_hash(self, mm_hash: str,
                           target_state: SlotState | None = None
                           ) -> int | None:
        """Find slot index matching mm_hash and optionally a specific state."""
        hash_bytes = mm_hash.encode("utf-8")[:_MAX_HASH_LEN]
        for i in range(self._max_slots):
            fields = self._read_slot(i)
            state = fields[0]
            slot_hash = fields[1].rstrip(b"\x00")
            if slot_hash == hash_bytes:
                if target_state is None or state == target_state:
                    return i
        return None

    def _find_free_slot(self) -> int | None:
        for i in range(self._max_slots):
            fields = self._read_slot(i)
            if fields[0] == SlotState.FREE:
                return i
        return None

    def _try_compact_data_region(self):
        """Reset bump allocator if all slots are FREE."""
        for i in range(self._max_slots):
            fields = self._read_slot(i)
            if fields[0] != SlotState.FREE:
                return
        self._next_data_offset = 0
        logger.debug("Compacted EC SHM data region")

    def write_tensor(self, mm_hash: str, tensor: torch.Tensor) -> bool:
        """Write a CPU tensor into the SHM buffer.

        Args:
            mm_hash: Unique identifier for the encoder cache entry.
            tensor: CPU tensor to write. Must be contiguous.

        Returns:
            True if successful, False if buffer is full.
        """
        assert tensor.is_cpu, "Tensor must be on CPU for SHM write"
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        data_size = tensor.nbytes

        # Check if already exists (idempotent)
        existing = self._find_slot_by_hash(mm_hash)
        if existing is not None:
            logger.debug("EC for hash %s already in SHM, skipping", mm_hash)
            return True

        # Try compaction if needed
        if self._next_data_offset + data_size > self._data_size:
            self._try_compact_data_region()

        # Check space
        if self._next_data_offset + data_size > self._data_size:
            logger.warning(
                "EC SHM buffer full: need %d bytes, have %d available",
                data_size, self._data_size - self._next_data_offset,
            )
            return False

        # Find free slot
        slot_idx = self._find_free_slot()
        if slot_idx is None:
            logger.warning("EC SHM: no free slots available")
            return False

        # Write slot header with WRITING state
        rel_offset = self._next_data_offset
        self._write_slot(
            slot_idx, SlotState.WRITING, mm_hash,
            rel_offset, data_size, tuple(tensor.shape), tensor.dtype,
        )
        memory_fence()

        # Copy tensor data to SHM data region
        abs_offset = self._data_offset + rel_offset
        tensor_bytes = bytes(tensor.untyped_storage())
        self._buf[abs_offset:abs_offset + data_size] = tensor_bytes

        # Advance bump allocator
        self._next_data_offset += data_size

        # Mark as READY
        memory_fence()
        self._write_slot_state(slot_idx, SlotState.READY)
        memory_fence()

        return True

    def read_tensor(self, mm_hash: str,
                    device: str = "cuda") -> torch.Tensor | None:
        """Read a tensor from SHM and copy to the specified device.

        Args:
            mm_hash: Identifier for the encoder cache entry.
            device: Target device for the tensor.

        Returns:
            Tensor on target device, or None if not found/not ready.
        """
        memory_fence()
        slot_idx = self._find_slot_by_hash(mm_hash, SlotState.READY)
        if slot_idx is None:
            return None

        # Mark as READING
        self._write_slot_state(slot_idx, SlotState.READING)
        memory_fence()

        # Parse slot metadata
        fields = self._read_slot(slot_idx)
        rel_offset = fields[2]
        data_size = fields[3]
        ndim = fields[4]
        shape = tuple(fields[5:5 + ndim])
        dtype_int = fields[5 + _MAX_DIMS]
        dtype = _INT_TO_DTYPE.get(dtype_int, torch.float16)

        # Read tensor data from SHM
        abs_offset = self._data_offset + rel_offset
        raw_bytes = bytes(self._buf[abs_offset:abs_offset + data_size])

        # Reconstruct tensor
        cpu_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=dtype)
        cpu_tensor = cpu_tensor.reshape(shape)

        # Copy to target device
        result = cpu_tensor.to(device, non_blocking=True)

        # Mark slot as FREE
        memory_fence()
        self._write_slot_state(slot_idx, SlotState.FREE)
        memory_fence()

        return result

    def has_entry(self, mm_hash: str) -> bool:
        """Check if a READY entry exists for the given mm_hash."""
        memory_fence()
        return self._find_slot_by_hash(mm_hash, SlotState.READY) is not None

    def close(self):
        """Close the shared memory handle."""
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        """Remove the shared memory segment (producer only)."""
        try:
            self._shm.unlink()
            logger.info("Unlinked EC SHM buffer '%s'", self._name)
        except Exception:
            pass

    def __del__(self):
        self.close()
        if self._create:
            self.unlink()
