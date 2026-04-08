# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mooncake TransferEngine (TCP) transport for encoder cache transfer.

This module provides a fallback transport for MooncakeECConnector when
POSIX shared memory is unavailable (e.g., cross-node EPD deployments).

Architecture:
  - Both producer and consumer pre-allocate a pinned CPU staging buffer
    divided into fixed-size slots.
  - The staging buffer is registered with Mooncake's TransferEngine once.
  - A ZMQ REP/REQ side channel on the producer carries tensor metadata
    (virtual address, shape, dtype) so the consumer can pull data.
  - The consumer initiates data transfer via TransferEngine's
    batch_transfer_sync_write (pull model).

Data flow:
  Producer: GPU → CPU (staging slot) → mark READY → answer ZMQ queries
  Consumer: ZMQ query → get metadata → TE pull → CPU staging → GPU
"""

import ctypes
import json
import threading
from enum import IntEnum
from typing import Optional

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Mooncake is not installed. MooncakeECConnector TCP fallback will "
        "be unavailable. Install Mooncake by following: "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md"
    )
    TransferEngine = None

# torch dtype <-> int mapping (matches mooncake_ec_shm_buffer.py)
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

# Default staging configuration
_DEFAULT_STAGING_SLOTS = 16
_DEFAULT_STAGING_SLOT_SIZE = 200 * 1024 * 1024  # 200 MB per slot
_DEFAULT_ZMQ_METADATA_PORT = 14999


class _StagingSlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    READING = 3


class _SlotMeta:
    """Metadata for one staging buffer slot."""

    __slots__ = ("state", "mm_hash", "va", "nbytes", "shape", "dtype")

    def __init__(self):
        self.state = _StagingSlotState.FREE
        self.mm_hash: str = ""
        self.va: int = 0
        self.nbytes: int = 0
        self.shape: tuple = ()
        self.dtype: torch.dtype = torch.float16


class StagingBuffer:
    """Pre-allocated pinned CPU staging buffer with slot-based management.

    Managed entirely within a single process (producer or consumer).
    Thread-safe via a per-instance lock.

    Args:
        num_slots: Number of fixed-size slots.
        slot_size: Maximum bytes per slot.
    """

    def __init__(self, num_slots: int, slot_size: int):
        self._num_slots = num_slots
        self._slot_size = slot_size
        self._total_size = num_slots * slot_size
        self._lock = threading.Lock()

        # Allocate a contiguous pinned numpy array.
        # numpy arrays allocated this way are page-aligned and can be
        # registered with Mooncake TransferEngine.
        self._buf = np.zeros(self._total_size, dtype=np.uint8)

        # Virtual address of the underlying buffer for TransferEngine
        self._base_va: int = self._buf.ctypes.data

        self._slots: list[_SlotMeta] = [
            _SlotMeta() for _ in range(num_slots)
        ]
        # mm_hash → slot index for fast lookup
        self._hash_to_slot: dict[str, int] = {}

        logger.info(
            "StagingBuffer: %d slots x %d bytes = %d bytes total, base_va=%#x",
            num_slots, slot_size, self._total_size, self._base_va,
        )

    @property
    def base_va(self) -> int:
        return self._base_va

    @property
    def total_size(self) -> int:
        return self._total_size

    def slot_va(self, slot_idx: int) -> int:
        return self._base_va + slot_idx * self._slot_size

    def _find_free_slot(self) -> Optional[int]:
        for i, slot in enumerate(self._slots):
            if slot.state == _StagingSlotState.FREE:
                return i
        return None

    def write_tensor(self, mm_hash: str, tensor: torch.Tensor) -> bool:
        """Write a CPU tensor into a free staging slot.

        Args:
            mm_hash: Unique identifier for this encoder cache entry.
            tensor: Contiguous CPU tensor to write.

        Returns:
            True on success, False if no free slot is available or
            tensor is too large.
        """
        assert tensor.is_cpu, "Tensor must be on CPU for staging write"
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        nbytes = tensor.nbytes
        if nbytes > self._slot_size:
            logger.warning(
                "TCP staging: tensor for hash %s is %d bytes, exceeds "
                "slot_size %d. Cannot stage.",
                mm_hash, nbytes, self._slot_size,
            )
            return False

        with self._lock:
            # Idempotent: already staged
            if mm_hash in self._hash_to_slot:
                return True

            slot_idx = self._find_free_slot()
            if slot_idx is None:
                logger.warning(
                    "TCP staging: no free slots for hash %s", mm_hash
                )
                return False

            slot = self._slots[slot_idx]
            slot.state = _StagingSlotState.WRITING
            slot.mm_hash = mm_hash
            slot.va = self.slot_va(slot_idx)
            slot.nbytes = nbytes
            slot.shape = tuple(tensor.shape)
            slot.dtype = tensor.dtype

            # Copy tensor bytes into staging buffer
            offset = slot_idx * self._slot_size
            tensor_bytes = tensor.numpy().tobytes()
            self._buf[offset:offset + nbytes] = np.frombuffer(
                tensor_bytes, dtype=np.uint8
            )

            slot.state = _StagingSlotState.READY
            self._hash_to_slot[mm_hash] = slot_idx

        logger.debug(
            "TCP staging: wrote hash %s → slot %d, shape=%s, nbytes=%d",
            mm_hash, slot_idx, slot.shape, nbytes,
        )
        return True

    def get_slot_meta(self, mm_hash: str) -> Optional[dict]:
        """Return slot metadata dict for a READY entry, or None."""
        with self._lock:
            slot_idx = self._hash_to_slot.get(mm_hash)
            if slot_idx is None:
                return None
            slot = self._slots[slot_idx]
            if slot.state != _StagingSlotState.READY:
                return None
            return {
                "va": slot.va,
                "nbytes": slot.nbytes,
                "shape": list(slot.shape),
                "dtype_int": _DTYPE_TO_INT.get(slot.dtype, 0),
            }

    def free_slot(self, mm_hash: str) -> None:
        """Release the slot occupied by mm_hash."""
        with self._lock:
            slot_idx = self._hash_to_slot.pop(mm_hash, None)
            if slot_idx is None:
                return
            slot = self._slots[slot_idx]
            slot.state = _StagingSlotState.FREE
            slot.mm_hash = ""

    def read_bytes_into(self, mm_hash: str,
                        dst_buf: np.ndarray) -> Optional[dict]:
        """Copy staging slot bytes into dst_buf.

        Used on the consumer side when the consumer pulls via local copy
        (same-node TCP on loopback; the TE already placed data here).
        Returns slot metadata or None if not found.
        """
        with self._lock:
            slot_idx = self._hash_to_slot.get(mm_hash)
            if slot_idx is None:
                return None
            slot = self._slots[slot_idx]
            if slot.state not in (
                _StagingSlotState.READY, _StagingSlotState.READING
            ):
                return None
            meta = {
                "va": slot.va,
                "nbytes": slot.nbytes,
                "shape": list(slot.shape),
                "dtype_int": _DTYPE_TO_INT.get(slot.dtype, 0),
            }
        return meta

    def get_numpy_view(self, slot_idx: int, nbytes: int) -> np.ndarray:
        """Return a numpy view of slot data (for frombuffer)."""
        offset = slot_idx * self._slot_size
        return self._buf[offset:offset + nbytes]

    def find_slot_idx(self, mm_hash: str) -> Optional[int]:
        with self._lock:
            return self._hash_to_slot.get(mm_hash)


class MooncakeTCPTransport:
    """Mooncake TransferEngine (TCP) transport for EC tensors.

    Manages:
    - TransferEngine lifecycle (TCP protocol)
    - Pinned CPU staging buffer registration
    - ZMQ side channel for tensor metadata exchange
    - Consumer-initiated pull transfers

    Args:
        is_producer: Whether this instance runs on the encoder (producer) node.
        producer_host: Producer's hostname or IP (required for consumer).
        zmq_metadata_port: TCP port for ZMQ metadata REP/REQ (default: 14999).
        staging_slots: Number of staging buffer slots (default: 16).
        staging_slot_size: Max bytes per slot (default: 200 MB).
    """

    def __init__(
        self,
        is_producer: bool,
        producer_host: str = "",
        zmq_metadata_port: int = _DEFAULT_ZMQ_METADATA_PORT,
        staging_slots: int = _DEFAULT_STAGING_SLOTS,
        staging_slot_size: int = _DEFAULT_STAGING_SLOT_SIZE,
    ):
        if TransferEngine is None:
            raise RuntimeError(
                "Mooncake is not installed. Cannot use TCP transport. "
                "Install Mooncake: "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md"
            )

        self._is_producer = is_producer
        self._producer_host = producer_host
        self._zmq_metadata_port = zmq_metadata_port
        self._staging_slots = staging_slots
        self._staging_slot_size = staging_slot_size

        self._hostname = get_ip()
        self._engine = TransferEngine()
        self._te_port: int = 0
        self._staging: StagingBuffer = StagingBuffer(
            staging_slots, staging_slot_size
        )

        self._zmq_ctx = None
        self._zmq_rep_socket = None   # producer side
        self._zmq_req_socket = None   # consumer side
        self._zmq_thread: Optional[threading.Thread] = None
        self._zmq_stop = threading.Event()

        self._initialize()

    def _initialize(self):
        """Initialize TransferEngine and register staging buffer."""
        logger.info(
            "MooncakeTCPTransport: initializing TransferEngine (TCP), "
            "role=%s", "producer" if self._is_producer else "consumer",
        )

        ret = self._engine.initialize(self._hostname, "P2PHANDSHAKE", "tcp", "")
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine TCP initialization failed (ret={ret})"
            )

        self._te_port = self._engine.get_rpc_port()
        logger.info(
            "MooncakeTCPTransport: TransferEngine listening at %s:%d",
            self._hostname, self._te_port,
        )

        # Register staging buffer with TransferEngine
        va = self._staging.base_va
        size = self._staging.total_size
        ret = self._engine.batch_register_memory([va], [size])
        if ret != 0:
            raise RuntimeError(
                f"Mooncake staging buffer registration failed (ret={ret})"
            )
        logger.info(
            "MooncakeTCPTransport: staging buffer registered, "
            "va=%#x, size=%d", va, size,
        )

        # Set up ZMQ
        import zmq
        self._zmq_ctx = zmq.Context()

        if self._is_producer:
            self._zmq_rep_socket = self._zmq_ctx.socket(zmq.REP)
            self._zmq_rep_socket.bind(
                f"tcp://*:{self._zmq_metadata_port}"
            )
            self._zmq_thread = threading.Thread(
                target=self._zmq_rep_loop,
                name="mooncake-ec-zmq-rep",
                daemon=True,
            )
            self._zmq_thread.start()
            logger.info(
                "MooncakeTCPTransport: ZMQ metadata server listening on "
                "port %d", self._zmq_metadata_port,
            )
        else:
            if not self._producer_host:
                raise ValueError(
                    "MooncakeTCPTransport: consumer requires 'producer_host' "
                    "in ec_connector_extra_config"
                )
            self._zmq_req_socket = self._zmq_ctx.socket(zmq.REQ)
            self._zmq_req_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
            self._zmq_req_socket.connect(
                f"tcp://{self._producer_host}:{self._zmq_metadata_port}"
            )
            logger.info(
                "MooncakeTCPTransport: connected to producer ZMQ at "
                "%s:%d", self._producer_host, self._zmq_metadata_port,
            )

    def _zmq_rep_loop(self):
        """Background thread: answer consumer metadata queries (producer only)."""
        logger.debug("MooncakeTCPTransport: ZMQ REP loop started")
        while not self._zmq_stop.is_set():
            try:
                # Poll with timeout to allow clean shutdown
                if not self._zmq_rep_socket.poll(timeout=500):
                    continue
                raw = self._zmq_rep_socket.recv()
                mm_hash = raw.decode("utf-8")

                meta = self._staging.get_slot_meta(mm_hash)
                if meta is not None:
                    # Include our TE port so consumer can build the session
                    meta["te_port"] = self._te_port
                    meta["te_host"] = self._hostname
                    response = json.dumps({"found": True, **meta})
                else:
                    response = json.dumps({"found": False})

                self._zmq_rep_socket.send(response.encode("utf-8"))
            except Exception as exc:
                if not self._zmq_stop.is_set():
                    logger.warning(
                        "MooncakeTCPTransport: ZMQ REP error: %s", exc
                    )
        logger.debug("MooncakeTCPTransport: ZMQ REP loop stopped")

    def get_rpc_session(self) -> str:
        """Return this node's TransferEngine session string."""
        return f"{self._hostname}:{self._te_port}"

    # ------------------------------------------------------------------
    # Producer-side API
    # ------------------------------------------------------------------

    def write_tensor(self, mm_hash: str, cpu_tensor: torch.Tensor) -> bool:
        """Stage a CPU tensor for consumer to pull via TransferEngine.

        Args:
            mm_hash: Unique hash for this encoder cache entry.
            cpu_tensor: Contiguous CPU tensor.

        Returns:
            True on success, False if staging buffer is full.
        """
        return self._staging.write_tensor(mm_hash, cpu_tensor)

    def free_entry(self, mm_hash: str) -> None:
        """Release staging slot for mm_hash (call after consumer confirms receipt)."""
        self._staging.free_slot(mm_hash)

    # ------------------------------------------------------------------
    # Consumer-side API
    # ------------------------------------------------------------------

    def has_entry(self, mm_hash: str) -> tuple[bool, dict]:
        """Query producer via ZMQ: does it have a READY entry for mm_hash?

        Returns:
            (found, metadata_dict). metadata_dict is empty if not found.
        """
        if self._zmq_req_socket is None:
            return False, {}

        try:
            self._zmq_req_socket.send(mm_hash.encode("utf-8"))
            raw = self._zmq_req_socket.recv()
            data = json.loads(raw.decode("utf-8"))
            if data.get("found"):
                return True, data
        except Exception as exc:
            logger.warning(
                "MooncakeTCPTransport: ZMQ query for hash %s failed: %s",
                mm_hash, exc,
            )
        return False, {}

    def pull_tensor(
        self,
        mm_hash: str,
        meta: dict,
        device: str = "cuda",
    ) -> Optional[torch.Tensor]:
        """Pull tensor from producer's staging buffer via TransferEngine.

        Args:
            mm_hash: Encoder cache hash.
            meta: Metadata dict returned by has_entry() (contains va, nbytes,
                  shape, dtype_int, te_host, te_port).
            device: Target device for the output tensor.

        Returns:
            Tensor on the target device, or None on failure.
        """
        producer_va: int = meta["va"]
        nbytes: int = meta["nbytes"]
        shape: list = meta["shape"]
        dtype_int: int = meta["dtype_int"]
        te_host: str = meta["te_host"]
        te_port: int = meta["te_port"]

        dtype = _INT_TO_DTYPE.get(dtype_int, torch.float16)
        producer_session = f"{te_host}:{te_port}"

        # Use a free consumer staging slot as receive buffer
        slot_idx = self._find_free_consumer_slot(nbytes)
        if slot_idx is None:
            logger.warning(
                "TCP transport: no free consumer staging slot for hash %s",
                mm_hash,
            )
            return None

        dst_va = self._staging.slot_va(slot_idx)

        try:
            ret = self._engine.batch_transfer_sync_write(
                producer_session,
                [producer_va],   # source: producer's staging slot VA
                [dst_va],        # dest: consumer's staging slot VA
                [nbytes],
            )
            if ret != 0:
                logger.warning(
                    "TCP transport: batch_transfer_sync_write failed "
                    "(ret=%d) for hash %s", ret, mm_hash,
                )
                return None
        except Exception as exc:
            logger.warning(
                "TCP transport: transfer error for hash %s: %s", mm_hash, exc
            )
            return None

        # Read bytes from consumer's staging slot
        offset = slot_idx * self._staging_slot_size
        raw_bytes = bytes(self._staging._buf[offset:offset + nbytes])
        cpu_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=dtype)
        cpu_tensor = cpu_tensor.reshape(shape)

        result = cpu_tensor.to(device, non_blocking=True)

        # Free the consumer staging slot
        self._free_consumer_slot(slot_idx)

        logger.debug(
            "TCP transport: pulled hash %s from %s, shape=%s, nbytes=%d",
            mm_hash, producer_session, shape, nbytes,
        )
        return result

    def _find_free_consumer_slot(self, nbytes: int) -> Optional[int]:
        """Find a free staging slot large enough for nbytes."""
        if nbytes > self._staging_slot_size:
            return None
        with self._staging._lock:
            for i, slot in enumerate(self._staging._slots):
                if slot.state == _StagingSlotState.FREE:
                    slot.state = _StagingSlotState.READING
                    return i
        return None

    def _free_consumer_slot(self, slot_idx: int) -> None:
        with self._staging._lock:
            self._staging._slots[slot_idx].state = _StagingSlotState.FREE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Shutdown ZMQ and release resources."""
        self._zmq_stop.set()
        if self._zmq_thread is not None:
            self._zmq_thread.join(timeout=2.0)
        if self._zmq_rep_socket is not None:
            self._zmq_rep_socket.close(linger=0)
        if self._zmq_req_socket is not None:
            self._zmq_req_socket.close(linger=0)
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
        logger.info("MooncakeTCPTransport: closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
