# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MooncakeECConnector: Encoder Cache transfer with SHM primary transport
and Mooncake TransferEngine (TCP) fallback.

Primary transport (same-node):
    POSIX shared memory via ECSharedMemoryBuffer — zero extra dependency,
    low-latency memory copy.

Fallback transport (cross-node or SHM-full):
    Mooncake TransferEngine over TCP via MooncakeTCPTransport — pinned CPU
    staging buffer + ZMQ metadata side channel + TE pull transfer.

Unlike KV cache transfer (fixed pre-allocated tensors), EC tensors are
dynamically created with variable sizes, so a dedicated slot-based buffer
is used for both transports.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_shm_buffer import (
    ECSharedMemoryBuffer,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    """Metadata for a single multimodal encoder cache entry."""
    mm_hash: str
    num_token: int
    # TCP metadata, populated by scheduler when entry is found via TCP path
    tcp_meta: dict = field(default_factory=dict)

    @staticmethod
    def make_meta(mm_hash: str, num_token: int,
                  tcp_meta: Optional[dict] = None) -> "MMMeta":
        return MMMeta(
            mm_hash=mm_hash,
            num_token=num_token,
            tcp_meta=tcp_meta or {},
        )


@dataclass
class MooncakeECConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker each step.

    Contains the list of mm_hashes that the consumer worker should
    load, along with which transport path (SHM or TCP) to use.
    """
    mm_datas_to_load: list[MMMeta] = field(default_factory=list)

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas_to_load.append(mm_data)


class MooncakeECConnector(ECConnectorBase):
    """EC connector with SHM primary and TCP fallback transport.

    Producer (encoder instance): computes encoder outputs and writes
    them to SHM first. If SHM is full, falls back to TCP staging buffer.

    Consumer (PD instance): checks SHM first; if not found (cross-node or
    SHM unavailable), queries producer via ZMQ and pulls via TransferEngine.

    Configuration via ec_connector_extra_config:
        shm_name (str): Name for the POSIX SHM segment. Default: derived
            from engine_id.
        max_slots (int): Maximum concurrent EC entries in the SHM buffer.
            Default: 64.
        producer_host (str): Producer's hostname/IP for TCP fallback.
            Required on consumer when using TCP fallback.
        zmq_metadata_port (int): ZMQ port for metadata side channel.
            Default: 14999.
        staging_slots (int): Number of TCP staging buffer slots.
            Default: 16.
        staging_slot_size (int): Max bytes per TCP staging slot.
            Default: 200 MB.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError(
                "ec_transfer_config must be set for MooncakeECConnector"
            )

        engine_id = transfer_config.engine_id or "default"

        # SHM configuration
        self._shm_name: str = transfer_config.get_from_extra_config(
            "shm_name", f"vllm_ec_{engine_id[:16]}"
        )
        self._buffer_size = int(transfer_config.ec_buffer_size)
        self._max_slots: int = transfer_config.get_from_extra_config(
            "max_slots", 64
        )
        self._buffer_device: str = transfer_config.ec_buffer_device or "cuda"

        # TCP fallback configuration
        self._producer_host: str = transfer_config.get_from_extra_config(
            "producer_host", ""
        )
        self._zmq_metadata_port: int = transfer_config.get_from_extra_config(
            "zmq_metadata_port", 14999
        )
        self._staging_slots: int = transfer_config.get_from_extra_config(
            "staging_slots", 16
        )
        self._staging_slot_size: int = transfer_config.get_from_extra_config(
            "staging_slot_size", 200 * 1024 * 1024  # 200 MB
        )

        # Scheduler-side state: mm_hashes pending load
        self._mm_datas_need_loads: dict[str, tuple[int, dict]] = {}

        # SHM buffer (lazily initialized on first use)
        self._shm_buffer: Optional[ECSharedMemoryBuffer] = None

        # TCP transport (lazily initialized on first need)
        self._tcp_transport = None

        # Worker-side tracking for finished transfers
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

        logger.info(
            "MooncakeECConnector initialized: role=%s, shm_name=%s, "
            "buffer_size=%d, max_slots=%d",
            role.name, self._shm_name, self._buffer_size, self._max_slots,
        )

    # ==============================
    # Lazy initialization
    # ==============================

    def _ensure_shm_buffer(self) -> ECSharedMemoryBuffer:
        """Lazily initialize the SHM buffer."""
        if self._shm_buffer is None:
            create = self.is_producer and not self.is_consumer
            self._shm_buffer = ECSharedMemoryBuffer(
                name=self._shm_name,
                size=self._buffer_size,
                create=create,
                max_slots=self._max_slots,
            )
        return self._shm_buffer

    def _ensure_tcp_transport(self):
        """Lazily initialize the TCP transport (Mooncake TransferEngine)."""
        if self._tcp_transport is None:
            from vllm.distributed.ec_transfer.ec_connector.\
                mooncake_ec_tcp_transport import MooncakeTCPTransport
            self._tcp_transport = MooncakeTCPTransport(
                is_producer=self.is_producer and not self.is_consumer,
                producer_host=self._producer_host,
                zmq_metadata_port=self._zmq_metadata_port,
                staging_slots=self._staging_slots,
                staging_slot_size=self._staging_slot_size,
            )
            logger.info(
                "MooncakeECConnector: TCP transport initialized (session=%s)",
                self._tcp_transport.get_rpc_session(),
            )
        return self._tcp_transport

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Load encoder caches into the GPU encoder cache.

        Tries SHM first; falls back to TCP TransferEngine for entries
        not found in SHM.

        Called on the consumer worker before model execution.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeECConnectorMetadata)

        if not metadata.mm_datas_to_load:
            return

        for mm_data in metadata.mm_datas_to_load:
            if mm_data.mm_hash in encoder_cache:
                logger.debug(
                    "EC for hash %s already in encoder_cache, skipping",
                    mm_data.mm_hash,
                )
                self._finished_recving.add(mm_data.mm_hash)
                continue

            tensor = self._load_one(mm_data)
            if tensor is not None:
                encoder_cache[mm_data.mm_hash] = tensor
                self._finished_recving.add(mm_data.mm_hash)
            else:
                logger.warning(
                    "Failed to load EC for hash %s via any transport",
                    mm_data.mm_hash,
                )

    def _load_one(self, mm_data: MMMeta) -> Optional[torch.Tensor]:
        """Attempt to load a single EC tensor, SHM first then TCP."""
        # --- SHM path ---
        try:
            shm_buffer = self._ensure_shm_buffer()
            tensor = shm_buffer.read_tensor(
                mm_data.mm_hash, device=self._buffer_device
            )
            if tensor is not None:
                logger.debug(
                    "Loaded EC from SHM for hash %s, shape=%s",
                    mm_data.mm_hash, tensor.shape,
                )
                return tensor
        except FileNotFoundError:
            logger.debug(
                "SHM buffer '%s' not found, trying TCP for hash %s",
                self._shm_name, mm_data.mm_hash,
            )
        except Exception as exc:
            logger.warning(
                "SHM read error for hash %s: %s", mm_data.mm_hash, exc
            )

        # --- TCP fallback path ---
        if not mm_data.tcp_meta:
            # No TCP metadata was gathered at scheduler time; can't pull
            return None

        try:
            tcp = self._ensure_tcp_transport()
            tensor = tcp.pull_tensor(
                mm_data.mm_hash, mm_data.tcp_meta, device=self._buffer_device
            )
            if tensor is not None:
                logger.debug(
                    "Loaded EC via TCP for hash %s, shape=%s",
                    mm_data.mm_hash, tensor.shape,
                )
            return tensor
        except Exception as exc:
            logger.warning(
                "TCP pull error for hash %s: %s", mm_data.mm_hash, exc
            )
            return None

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache after encoder execution.

        Tries SHM first; falls back to TCP staging buffer if SHM is full.

        Called on the producer worker after each encoder forward pass.
        """
        if not self.is_producer:
            return

        if mm_hash not in encoder_cache:
            logger.warning(
                "mm_hash %s not found in encoder_cache, cannot save", mm_hash
            )
            return

        ec_tensor = encoder_cache[mm_hash]
        cpu_tensor = ec_tensor.detach().cpu().contiguous()

        # --- SHM path (primary) ---
        shm_ok = False
        try:
            shm_buffer = self._ensure_shm_buffer()
            shm_ok = shm_buffer.write_tensor(mm_hash, cpu_tensor)
        except Exception as exc:
            logger.warning("SHM write error for hash %s: %s", mm_hash, exc)

        if shm_ok:
            self._finished_sending.add(mm_hash)
            logger.debug(
                "Saved EC to SHM for hash %s, shape=%s, dtype=%s",
                mm_hash, cpu_tensor.shape, cpu_tensor.dtype,
            )
            return

        logger.debug(
            "SHM write failed for hash %s (buffer full?), trying TCP",
            mm_hash,
        )

        # --- TCP fallback path ---
        try:
            tcp = self._ensure_tcp_transport()
            tcp_ok = tcp.write_tensor(mm_hash, cpu_tensor)
            if tcp_ok:
                self._finished_sending.add(mm_hash)
                logger.debug(
                    "Saved EC via TCP staging for hash %s, shape=%s",
                    mm_hash, cpu_tensor.shape,
                )
            else:
                logger.warning(
                    "TCP staging also full, failed to save EC for hash %s",
                    mm_hash,
                )
        except Exception as exc:
            logger.warning(
                "TCP write error for hash %s: %s", mm_hash, exc
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return sets of mm_hashes that finished sending/receiving."""
        sending = self._finished_sending if self._finished_sending else None
        recving = self._finished_recving if self._finished_recving else None
        self._finished_sending = set()
        self._finished_recving = set()
        return sending, recving

    # ==============================
    # Scheduler-side methods
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        """Check if encoder cache is available (SHM or TCP).

        Returns True only for consumer role when the entry is READY
        in either the SHM buffer or the TCP transport.
        """
        if self.is_producer and not self.is_consumer:
            return False

        # --- SHM path ---
        try:
            shm_buffer = self._ensure_shm_buffer()
            if shm_buffer.has_entry(identifier):
                return True
        except FileNotFoundError:
            logger.debug(
                "SHM buffer '%s' not yet created by producer", self._shm_name
            )
        except Exception as exc:
            logger.warning("SHM has_entry error for %s: %s", identifier, exc)

        # --- TCP fallback path ---
        try:
            tcp = self._ensure_tcp_transport()
            found, _ = tcp.has_entry(identifier)
            return found
        except Exception as exc:
            logger.debug(
                "TCP has_entry error for %s: %s", identifier, exc
            )

        return False

    def update_state_after_alloc(
        self, request: "Request", index: int
    ) -> None:
        """Track encoder cache entries that need to be loaded."""
        mm_hash = request.mm_features[index].identifier
        if not self.is_consumer:
            return

        # Check SHM first
        try:
            shm_buffer = self._ensure_shm_buffer()
            if shm_buffer.has_entry(mm_hash):
                num_encoder_token = request.get_num_encoder_embeds(index)
                self._mm_datas_need_loads[mm_hash] = (num_encoder_token, {})
                return
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # Check TCP
        try:
            tcp = self._ensure_tcp_transport()
            found, tcp_meta = tcp.has_entry(mm_hash)
            if found:
                num_encoder_token = request.get_num_encoder_embeds(index)
                self._mm_datas_need_loads[mm_hash] = (
                    num_encoder_token, tcp_meta
                )
        except Exception as exc:
            logger.debug(
                "TCP update_state_after_alloc error for %s: %s", mm_hash, exc
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build metadata telling worker which EC entries to load."""
        meta = MooncakeECConnectorMetadata()
        for mm_hash, (num_token, tcp_meta) in self._mm_datas_need_loads.items():
            meta.add_mm_data(
                MMMeta.make_meta(mm_hash, num_token, tcp_meta)
            )
        self._mm_datas_need_loads.clear()
        return meta

    def __del__(self):
        if self._shm_buffer is not None:
            self._shm_buffer.close()
            if self.is_producer and not self.is_consumer:
                self._shm_buffer.unlink()
        if self._tcp_transport is not None:
            self._tcp_transport.close()
