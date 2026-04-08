# MooncakeECConnector Design Document

## 1. Overview

MooncakeECConnector is an encoder cache (EC) transfer connector for vLLM's EPD (Encoder-Prefill-Decode) disaggregation feature. It transfers encoder cache tensors (multimodal embeddings) from an encoder instance (producer) to a PD instance (consumer).

It provides two complementary transports:

| Transport | Scope | Dependency | When used |
|-----------|-------|------------|-----------|
| POSIX Shared Memory | Same-node | None (stdlib) | Primary path |
| Mooncake TransferEngine (TCP) | Cross-node or SHM-full | `mooncake` | Fallback path |

### Motivation

In EPD disaggregation, the encoder computation is separated from the prefill-decode (PD) instance. The encoder produces multimodal embeddings (e.g., image/video features) that the PD instance needs for subsequent generation. Without a transfer mechanism, the PD instance would have to redundantly re-compute these embeddings.

The existing `ECExampleConnector` uses disk-based storage (safetensors files), which is suitable for debugging but has significant I/O overhead. MooncakeECConnector uses POSIX shared memory for near-zero-copy same-node transfer and Mooncake's TransferEngine over TCP for cross-node transfer.

### Key Challenge

Unlike KV cache transfer (fixed, pre-allocated tensors registered once with the transfer engine), encoder cache tensors are **dynamically created** with **variable sizes** depending on the multimodal input (e.g., different image resolutions produce different numbers of encoder tokens). This requires a dedicated slot-based buffer for both transports.

Reference: [PR #33714 comment](https://github.com/vllm-project/vllm/pull/33714#issuecomment-3882716972)

## 2. Architecture

### 2.1 Transport Selection (Fallback Chain)

```
save_caches / has_cache_item / start_load_caches
    ↓
    [1] Try SHM (ECSharedMemoryBuffer)
        ✓ Success → done
        ✗ Fail (buffer full / FileNotFoundError)
    ↓
    [2] Try TCP (MooncakeTCPTransport)
        ✓ Success → done
        ✗ Fail (mooncake not installed / transfer error)
    ↓
    [3] Log warning → consumer falls back to local encoder computation
```

### 2.2 System Diagram

```
+---------------------------+     Primary: POSIX SHM     +---------------------------+
|   Encoder Instance        |  ========================> |    PD Instance            |
|   (ec_producer)           |                            |    (ec_consumer)          |
|                           |     Fallback: TCP (TE)     |                           |
|   Encoder Forward         |  ~~~~~~~~~~~~~~~~~~~~~~~~> |   Scheduler:              |
|        |                  |                            |   has_cache_item()        |
|        v                  |                            |        |                  |
|   save_caches()           |   ZMQ metadata side chan.  |        v                  |
|   1. GPU → CPU            |  <~~~~~~~~~~~~~~~~~~~~~~~~ |   update_state_after_alloc|
|   2. CPU → SHM            |                            |        |                  |
|      or                   |                            |        v                  |
|   2. CPU → TCP staging    |                            |   build_connector_meta()  |
|   3. ZMQ REP server       |                            |        |                  |
|      (answers metadata)   |                            |        v                  |
|                           |                            |   start_load_caches()     |
|                           |                            |   1. SHM → CPU → GPU     |
|                           |                            |      or                   |
|                           |                            |   1. ZMQ query metadata  |
|                           |                            |   2. TE pull → CPU → GPU |
+---------------------------+                            +---------------------------+
```

### 2.3 Components

| Component | File | Purpose |
|-----------|------|---------|
| `ECSharedMemoryBuffer` | `mooncake_ec_shm_buffer.py` | SHM buffer with slot-based allocation |
| `StagingBuffer` | `mooncake_ec_tcp_transport.py` | Pinned CPU staging buffer for TE registration |
| `MooncakeTCPTransport` | `mooncake_ec_tcp_transport.py` | TransferEngine (TCP) + ZMQ side channel |
| `MooncakeECConnector` | `mooncake_ec_connector.py` | ECConnectorBase implementation |
| `MooncakeECConnectorMetadata` | `mooncake_ec_connector.py` | Scheduler-to-worker metadata |
| Factory registration | `factory.py` | Connector discovery |

## 3. Shared Memory Transport (Primary)

### 3.1 Memory Layout

The SHM buffer is a single contiguous POSIX shared memory region:

```
+-------------------------------------------------------------------+
| GLOBAL HEADER (64 bytes)                                          |
|   magic:        8B  (0xEC_CAFE_DEAD_BEEF)                         |
|   version:      4B  (1)                                           |
|   max_slots:    4B  (default 64)                                  |
|   data_offset:  8B  (byte offset to data region)                  |
|   data_size:    8B  (data region capacity)                        |
|   padding:     32B                                                |
+-------------------------------------------------------------------+
| SLOT DIRECTORY (max_slots x 256 bytes)                            |
|   Each slot:                                                      |
|     state:        4B   (FREE/WRITING/READY/READING)               |
|     mm_hash:    128B   (null-terminated UTF-8)                    |
|     data_offset:  8B   (relative offset in data region)           |
|     data_size:    8B   (tensor bytes)                             |
|     ndim:         4B   (number of dimensions)                     |
|     shape:       32B   (up to 8 dims, 4B each)                   |
|     dtype:        4B   (torch dtype enum)                         |
|     padding:     68B                                              |
+-------------------------------------------------------------------+
| DATA REGION (remaining bytes)                                     |
|   [tensor bytes for slot 0] [tensor bytes for slot 1] ...         |
+-------------------------------------------------------------------+
```

With default settings (64 slots), the slot directory takes ~16 KB, leaving nearly all of the configured buffer size (default 1 GB) for tensor data.

### 3.2 Slot State Machine

```
FREE (0)  --[producer acquires]--> WRITING (1)
WRITING   --[data copied, fence]--> READY (2)
READY     --[consumer acquires]--> READING (3)
READING   --[data read, fence]-->  FREE (0)
```

State transitions use `memory_fence()` from `vllm/distributed/device_communicators/shm_broadcast.py` for cross-process visibility.

### 3.3 Data Region Allocation

A simple bump allocator manages the data region:

- `next_data_offset` tracks the next available position
- When writing, check if `next_data_offset + tensor_size <= data_region_size`
- When all slots are FREE, the allocator resets to 0 (compaction)
- If the buffer is full, `write_tensor()` returns `False` → fallback to TCP

## 4. TCP Transport via Mooncake TransferEngine (Fallback)

### 4.1 Design Rationale

The TCP transport uses Mooncake's `TransferEngine` with the `"tcp"` protocol. Unlike the KV cache connector (which pre-registers fixed GPU addresses), EC tensors are variable-sized, requiring dynamic staging:

- Both producer and consumer pre-allocate a **pinned CPU staging buffer** divided into `staging_slots` fixed-size slots of `staging_slot_size` bytes each.
- The staging buffer is registered with `TransferEngine` **once** at initialization (fixed virtual address).
- A **ZMQ REQ/REP side channel** carries tensor metadata (virtual address, shape, dtype) from producer to consumer.
- The consumer uses a **pull model**: after getting metadata via ZMQ, it calls `batch_transfer_sync_write` to pull data from producer's staging slot into its own staging slot, then copies to GPU.

### 4.2 Staging Buffer Layout

```
+------------------------------------------------------------------+
| Pinned CPU numpy array (staging_slots × staging_slot_size bytes)  |
|                                                                   |
|  [Slot 0: up to staging_slot_size bytes]                         |
|  [Slot 1: up to staging_slot_size bytes]                         |
|  ...                                                              |
|  [Slot N-1: up to staging_slot_size bytes]                       |
+------------------------------------------------------------------+
```

Slot state is tracked in a Python dict (not shared memory), protected by a threading lock. State machine: `FREE → WRITING → READY → READING → FREE`.

### 4.3 TransferEngine Initialization

Both producer and consumer initialize TransferEngine at startup:

```python
engine = TransferEngine()
ret = engine.initialize(hostname, "P2PHANDSHAKE", "tcp", "")
te_port = engine.get_rpc_port()

# Register staging buffer (once)
ret = engine.batch_register_memory([staging_base_va], [staging_total_size])
```

### 4.4 ZMQ Side Channel

The producer runs a ZMQ `REP` server on a configurable port (`zmq_metadata_port`). The consumer sends a `REQ` with the `mm_hash` and receives:

```json
{
  "found": true,
  "va": 140234567890,
  "nbytes": 4718592,
  "shape": [576, 4096],
  "dtype_int": 0,
  "te_host": "10.0.0.1",
  "te_port": 12345
}
```

Or `{"found": false}` if the entry is not yet ready.

### 4.5 Data Flow (TCP Path)

**Producer:**
```
Encoder forward → GPU tensor
    → .detach().cpu().contiguous()         # GPU → CPU
    → staging_buffer.write_tensor()        # CPU → pinned staging slot
    → slot marked READY in metadata dict
    → ZMQ REP server answers queries
```

**Consumer (scheduler thread):**
```
update_state_after_alloc(mm_hash):
    → zmq_req.send(mm_hash)
    → receive {found, va, nbytes, shape, dtype, te_host, te_port}
    → store in _mm_datas_need_loads[mm_hash] = (num_token, tcp_meta)
```

**Consumer (worker thread):**
```
start_load_caches():
    → for each mm_data with tcp_meta:
        → tcp_transport.pull_tensor(mm_hash, tcp_meta, device)
            → find free consumer staging slot
            → engine.batch_transfer_sync_write(
                   f"{te_host}:{te_port}",
                   [producer_slot_va],     # source
                   [consumer_slot_va],     # dest (own staging)
                   [nbytes])
            → torch.frombuffer(consumer_staging_bytes) → reshape
            → tensor.to("cuda", non_blocking=True)
        → encoder_cache[mm_hash] = tensor
```

### 4.6 Transfer Engine Session

The session string `"{hostname}:{te_port}"` identifies the remote peer. This is sufficient for TransferEngine's P2PHANDSHAKE protocol to establish a TCP connection and perform the transfer.

## 5. Connector Implementation

### 5.1 Scheduler-Side Flow

```
_try_schedule_encoder_inputs() in scheduler.py
    |
    v
has_cache_item(mm_hash)
    |-- SHM: check slot READY
    |-- TCP: ZMQ query to producer
    |
    v (if True)
update_state_after_alloc()
    |-- SHM: record (mm_hash, num_token, tcp_meta={})
    |-- TCP: record (mm_hash, num_token, tcp_meta={...})
    |
    v
build_connector_meta()
    |-- creates MooncakeECConnectorMetadata with mm_datas_to_load list
    |-- each MMMeta carries tcp_meta for worker to use
```

### 5.2 Worker-Side Flow

**Producer (encoder instance):**
```
Encoder forward pass
    |
    v
save_caches(encoder_cache, mm_hash)
    |-- ec_tensor.detach().cpu().contiguous()     # GPU → CPU
    |-- shm_buffer.write_tensor() → True          # SHM path ✓
    |     OR
    |-- tcp_transport.write_tensor() → True       # TCP fallback
    v
(tensor now available for consumer)
```

**Consumer (PD instance):**
```
start_load_caches(encoder_cache)
    |
    v (for each mm_hash in metadata)
_load_one(mm_data):
    |-- shm_buffer.read_tensor() → tensor         # SHM path ✓
    |     OR (not found)
    |-- tcp_transport.pull_tensor(mm_hash, tcp_meta) → tensor  # TCP path ✓
    v
encoder_cache[mm_hash] = tensor
```

### 5.3 Lazy Initialization

All transports are initialized on first use:

- **SHM**: initialized on first `save_caches()` (producer) or `has_cache_item()` (consumer)
- **TCP**: initialized on first fallback trigger; if `mooncake` is not installed, logs an error and the entry is skipped (consumer falls back to local computation)

## 6. Configuration

### 6.1 CLI Arguments

```bash
# Producer (encoder instance) — SHM only
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --ec-connector MooncakeECConnector \
    --ec-role ec_producer \
    --ec-buffer-size 1073741824 \
    --ec-connector-extra-config '{"shm_name": "vllm_ec_buf"}'

# Producer — with TCP fallback enabled
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --ec-connector MooncakeECConnector \
    --ec-role ec_producer \
    --ec-buffer-size 1073741824 \
    --ec-connector-extra-config '{
        "shm_name": "vllm_ec_buf",
        "zmq_metadata_port": 14999,
        "staging_slots": 16,
        "staging_slot_size": 209715200
    }'

# Consumer (PD instance) — with TCP fallback
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --ec-connector MooncakeECConnector \
    --ec-role ec_consumer \
    --ec-buffer-size 1073741824 \
    --ec-connector-extra-config '{
        "shm_name": "vllm_ec_buf",
        "producer_host": "10.0.0.1",
        "zmq_metadata_port": 14999,
        "staging_slots": 16,
        "staging_slot_size": 209715200
    }'
```

### 6.2 Configuration Parameters

**SHM Transport:**

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| `ec_connector` | `ECTransferConfig.ec_connector` | — | Must be `"MooncakeECConnector"` |
| `ec_role` | `ECTransferConfig.ec_role` | — | `"ec_producer"`, `"ec_consumer"`, or `"ec_both"` |
| `ec_buffer_size` | `ECTransferConfig.ec_buffer_size` | `1e9` (1 GB) | Total SHM buffer size in bytes |
| `shm_name` | `ec_connector_extra_config` | `vllm_ec_{engine_id[:16]}` | POSIX SHM segment name |
| `max_slots` | `ec_connector_extra_config` | `64` | Max concurrent EC entries in SHM |

**TCP Transport (Fallback):**

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| `producer_host` | `ec_connector_extra_config` | `""` | Producer's hostname/IP; **required on consumer** when using TCP fallback |
| `zmq_metadata_port` | `ec_connector_extra_config` | `14999` | TCP port for ZMQ metadata REP/REQ; must match on both sides |
| `staging_slots` | `ec_connector_extra_config` | `16` | Number of slots in TCP staging buffer |
| `staging_slot_size` | `ec_connector_extra_config` | `209715200` (200 MB) | Max bytes per staging slot; must exceed largest expected EC tensor |

**Important:** `shm_name` and `ec_buffer_size` must match between producer and consumer. `zmq_metadata_port` and `staging_slot_size` must also match.

## 7. Error Handling and Edge Cases

### 7.1 SHM Buffer Full → TCP Fallback
When the SHM buffer cannot accommodate a new tensor:
- `write_tensor()` returns `False`
- Producer falls back to TCP staging (`MooncakeTCPTransport.write_tensor()`)
- Consumer's `has_cache_item()` checks both SHM and TCP

### 7.2 TCP Staging Full
When both SHM and TCP staging are exhausted:
- Producer logs a warning
- Consumer's `has_cache_item()` returns `False`
- Scheduler falls back to local encoder computation

### 7.3 Mooncake Not Installed
- `_ensure_tcp_transport()` raises `RuntimeError` with install instructions
- Caught at the call site; logs a warning
- SHM path continues to work unaffected

### 7.4 Stale SHM Cleanup
On producer startup, stale SHM from a previous run is detected and removed before creating a fresh segment.

### 7.5 Consumer Starts Before Producer
- `has_cache_item()` catches `FileNotFoundError` for SHM; ZMQ timeout for TCP
- Returns `False` in both cases → consumer computes encoder locally

### 7.6 ZMQ Metadata Timeout
The consumer ZMQ socket has a 5-second receive timeout (`RCVTIMEO`). On timeout, `has_entry()` returns `(False, {})` and the entry is skipped.

## 8. Performance Considerations

### 8.1 SHM Data Copy Path
```
Producer: GPU tensor → CPU copy → SHM memcpy
Consumer: SHM memcpy → CPU tensor → GPU copy
```
Total: 2 GPU-CPU copies + 2 SHM memcpys per tensor.

### 8.2 TCP Data Copy Path
```
Producer: GPU tensor → CPU copy → pinned staging buffer
Consumer: TE batch_transfer_sync_write (TCP) → pinned staging → GPU copy
```
Total: 1 GPU-CPU copy (producer) + 1 TCP transfer + 1 CPU-GPU copy (consumer).

### 8.3 Typical Tensor Sizes

| Input | Approx Tokens | Hidden Dim | Dtype | Size |
|-------|--------------|------------|-------|------|
| 224×224 image | ~576 | 4096 | fp16 | ~4.5 MB |
| 512×512 image | ~3072 | 4096 | fp16 | ~24 MB |
| 1024×1024 image | ~12288 | 4096 | fp16 | ~96 MB |

Default `staging_slot_size` of 200 MB covers up to 1024×1024 images. For larger tensors, increase `staging_slot_size` accordingly.

With a 1 GB SHM buffer, this supports ~10 concurrent 1024×1024 images or ~200 concurrent 224×224 images.

### 8.4 Future Optimizations
- **CUDA IPC / pinned memory:** Eliminate CPU intermediate copies for same-node deployments
- **RDMA backend:** For InfiniBand-equipped clusters, switch from TCP to RDMA in TransferEngine for higher bandwidth
- **Async transfers:** Use background threads for non-blocking GPU-CPU copies

## 9. Comparison with Other EC Connectors

| Feature | ECExampleConnector | MooncakeECConnector (SHM) | MooncakeECConnector (TCP) |
|---------|-------------------|--------------------------|--------------------------|
| Transport | Disk (safetensors) | POSIX shared memory | Mooncake TE over TCP |
| Latency | High (disk I/O) | Low (memory copy) | Medium (TCP transfer) |
| Multi-node | Yes (shared FS) | No (same-node only) | Yes (cross-node) |
| Dependencies | safetensors | None (stdlib) | `mooncake`, `zmq` |
| SHM-full fallback | N/A | N/A | Yes |
| Use case | Debug / experiment | Production (same-node) | Production (cross-node) |

## 10. File Structure

```
vllm/distributed/ec_transfer/ec_connector/
    base.py                        # ECConnectorBase (existing)
    example_connector.py           # ECExampleConnector (existing)
    factory.py                     # ECConnectorFactory (modified: registration)
    mooncake_ec_connector.py       # MooncakeECConnector (SHM + TCP fallback)
    mooncake_ec_shm_buffer.py      # ECSharedMemoryBuffer (SHM transport)
    mooncake_ec_tcp_transport.py   # MooncakeTCPTransport + StagingBuffer (TCP)
```
