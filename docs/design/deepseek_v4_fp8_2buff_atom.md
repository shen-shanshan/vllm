# DeepSeek-V4 fp8 2-Buffer KV Cache（ATOM op4/op5 移植）设计文档

> 状态：实现完成（本机编译 + 纯逻辑校验通过），GPU 端首跑验证由使用者执行。
> 环境变量：`VLLM_ROCM_USE_AITER_DSV4_FP8`（默认关闭）

## 1. 背景与目标

本改动将 ATOM（AMD 推理引擎）中 DeepSeek-V4 的 **fp8 2-Buffer KV Cache 方案**移植到本 vLLM fork：
以 aiter 的 **op4（prefill）/ op5（decode）融合注意力 kernel** 替换现有 Triton 稀疏注意力路径，
并将 KV cache 布局从单缓冲 584B/token 改为 **双缓冲布局**（fp8 NoPE 池 + 并行 bf16 RoPE 池）。

**移植的优化**（来自 ATOM）：

| 优化 | aiter 符号 | 说明 |
|------|-----------|------|
| op4 prefill | `aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus` | 直接消费 fp8 前缀池 + op 量化 Q/K，无 dequant、无 torch quant |
| op5 decode | `aiter.mla.mla_decode_fwd_v4_nm` | 融合 sparse decode（page_size=1 CSR 寻址），支持 split-K |
| 融合 Q/K norm+rope+quant | `aiter.ops.fused_qk_norm_rope_cache_quant.fused_qk_norm_rope_group_quant` | 一次 launch 完成 Q 无权重 RMSNorm + KV 权重 RMSNorm + GPT-J RoPE + e8m0 每 64-tile 量化 |
| 2-Buffer 布局 | — | RoPE 永不过量化（精度优于 584B 方案），NoPE 行内联重复 e8m0 scale |
| GQA pad-to-16 | — | asm kernel 仅支持 gqa ∈ {16,32,64,128}；TP8=8 头 → pad 16 后裁掉 |
| SWA 环写 | Triton scatter | 每请求独立 ring 区，行 = `slot × R + pos % R` |

**明确不做**：PCP/DCP/PD 分离、prefix caching（新路径）、dense 层（保留旧路径）。

## 2. 总体设计

### 2.1 侵入面控制

- **新增 1 个文件**：`vllm/models/deepseek_v4/amd/fp8_2buff.py`（全部新代码）
- **修改 3 个文件**：`amd/rocm.py`（ROCm-only）、`compressor.py`（+1 个形状门控分支）、`envs.py`（+1 个环境变量）
- **CUDA 路径零影响**：`amd/` 模块由 `deepseek_v4/__init__.py` 平台选择保证 CUDA/xpu 永不导入；
  compressor.py 的分支为形状门控（`head_dim==512 && uint8 && ndim==3 && shape[-1]!=584`），
  CUDA 的 584B/bf16 布局永不命中，且 AMD 模块为分支内懒导入。

### 2.2 KV Cache 布局：单层扁平字节池（region-contiguous）

每层 **一个** uint8 扁平池（`V4AtomFp82BuffSpec(MLAAttentionSpec)`，定义于 fp8_2buff.py），
region 顺序为 `swa_nope | main_nope | swa_rope | main_rope`：

- **NoPE 平面行 = 512B**：448B fp8 NoPE + 14B 重复 e8m0 scale（7 tile × 2）+ 50B pad
- **RoPE 平面行 = 128B**：64 × bf16（永不过量化）
- 每个平面视图**密集连续**（aiter asm kernel 的硬性要求）；`swa_nope|main_nope` 相邻构成
  op4/op5 消费的统一 `unified_nope` 行空间（同样 `unified_rope`）
- `head_size` = 640B + SWA 环摊还（`ceil(ring_bytes / (min_blocks × block_size))`，
  移植 ATOM `_proxy_page_bytes` 数学）：`num_blocks ≥ min_blocks` 时池可容纳完整环 + 全部 paged 行
- vLLM block manager 只管理 main 区域 block 的分配回收；block b → 行
  `b × rows_per_block`（`rows_per_block = block_size // compress_ratio`，与 584B 布局同几何）

### 2.3 SWA 环（ring）

- 每层环区 = `max_num_seqs × (window + spec_steps)` 行，位于池头部，**不走 vLLM block 管理**
- 环行 = `slot × R + pos % R`（`R = window + spec_steps`，spec 步宽保证 draft 不别名主 token 的读窗）
- **slot 分配**：key = 请求首 block id（`block_table[:,0]`）——驻留期间稳定、并发请求间唯一
  （本路径无 prefix caching 故无共享）；请求离开即回收。**槽回收安全性**由不变量保证：
  每次 prefill 写尾部 `min(chunk_len, R)` 个 token，下一 chunk 读窗起点
  `chunk_start - window + 1 ≥ chunk_start - R`，故可读的行必已被本请求重写
- 写路径：decode 在**注意力前** scatter 全部 `1+k` 个 token（MTP draft 需读同一步写入的行）；
  prefill 在**注意力后** scatter chunk 尾部（避免覆盖 prefill 自身仍在读的环行）

### 2.4 索引翻译

- **环行/prefill-extend 索引**：builder 内 CPU numpy 构建 + H2D 持久 buffer
  （builder 在 capture 外运行，vLLM v1 惯例）
- **压缩行（CSA topk / HCA）索引**：复用 fork 现有
  `compute_global_topk_ragged_indices_and_indptr`（rocm.py，in-forward GPU 翻译，
  topk buffer 只在 forward 时写入），池行 = `ring_rows + 压缩行索引`
- **合并**：op4/op5 各消费一条 ragged 流 `[comp | ring]`；
  **merged indptr = comp_indptr + ring_indptr 的 GPU 逐元素相加**——完全消除
  host 公式与 GPU lens 的长度同步风险；索引合并由一个 64 宽 masked Triton kernel 完成
- prefill 的 prefix/extend 流按 decode-first 全 token 构建后在 forward 中 rebase 到
  `[0, npref)`（`indptr[num_decode_tokens:...] - indptr[num_decode_tokens]`）

### 2.5 Compressor 写入

fork 的 fused compress kernel（`_fused_kv_compress_norm_rope_insert_sparse_attn`）逐行移植为
`_fused_kv_compress_norm_rope_insert_2buff`，仅改最终 store 段：

- 448B fp8 NoPE → `main_nope` 行 [0,448)
- 7 个 e8m0 scale 复制 2 份内联 → 行 [448,462)，50B 零 pad → [462,512)
- 64 bf16 RoPE → `main_rope` 行
- 行 = `block_id × rows_per_block + pos_in_block // compress_ratio`
- **scale 语义与 584B 路径逐位一致**（`2^ceil(log2(amax/448))`、bias 127、per-64-tile），
  仅存储布局不同；量化值本身直接复用
- 分派：compressor.py 的 store 选择处新增形状门控分支（见 2.1），两阶段 split 路径在
  2buff 池上回退为单 pass（正确，split 优化列为后续工作）

### 2.6 关键简化：eager 断点

fork 的 `attention_impl` 带 `@eager_break_during_capture`（attention.py:466）——2buff 前向
全程运行在 cudagraph 的 eager 断点内，**动态形状、动态切片、D2H 全部合法**，
无需任何 cudagraph 形状烘焙体操（持久 buffer 仅用于性能）。

### 2.7 MTP / Spec Decode

- decode 索引按 per-token 绝对位置构建（draft 位置 = `main_pos + t`，由 vLLM positions 提供）
- decode ring scatter 覆盖 `1+k` 个 token（`write_per_batch = min(max_decode_query_len, R)`），
  draft 的窗口自然包含主 token 与先前的 draft 行
- comp 长度语义与 fork 现有 decode 路径完全一致（同一 helper、同一 is_valid 输入）

## 3. 详细改动内容

### 3.1 新增 `vllm/models/deepseek_v4/amd/fp8_2buff.py`（约 1100 行）

| 区域 | 内容 |
|------|------|
| 布局常量 | `V4_DIM_NOPE=448`、`V4_DIM_ROPE=64`、`V4_DIM_SCALE_DUP=14`、`V4_NOPE_ROW_BYTES=512`、`V4_ROPE_ROW_BYTES=128`、`V4_ENTRY_BYTES=640`（对照 ATOM `v4_quant.py`） |
| 门控 | `atom2buff_available(kv_cache_dtype)`：dtype 必须为 `fp8_ds_mla` + env 开启 + gfx950/gfx1250 + aiter 三符号可导入（失败一次性告警并回退） |
| 尺寸数学 | `atom2buff_ring_rows` / `atom2buff_head_size` / `v4_atom2buff_*_from_config`（纯函数，可单测） |
| Spec | `V4AtomFp82BuffSpec(MLAAttentionSpec)`：override `real_page_size_bytes = block_size × head_size`；MRO 自动解析注册，零 registry 改动 |
| 池切片 | `slice_atom2buff_pool_views` → `Atom2BuffPoolViews`（`unified_nope/ring_nope/main_nope/unified_rope/...` 密集视图；按 data_ptr 幂等缓存于 layer） |
| 量化包装 | `rocm_fp8_2buff_qk_norm_rope_quant`：cos/sin 取 fork rotary cache `[max_pos, 2·rd]` 的 per-pair 偶下标（`[:, 0:rd:2]` / `[:, rd:2·rd:2]`） |
| op4/op5 包装 | `rocm_fp8_2buff_prefill` / `rocm_fp8_2buff_decode`（GQA pad16 用预分配 scratch、N<T_pad 裁剪、split-K） |
| Split-K 启发式 | `kv_splits_heuristic`（ATOM `_kv_splits_heuristic` 移植：仅依赖捕获期标量，prev_pow2 下取整，上限 64） |
| 环写 | `swa_ring_scatter_2buff` + Triton kernel（纯字节 scatter，grid = (bs, write_per_batch)，行内 `w ≥ tok_n` 早退） |
| Compressor store | `_fused_kv_compress_norm_rope_insert_2buff` + `compress_norm_rope_store_2buff` launcher（planes 经 metadata 传入） |
| 索引纯函数 | `build_ring_indices_cpu` / `build_extend_indices_cpu` / `ragged_from_lists`（尾重复 padding）/ `merge_ragged_indices` + Triton merge kernel |
| Slot 分配 | `V4RingSlotAllocator`（host dict，key = 首 block id；耗尽即报错） |
| 拒绝检查 | `atom2buff_reject_prefix_caching` |

### 3.2 修改 `vllm/models/deepseek_v4/amd/rocm.py`（+612 行）

- `DeepseekV4ROCMAiterMLAAttention.__init__`：门控（env + 架构 + aiter 导入，仅
  `compress_ratio > 1` 层）→ `self._atom_2buff`；prefix-caching 拒绝；环几何
  （`ring_rows`/`ring_slots`/`topk_tokens`/`max_comp`）；`backend_cls` 切换到新 backend
- `get_kv_cache_spec` 覆盖：2buff 开 → `v4_atom2buff_spec`；关 → 基类（dense 层恒走旧路径）
- `_atom2buff_pool_views`：按 data_ptr 幂等的池视图缓存
- `_fused_qnorm_rope_kv_insert` 覆盖：2buff → aiter quant + 存 `_qkn_2buff` +
  给 metadata 挂 views + decode 注意力前环写；非 dict（profile run）→ 基类
- `forward_mqa`：2buff 分派到 `_forward_2buff`（decode/prefill 拆分沿用 swa metadata）
- `_forward_2buff_decode`：comp 翻译（fork helper，ratio-4 用 `topk_indices_buffer`、
  ratio-128 用 base builder 的 ragged 输出）+ `ring_rows` 偏移 → GPU indptr 相加 →
  merge → op5（pad scratch + split-K）
- `_forward_2buff_prefill`：comp 翻译 + prefix/extend rebase → merge → op4 →
  注意力后环写（`write_per_batch = min(max_prefill_query_len, R)`）
- 新增 `DeepseekV4Atom2BuffFp8Metadata` dataclass 与完整 builder：
  CPU 环/prefix/extend 索引构建 + 持久 GPU buffer 暂存 + `qo_indptr = arange(N+1)` +
  slot 分配；`DeepseekV4Atom2BuffFp8Backend`（复用既有 backend name，无 registry 改动，
  `get_kv_cache_shape` 返回扁平 `(num_blocks, block_size, head_size)`）

### 3.3 修改 `vllm/models/deepseek_v4/compressor.py`（+23 行）

store 分发处新增首个分支（形状门控 + 懒导入，见 2.1）；其余路径（CUDA cutedsl、两阶段、
indexer、bf16）零改动。

### 3.4 修改 `vllm/envs.py`（+7 行）

`VLLM_ROCM_USE_AITER_DSV4_FP8`（默认 False），类型化段与 lambda 段各一处。

### 3.5 修改 `docker/Dockerfile.rocm_base_gfx1250`（+16 行）

- 新增 `ARG AITER_COMMIT=""`（可选 commit pin，build_aiter 阶段非空时 checkout）
- 构建期校验：安装 aiter wheel 后 import 三符号并断言 `mla_decode_fwd_v4_nm` 存在，
  缺失即构建失败

### 3.6 新增 `tests/v1/attention/test_rocm_aiter_mla_fp8_2buff.py`

12 个 CPU-safe 用例：门控（env/dtype/架构/import 回退）、环行与 head_size 数学、
spec 字段与 registry 解析、池切片字节布局（密集 stride、region 顺序、bf16 解码）、
slot 分配器、decode/prefix/extend 索引（含 wrap 取模）、MTP draft 窗口、ragged padding、
split-K 启发式、prefix-caching 拒绝。

## 4. 验证配置

### 4.1 构建

```bash
# aiter 三符号缺失时构建直接失败；验证通过后固化 commit：
docker build -f docker/Dockerfile.rocm_base_gfx1250 \
  --build-arg AITER_COMMIT=<验证通过的 commit> ...
```

### 4.2 单测（本机已通过的纯逻辑校验 + ROCm 环境 pytest）

```bash
# 本机（无 GPU/torch）：numpy 桩 harness 对真实源码 44 项校验
python /tmp/p1_2buff_check.py

# ROCm 环境：
pytest tests/v1/attention/test_rocm_aiter_mla_fp8_2buff.py -v
pytest tests/v1/attention/test_rocm_aiter_mla_mtp_split.py \
      tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py \
      tests/v1/attention/test_sparse_mla_backends.py \
      tests/v1/attention/test_mla_backends.py \
      tests/v1/attention/test_attention_backends_selection.py \
      tests/v1/attention/test_rocm_attention_backends_selection.py
```

### 4.3 首跑（建议顺序）

```bash
# 1. 启用新路径（注意必须关闭 prefix caching）：
VLLM_ROCM_USE_AITER_DSV4_FP8=1 vllm serve <model> \
  --kv-cache-dtype fp8 --no-enable-prefix-caching \
  --tensor-parallel-size 8            # 或 DP8+EP：--data-parallel-size 8 --enable-expert-parallel

# 2. 重点对拍项（精度异常时优先检查）：
#    a. aiter 三符号真实签名 vs ATOM recipes/DeepSeek-V4.md 的 aiter 版本
#    b. cos_sin per-pair 偶下标提取（rocm_fp8_2buff_qk_norm_rope_quant）
#    c. compressor store 的 scale/e4nv 有限值编码（公式已确认逐位一致，
#       仅 fp8 位布局需与 ATOM 参考对拍）
```

### 4.4 精度与性能

```bash
# 精度：GSM8K lm_eval，TP8 与 DP8+EP 各一次，
# 对比 bf16 路径与现有 fp8_ds_mla（584B Triton）路径
# 性能 A/B：同模型同 batch，新路径 vs 584B Triton 基线
```

## 5. 使用约束

| 约束 | 说明 |
|------|------|
| 架构 | 仅 **gfx950 / gfx1250**（aiter op4/op5 的发布范围）；其他架构自动告警回退 584B 路径 |
| KV dtype | 必须 fp8 系（复用 `fp8_ds_mla` 解析字符串 → uint8）；bf16 请求走旧路径 |
| Prefix caching | **不支持**：与 `VLLM_ROCM_USE_AITER_DSV4_FP8=1` 同时启用时启动即报错（`ValueError`） |
| 并行模式 | 仅 **TP8 / DP8+EP**；不支持 PCP/DCP/PD 分离 |
| 层覆盖 | 仅 `compress_ratio > 1`（CSA/HCA）层走新路径；dense 层保持旧 SWA 路径 |
| MTP | 支持（decode 路径按 per-token 位置构建）；draft 语义与 ATOM 一致 |
| 显存 | 640B/token（比 584B 多 ~11%，ATOM 平价）；SWA/indexer cache 保留分配但新路径下闲置（~1GB/61 层） |
| 回退 | env 关闭 / 架构不符 / aiter 缺失任一条件 → 完全回退现有 Triton + 584B 路径（零行为变化） |
| aiter 版本 | 需包含 op4/op5 三符号（以 `recipes/DeepSeek-V4.md` 的 ATOM 验证版本为准；docker 构建期强校验） |

## 6. 已知限制与后续工作

1. **GPU 首跑验证**（使用者执行）：aiter 签名 / cos_sin / scale 编码三项对拍 + GSM8K + perf A/B
2. **decode 融合环写**：当前 decode 环写为独立 scatter kernel；ATOM 将其融合进
   `fused_qk_norm_rope_group_quant`（`swa_plane` + `swa_dest_rows`），可作后续 perf 优化
3. **两阶段 split compressor**：ratio-128 prefill 在 2buff 池上回退为单 pass；
   split-occupancy 版本可后续移植
4. **索引构建 CPU 化**：当前环/prefix/extend 索引为 builder 内 numpy 构建 + H2D；
   ATOM 的 fused GPU 索引 kernel（`write_v4_paged_*_indices`）可后续移植消除 D2H
5. **首 block id 的 D2H**：slot 分配每步做一次 `block_table[:,0]` 的 D2H（微小同步）；
   ATOM 的 req_ids passthrough（运行时 wrap `GPUModelRunner`）可作后续优化
6. **prefix caching**：与 ATOM 插件模式持平（不支持）；如需支持需设计环区校验/重建机制

## 7. 移植参照（ATOM 源文件）

- `atom/plugin/vllm/deepseek_v4_bridge.py`：尺寸/切片数学、索引翻译语义、槽分配
- `atom/model_ops/v4_kernels/v4_quant.py`：布局常量与量化约定
- `atom/model_ops/v4_kernels/paged_prefill.py` / `paged_decode.py`：op4/op5 调用点与 GQA pad、split-K
- `atom/model_ops/v4_kernels/qk_norm_rope_maybe_quant.py`：quant kernel 参数契约
- `atom/model_ops/v4_kernels/state_writes.py`：环写语义（`swa_write_2buff_prepacked`）
