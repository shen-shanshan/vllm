# ROCm Sparse Indexer GPU Memory Fault 排查报告

## 1. 问题背景

在 8 张 gfx950 GPU 上以 TP8 部署 DeepSeek V4，并启用 ROCm AITER
sparse attention 时，高并发的混合 prefill/decode 负载会触发 GPU memory
access fault。故障会导致 worker 退出、请求失败以及服务健康检查异常。

现象与显式设置 `max_model_len` 有明显相关性，但并不局限于某一个长度；
切换 CUDA Graph 模式也不能消除故障。未显式设置 `max_model_len` 时测试可以
完成，说明该参数会改变执行形状、kernel 调度或内存布局，从而影响故障是否
暴露，但它本身不是越界地址的直接来源。

原始问题见 [Fangzhou-Ai/vllm#13](https://github.com/Fangzhou-Ai/vllm/issues/13)。
早期修复曾修改通用的 `sampler.cu` top-k kernel，以初始化未写满的 shared
memory 输出槽位。该修改能够阻止崩溃，但进一步排查表明，top-k 只是放大了
上游 ROCm 计算产生的异常值，并不是最先产生错误数据的位置。

## 2. 根因与故障链路

### 2.1 已确认的直接根因

ROCm sparse indexer 的 decode 路径调用 AITER
`deepgemm_fp8_paged_mqa_logits` 后，AITER 在 gfx950 上会间歇性地在有效
logits 范围内产生 `NaN` 或 `±Inf`。

通用 histogram top-k kernel 默认输入 logits 是可比较、可排序的有限值。
当有效范围包含 `NaN` 时，部分候选无法正常参与比较，top-k 的 shared-memory
临时输出可能没有填满。kernel 随后仍将所有槽位复制到全局
`topk_indices_buffer`，因此未初始化内容会表现为随机的巨大正索引或小于
`-1` 的负值。

下游 sparse-attention decode 路径把非负索引视为有效 token 索引，并据此
计算 block-table 位置。巨大随机索引最终转化为越界 block-table 或 KV-cache
访问，从而触发 ROCm GPU memory access fault。

完整故障链路如下：

```text
AITER gfx950 paged-MQA 产生 NaN/Inf logits
  → histogram top-k 无法填满输出槽位
  → 未初始化的 shared-memory 索引被复制到全局 buffer
  → sparse attention 将巨大正索引当作有效索引
  → block-table/KV-cache 越界访问
  → GPU memory access fault
```

### 2.2 排除的方向

逐层采样与参考计算得到以下证据：

- 异常批次中，`q`、`weights`、对应的 FP8 KV-cache 数据及 scale 均为有限值；
- KV scale 为正常值，block table 指向的物理块也在已分配 cache block
  范围内；
- 使用相同的 `q`、KV、scale 和 weights 做直接 FP32 参考计算，结果全部有限；
- AITER 对相同位置输出 `NaN`，因此异常首先出现在
  `deepgemm_fp8_paged_mqa_logits` 的输出；
- 异常行的 `row_end`、`seq_lens`、`next_n` 和 `num_rows` 一致，排除了行
  边界计算错误；
- 捕获时 `hidden_states` 行数、`num_padded_tokens` 及输出 view 范围一致，
  排除了 graph padding/view 越界；
- workspace 在调用前已填充为 `-Inf`，异常位置却变成 `NaN`，因此不是简单的
  workspace 漏写；
- 切换 graph 模式仍可复现，说明 CUDA Graph 不是根因。

### 2.3 当前结论的边界

已经能够确定 vLLM 内部的首个坏数据来自 AITER 的 gfx950 paged-MQA kernel，
而不是 KV-cache、metadata 或通用 top-k 自身主动生成 `NaN`。但目前尚未将
AITER 内部问题继续定位到某一条指令、特定 tile 配置或竞争条件。因此，本次
修改修复的是 vLLM 与外部 AITER kernel 之间缺少有限值校验的接口问题，并
阻断崩溃链路；它不代表 AITER 内部的数值问题已经被修复。

## 3. 修复方案

修复放在 ROCm AITER paged-MQA 输出和通用 top-k kernel 之间，不再修改
`csrc/libtorch_stable/sampler.cu`。

在 decode logits 传入 `top_k_per_row_decode` 前，使用原地
`nan_to_num_` 建立“top-k 输入必须为有限值”的契约：

- `NaN` 映射为该 dtype 的有限最小值；
- `-Inf` 映射为有限最小值；
- `+Inf` 映射为有限最大值。

这样可以保留正负无穷的排序方向，同时确保 histogram top-k 的每个输入都
可比较。top-k 使用 `seq_lens` 限定每行有效区，因此将有效区之外原有的
`-Inf` 替换为有限最小值不会扩大可选范围。

修复位置仅属于 ROCm sparse indexer decode 路径，对 NVIDIA 路径和通用
sampler kernel 没有影响。

验证过程中保留了异常检测 instrumentation。修复后仍能观察到 AITER 原始
输出中的 `NaN/Inf`，但归一化后的有效 logits 全部有限，top-k 不再返回越界
索引。清理 instrumentation 后，三轮并发测试均完成且未再出现 GPU memory
access fault。

## 4. 同类问题的定位方法

### 4.1 先找首个错误数据，而不是只看首次崩溃位置

GPU 异步执行会使报错位置明显滞后于真正出错的 kernel。`HIP_LAUNCH_BLOCKING=1`
可以缩短范围，但同步边界只能说明此前某个操作出错。应沿数据流向上游逐层
检查：

1. 在消费者使用索引前检查范围；
2. 在生产者返回后立即检查；
3. 继续检查生产者的输入和 metadata；
4. 找到“输入正常、输出首次异常”的最小边界。

本问题中，memory fault 出现在 sparse attention，但非法索引在 top-k 后已
存在，而非有限 logits 在 AITER paged-MQA 后已经出现。

### 4.2 同时验证数据、形状和元数据

对索引类 GPU 故障，建议至少记录：

- tensor 的 shape、stride、dtype、device 和有效 view 范围；
- batch size、实际 token 数、padding token 数及 speculative decode 宽度；
- 每行 `seq_len`、row start/end 和 top-k 大小；
- block table 的最小值、最大值及物理 cache block 总数；
- 输入输出的 `NaN`、`+Inf`、`-Inf` 数量；
- 输出索引中 `< -1` 或 `>= seq_len` 的数量及样例。

只初始化全局输出 buffer 并不足以证明生产者正确，因为 kernel 内部的
shared-memory 临时区可能在结束时重新覆盖全局 buffer。

### 4.3 使用同输入参考实现做差分

保留触发异常时的输入，用更简单、更高精度的参考实现计算相同位置：

- 如果参考实现也异常，继续检查输入、量化数据和 scale；
- 如果参考实现有限而优化 kernel 输出异常，问题可收敛到优化 kernel、
  kernel 配置或其 workspace；
- 如果两者均有限但消费者失败，应检查索引契约、shape 和生命周期。

本次 FP32 参考计算是区分 KV-cache 污染与 AITER kernel 输出污染的关键证据。

### 4.4 用可证伪假设组织 instrumentation

每轮日志应针对少量明确假设，并只在首批调用或检测到异常时输出，避免海量
日志改变时序或淹没证据。例如：

- logits 是否先出现非有限值；
- top-k 是否返回行范围外索引；
- block table 是否越界；
- workspace 是漏写、被覆盖还是计算出 `NaN`；
- graph padding 是否导致输入输出 view 不一致。

每个假设都应有明确的确认或排除条件。得到结论后应移除诊断代码，只保留
最小修复。

### 4.5 在正确的接口边界实施防护

修复应尽量靠近违反契约的生产者，同时避免修改跨平台通用 kernel：

- 外部 ROCm kernel 输出不满足下游有限值契约时，在 ROCm 调用侧规范化；
- 不要仅在最终消费者处做索引 bounds check，因为随机但碰巧落在范围内的
  索引仍可能静默破坏模型结果；
- 防护只能阻断故障链路时，要明确记录上游缺陷仍待修复，并推动 AITER
  侧进一步定位。

这种方式既降低对 NVIDIA 共用代码的回归风险，也让 workaround 的平台范围
和责任边界保持清晰。
