# V1-B1 正式采集资源预算与配置冻结（v1）

## 1. 目的、边界与决定

本文将 B1 的资源配置冻结为后续小规模分层试运行（7.5）的执行基线。它依据两次已完成的
远端实测，而不是第 8.2 节的逻辑存储上界推算；不改变 B1 指标、Schema、固定数据集或模型
生成语义。

**本次批准范围：** 可在满足第 5 节预检与停止条件时，执行一次最多 10 条的单 GPU 串行
分层试运行。

**本次不批准：** 直接扩展到全部 100 条固定输入，或静默关闭 Token Top-K、缩小 Page
Top-K、降低 Decode 采样频率。全量运行需要以 7.5 的累计时间、磁盘和跨样本稳定性结果为
新的决策依据。

## 2. 证据与可追溯性

实测环境为 NousResearch/Meta-Llama-3.1-8B-Instruct 本地快照、`LlamaForCausalLM`、BF16、
Greedy（`do_sample=false,num_beams=1,use_cache=true`）、`sdpa` 和单张 A100 40 GiB；B1
与无 Probe Baseline 均在 `cuda:4` 执行。代码提交为
`7d9030db2ebd5310ba1486306449e5d0d4a05676`；远端 `dirty=true` 仅来自未跟踪的 `data/`，
没有已跟踪代码差异。固定集 Manifest 的 `data_sha256` 为
`30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee`。

| 运行 | 样本 | 结论 | 配置快照 SHA-256 |
|---|---|---|---|
| 7.2：8K 资源验证 | `ruler_8k_001`、`govreport_8k_002` | Baseline/B1 ID 完全一致；短与长 Decode 均完整 | `bc9b158ac807ab9e26cce10a0c85a7a7a5864815b6e6d61db625be3b15e15fcf` |
| 7.3：16K/32K 可行性 | `ruler_32k_001`、`govreport_16k_001`、`pg19_001_32k` | 三条均完成；29 个工件文件哈希、Step 连续性与 RULER 答案检查通过 | `85864855a900d0a89d12db665944da0476eedd2aa71c8ff1292fb9e5b9ec4629` |

运行目录均在远端 `data/runs/`；其中 7.3 的 `RULER` 离线检查、工件完整性检查及
`k=32, Δ=1` 双 Scope RCO 派生产物不得覆盖或修改原始 Trace。

## 3. 实测资源基线

`generation_end_to_end_seconds` 包围 `model.generate` 与末尾 CUDA 同步，**不含** B1
写盘；`forward` 计时为同一 CUDA 流上的 Event，范围仅是根模型 `forward`。实际文件字节数
不含运行级 Manifest。因而下表不能用于声称端到端服务延迟或物理带宽。

| 输入 | 生成 / Trace Step | B1 生成墙钟 | Prefill forward | Decode 总计 / p95 | 峰值 reserved | 逻辑 / 实际 Trace | 写盘墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RULER 8K | 7 / 6 | 6.17 s | 2.09 s | 3.92 s / 0.678 s | 17.27 GiB | 9.56 / 6.88 MiB | 0.69 s |
| GovReport 8K | 800 / 799 | 8.65 min | 1.40 s | 517.46 s / 0.650 s | 17.27 GiB | 347.61 / 178.80 MiB | 36.71 s |
| RULER 32K | 7 / 6 | 21.39 s | 7.59 s | 13.61 s / 2.296 s | 24.41 GiB | 9.56 / 7.45 MiB | 1.13 s |
| GovReport 16K | 1024 / 1023 | 20.44 min | 3.08 s | 1223.17 s / 1.248 s | 24.41 GiB | 443.10 / 234.79 MiB | 47.50 s |
| PG19 32K | 941 / 940 | 36.54 min | 6.98 s | 2185.20 s / 2.331 s | 27.03 GiB | 407.71 / 217.91 MiB | 47.67 s |

GiB/MiB 由记录的字节数以二进制单位换算，并按两位小数展示；预算判断使用原始字节数。
长上下文的主要成本是逐 Step 的在线观察与写盘，而非 Prefill。单条 32K、近千步 Decode
能够完成，但不构成全量资源可行性的证明。

## 4. 冻结的 B1 执行配置

以下配置适用于 7.5；任何变更均须创建新的配置快照/Hash 和新的运行目录，而不是混入本
轮结果。

```yaml
schema_version: headkv-v1-b1-attention-v1
fixed_dataset:
  prompts: data/fixed_v2/prompts.jsonl
  manifest: data/fixed_v2/manifest.json
  required_data_sha256: 30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee
model:
  revision: 7d9030db2ebd5310ba1486306449e5d0d4a05676
  dtype: bfloat16
  attention_backend: sdpa
generation:
  do_sample: false
  num_beams: 1
  use_cache: true
  seed: 0
collection:
  prefill_window: 64
  key_block_size: 1024
  token_topk: 64
  save_token_topk: true
  page_size: 16
  page_topk: 32
  page_scopes: [all_visible, prompt_only]
  decode_chunk_steps: 128
storage:
  save_full_attention: false
  save_full_kv_cache: false
  save_generated_ids: true
```

每个样本先运行同配置 Baseline，再运行 B1，并要求生成 ID 完全一致。计时方法必须为
`cuda_event`；成功样本必须满足 `Prefill forward=1`、
`Decode forward=generated_length-1`、Trace Step 从 0 连续至末尾。仅以相同配置的工件做
资源比较，禁止把 `flash_attention_2`、不同模型快照或不同 Token Top-K 设置并入本冻结组。

## 5. 7.5 批次预算、预检与停止条件

### 5.1 分层试运行输入

本批共 10 条，目的在于覆盖数据集与长度，而不是给出数据集平均性能或质量结论：

| 组别 | 固定样本 |
|---|---|
| RULER 8K / 32K | `ruler_8k_001`、`ruler_32k_001` |
| GovReport 8K | `govreport_8k_001`、`govreport_8k_002` |
| GovReport 16K | `govreport_16k_001`、`govreport_16k_002` |
| PG19 8K | `pg19_001_8k`、`pg19_003_8k` |
| PG19 32K | `pg19_001_32k`、`pg19_003_32k` |

PG19 的 `001_8k` 自然 EOS 较早，`003_8k` 与 `003_32k` 已有高重复告警；保留它们是为了
验证真实异常样本的 Trace 与资源路径。它们不能单独支持“高质量长续写”的结论。

### 5.2 预检

1. 确认目标 GPU 没有其他计算进程，且可用显存不少于 32 GiB；实际 GPU 编号写入元数据。
2. 确认运行盘可用空间不少于 200 GiB；每次运行使用新的 `run_id`，不覆盖 7.2/7.3 工件。
3. 校验代码提交、冻结集 Manifest、模型快照、`sdpa`、BF16 与第 4 节配置；Baseline 完成并
   通过后才启动同批 B1。

### 5.3 批次上限和审查触发

本批采取单 GPU 串行执行，资源预留为 **不超过 5 GPU 小时、5 GiB 实际 Trace 文件**。
这是含 PG19 8K 未测变异的保守运行上限，不是全量预测。单条样本出现任一情形时，保留已
完成的工件、停止**后续**样本并审查，不以截断当前生成伪造完成结果：

- `peak_memory_reserved_bytes > 32 GiB`，或发生 OOM；
- 实际 Trace 文件超过 512 MiB，或逻辑数组超过 1 GiB；
- 写盘墙钟超过 90 s，或任何 Decode 分片/Manifest/Step 连续性校验失败；
- Baseline/B1 generated IDs 不一致、forward 数量契约不满足、RULER 答案不匹配；
- 自批次开始累计 B1 墙钟超过 5 GPU 小时，或运行盘剩余空间低于 200 GiB。

达到自然 EOS、`max_new_tokens` 上限或 PG19 重复告警本身不是运行失败；它们分别作为
可观察长度和任务质量诊断记录。不得为满足长度而改变生成参数。

## 6. 7.5 后的决策门

7.5 完成后单独汇总各长度/任务的成功率、Decode 长度、资源分位数、累计磁盘量、RCO
有效对数和 PG19 诊断。只有以下条件同时成立，才可以提出全量 100 条的运行提案：

1. 10 条均遵守第 4 节配置且 B1/ Baseline ID 完全一致；
2. 没有触发第 5.3 节的资源或完整性审查；
3. 10 条中至少一条 GovReport/PG19 长 Decode 提供足够的有效 RCO 对；
4. 根据实测分层成本，单独给出全量运行的 GPU 时间、磁盘空间、批次拆分、失败恢复和
   产物保留方案，并经审阅批准。

在此之前，本报告仅冻结执行边界，不冻结“稳定 Head”类别、PG19 质量或硬件性能结论。
