# V1-B1 全量 100 条运行计划与运行集合接口（v1）

## 1. 状态、目的与边界

**状态：待审核，未获执行授权。** 本文将已通过的 10 条分层试运行结果转化为全量 B1 的
执行提案：规定批次划分、资源预留、失败恢复和多个独立运行如何组成一个可审计的结果集合。
它不修改 B1 指标、单运行文件 Schema、模型生成语义或固定输入，也不批准启动任何模型。

单运行的指标和文件契约以
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md)
为准；10 条试运行的配置与证据以
[《V1-B1 正式采集资源预算与配置冻结》](v1_b1_resource_budget_v1.md) 为准。本文件只增加
“全量执行编排层”，不得用它替代前两者。

## 2. 依据与适用对象

适用输入为 `data/fixed_v2/prompts.jsonl` 中全部 100 条固定 Prompt，Manifest SHA-256 必须为：

```text
30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee
```

试运行表明：10 条均完成，B1/ Baseline 生成 ID 一致，完整性、RULER 与一条 32K 长 Decode
双 Scope RCO 均通过。该证据允许提出本计划，但不证明未采样的 90 条具有相同生成长度、
资源或任务质量。

全量运行必须保持 BF16、Greedy、`sdpa`、Batch=1、标准连续 KV Cache、`prefill_window=64`、
`key_block_size=1024`、Token Top-K=64、双 Scope Page Top-K=32、Page size=16 和最多 128 Step
Decode 分片。禁止为了全量成本静默切换到 `flash_attention_2`、降低 Page/Token Top-K 或改变
Decode 采样频率。

执行前由预检计划写入精确模型快照、代码提交、配置 Hash、设备与数据 Hash；本文不预先填写
未来运行的代码提交，避免把审核后的文档提交误当作运行代码版本。

## 3. 批次划分与实测外推

每个批次均为单 GPU 串行运行，先执行同配置 Baseline，再执行 B1。样本以冻结 `sample_id`
显式指定，不依赖运行时随机抽样。

| 批次 ID | 固定样本 | 数量 | 试运行上界外推的单批时间 | 原始 Trace 上界 |
|---|---|---:|---:|---:|
| `ruler_8k_b01` | `ruler_8k_001`–`ruler_8k_020` | 20 | 0.04 h | 0.14 GiB |
| `ruler_32k_b01` | `ruler_32k_001`–`ruler_32k_020` | 20 | 0.12 h | 0.15 GiB |
| `govreport_8k_b01` | `govreport_8k_001`–`govreport_8k_010` | 10 | 1.6 h | 1.75 GiB |
| `govreport_8k_b02` | `govreport_8k_011`–`govreport_8k_020` | 10 | 1.6 h | 1.75 GiB |
| `govreport_16k_b01` | `govreport_16k_001`–`govreport_16k_010` | 10 | 3.6 h | 2.31 GiB |
| `govreport_16k_b02` | `govreport_16k_011`–`govreport_16k_020` | 10 | 3.6 h | 2.31 GiB |
| `pg19_8k_b01` | `pg19_001_8k`,`pg19_003_8k`,`pg19_005_8k`,`pg19_007_8k`,`pg19_009_8k` | 5 | 0.8 h | 0.76 GiB |
| `pg19_8k_b02` | `pg19_011_8k`,`pg19_013_8k`,`pg19_015_8k`,`pg19_017_8k`,`pg19_019_8k` | 5 | 0.8 h | 0.76 GiB |
| `pg19_32k_b01` | `pg19_001_32k`,`pg19_003_32k`,`pg19_005_32k`,`pg19_007_32k`,`pg19_009_32k` | 5 | 3.6 h | 1.16 GiB |
| `pg19_32k_b02` | `pg19_011_32k`,`pg19_013_32k`,`pg19_015_32k`,`pg19_017_32k`,`pg19_019_32k` | 5 | 3.6 h | 1.16 GiB |

表中时间包含已测的 Baseline、B1 生成和 Trace 写盘，但不含排队时间；它使用每类已观测到的
较高单样本成本外推。10 批累计约 19.3 GPU 小时，因此申请 **24 单卡 GPU 小时**作为执行预算，
其中余量只用于模型加载、环境波动和一次受控恢复，不代表允许静默重跑。

## 4. 空间、预检与停止条件

全量原始 Trace 以实测类别上界估计约 12.3 GiB。若当前 10 条试运行 Trace 暂时并存，原始
Trace 合计约 13.8 GiB；全量多尺度 Page RCO JSON 按当前记录密度估计约 0.5 GiB，其他
Manifest、Baseline、核验报告和日志通常远小于 0.1 GiB。因此长期保留按 **16 GiB** 规划，
运行期为原子写入、派生报告和输入长度差异预留 **40 GiB**；启动任一批前，运行挂载点可用
空间仍必须不少于 **200 GiB**。

每批启动前必须完成：

1. 生成新的、不可覆盖的计划 JSON，并验证固定集/Manifest、模型快照、BF16、`sdpa`、生成
   配置、采集配置和精确代码提交；
2. 检查目标 GPU 无其他计算进程且可用显存不少于 32 GiB；检查输出根位于研究数据挂载点，
   而非本地根分区、`/tmp` 或 `/var/tmp`；
3. 为 Baseline、B1、日志和派生报告使用新的 `run_id`，不得覆盖 7.5 原始工件；
4. 将第 7.5 节的单样本阈值原样带入：32 GiB reserved、512 MiB 实际 Trace、1 GiB 逻辑数组、
   90 s 写盘；将每批上限保持为 5 GPU 小时和 5 GiB 实际 Trace。

任一阈值、OOM、输入/生成 ID 不一致、Forward/Step 契约失败、RULER 答案不匹配或工件 Hash
失败，均使该批次停止在当前已完成样本之后并进入审查；不得通过截断、删除失败证据或修改
采集配置来继续同一 `run_id`。

## 5. 运行集合接口

单个 `runs/<run_id>/` 仍遵循 B1 单运行 Schema。全量结果不合并目录，而由一个只追加、不可
覆盖的 `runs/<run_set_id>.manifest.json` 引用各批次。建议 Schema 如下：

```json
{
  "schema_version": "headkv-v1-b1-run-set-v1",
  "run_set_id": "b1_full_100_<date>_v1",
  "status": "planned | collecting | completed | partial | rejected",
  "fixed_dataset": {
    "prompts_path": "data/fixed_v2/prompts.jsonl",
    "manifest_path": "data/fixed_v2/manifest.json",
    "data_sha256": "..."
  },
  "execution_contract": {
    "model_snapshot": "...",
    "git_commit": "...",
    "dtype": "bfloat16",
    "attention_backend": "sdpa",
    "generation_config_sha256": "...",
    "collection_config_sha256": "..."
  },
  "batches": [
    {
      "batch_id": "govreport_16k_b01",
      "sample_ids": ["..."],
      "plan_path": "...",
      "baseline_path": "...",
      "run_dir": "...",
      "status": "planned | completed | failed | superseded",
      "run_metadata_sha256": "...",
      "file_manifest_sha256": "...",
      "validation_reports": ["..."]
    }
  ]
}
```

一个 `completed` 运行集合必须同时满足：所有 10 批状态为 `completed`；100 个 `sample_id`
恰好出现一次；每条输入 Hash 与冻结集一致；所有批次的模型、提交、dtype、backend、生成和
采集配置 Hash 一致；每批均有完整的 Baseline/B1、文件清单和核验报告。否则只能标记为
`partial` 或 `rejected`，不得作为全量统计输入。

该接口是本次计划需实现的最小新增工具边界：现有 `prepare_b1_pilot` 固定了 10 条试运行
选择，尚不能生成上述全量批次计划或运行集合 Manifest。审核通过后，应先实现并用构造输入
测试该规划/核验工具，再进行任何 GPU 启动。

## 6. 失败恢复与保留策略

若某批失败或触发预算，保留原计划、日志、已完成样本、`run_metadata.json` 与
`file_manifest.json`。审查原因后，新建带 `r1` 后缀的后继批次，只包含原批次中未完成的
样本；后继批次必须重新执行自己的 Baseline/B1，不得向旧目录追加文件。运行集合 Manifest
显式记录原批次为 `failed` 或 `superseded` 及其后继关系。

运行完成后的最小保留集为：每批原始 B1 Trace、计划、Baseline、配置快照、运行元数据、
文件清单，以及 RULER/完整性/RCO 核验报告。先前已被 7.5 覆盖的临时 Trace 可以清理；全量
原始 Trace 仅在离线分析、统计图表和结论审核冻结后，才可在明确批准下删除。NPZ 已压缩，
不将二次压缩视为容量方案。

## 7. 执行门

本文通过审核后，下一步是实现全量计划与运行集合的生成/核验工具，并在 CPU 构造输入上验证。
只有工具、预检和首批计划均经审阅确认后，才可启动 `ruler_8k_b01`；不得因本文存在而自动
开始全量 B1。
