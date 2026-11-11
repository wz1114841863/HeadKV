# Head Analysis 实验代码

本目录承载 HeadKV 的专用观测、采集和离线分析。B1/B2 正式 100 条采集及离线分析均已
完成，当前推进 V2 小模型验证。目标和进展见[研究总览](../../docs/research/研究总览.md)；
方法与结果分别见 [B1](../../docs/research/v1_b1_attention_analysis.md)、
[B2](../../docs/research/v1_b2_prefill_consistency.md) 和 [V2](../../docs/research/v2_kv_representation.md)。本文件只维护代码组织和使用方式。

## 代码组织

| 目录 | 职责与主要入口 |
|---|---|
| `common/` | `io.py`：共用读写；`statistics.py`：均值、排名、相关性及 Bootstrap。直接提取已有实现，保持 NumPy-only |
| `runtime/` | `attention_probe.py`：Llama/SDPA 的只读 Q/K/V 及可选 Pre-RoPE K；`model.py`：共用模型加载、dtype、维度和 EOS 辅助函数 |
| `collectors/` | `attention_stats.py`：B1 基础统计；`prefill_consistency.py`、`b2_attention.py`：B2 参考及在线统计；`kv_representation.py`：V2 分块统计与位置抽样 |
| `storage/` | `b1_reader.py`：B1 来源读取；`trace_writer.py`、`b2_trace_writer.py`：Attention 数据写入；`kv_representation.py`：V2 逐层读写及 BF16 位保存 |
| `runners/` | `input_selection.py`：共用固定输入选择；`baseline_decode.py`、`b1_runner.py`、`b2_runner.py`、`kv_representation_runner.py`：模型调用与采集编排；`resource_metrics.py`：资源计量 |
| `analysis/` | `b1_offline.py`、`page_rco.py`、`token_stability.py`、`metric_relationships.py`：B1 分析；`b2_consistency.py`、`b2_coverage.py`、`b2_relationships.py`、`b2_joint.py`：B2 来源、覆盖、关联与联合分析 |
| `scripts/` | 数据准备、模型下载、烟测、离线分析和结果整理的命令入口 |
| `tests/` | 对应模块的 CPU 回归测试 |

Probe 不计算指标或写文件，Collector 不安装模型 Hook，Writer 不持有跨 Step GPU
张量，离线分析不重新运行模型。专用研究逻辑放在上述模块，脚本负责调用和编排。
当前支持已验证的标准 Llama/SDPA 路径，不自动承诺其他 Backend。

`AttentionProbe` 支持两个同步回调，按 `sample_id/layer_id/forward_id` 和位置关联：

- `on_key_projection(context)`：`key_pre` 形状 `[1,Hkv,q_len,D]`，`token_positions` 是此次实际投影行的绝对位置。仅显式传入该回调时安装 `k_proj` 输出 Hook。
- `on_attention(context)`：`key/value` 是原模型 Cache 更新后实际传给 Attention 的 K_post/V，形状 `[1,Hkv,T_cache,D]`。Decode 中用 `query_positions` 定位新增 Token。

这两个 Context 都只提供临时 Tensor 引用。消费方在回调内抽样并复制到 CPU，不能修改或持有完整原始 Tensor。默认调用与 B1/B2 原有用法兼容，Probe 本身不保存 K_pre 或计算表示统计。

V2 采集使用 `KVRepresentationCollector(prompt_length, on_record, config)`：把它的
`on_key_projection/on_attention` 接到 Probe，并用 `KVRepresentationWriter.write_record` 接收
CPU 记录。`KVRepresentationConfig` 提供统计 `block_size` 和 Decode 检查点，默认 `(31,255)`，
RULER 使用 `(5,)`。样本完成后调用 Writer 的 `finalize()`，再清除 Collector 的 CPU Prefix 状态。

每个样本目录保存 `metadata.json`、`prefill/layer_NNN.npz` 和
`decode/step_SSS/layer_NNN.npz`。元数据记录范围、dtype、检查及文件字段；`read_record(directory,relative)`
恢复抽样原始 Tensor（包括 BF16），统计字段返回 NumPy 数组。逐层写入复用 `common.io`；不新增 Hash 门禁。
`runners/kv_representation_runner.py` 复用 B2 的来源读取、执行契约和固定轨迹回放，
将这些模块组装为 V2 采集入口。样本目录另存 `sample_metadata.json`（来源、状态与资源）和
`replay.json`（实际回放 ID 及预测）；运行根目录的 `run_metadata.json` 汇总所选及已完成样本。
只有回放 Top-1 全部匹配、全部 Layer/检查点齐全且历史 Prefix 不变，才标记完成。

## 数据准备与采集

`scripts/prepare_dataset.py` 生成固定输入，
`scripts/prompt_decode_smoke.py` 执行普通推理实验 A/B。参数说明使用各入口的
`--help`；固定输入来源与结果见[数据报告](../../docs/research/data_preparation.md)。

以下是新运行的示例。模型需使用同一快照，GPU 显式指定；已有正式数据无需重跑。

```bash
.venv/bin/python -m experiments.head_analysis.runners.b1_runner \
  --input data/fixed_v2/prompts.jsonl --manifest data/fixed_v2/manifest.json \
  --model <snapshot> --tokenizer <snapshot> --device cuda:N --dtype bfloat16 \
  --sample-id <sample-id> --run-dir data/runs/<new-b1-run>
```

B2 从完成的 B1 Run Set 加载 Prompt 和生成 ID，执行 teacher-forced replay；
Pilot 可用 `--verify-no-probe` 额外运行无 Probe 对照。

```bash
.venv/bin/python -m experiments.head_analysis.runners.b2_runner \
  --run-set data/runs/b1_full_100_20261029_v1.manifest.json --source-root . \
  --model <snapshot> --tokenizer <snapshot> --device cuda:N --dtype bfloat16 \
  --sample-id <sample-id> --verify-no-probe --run-dir data/runs/<new-b2-run>
```

V2 同样读取 B1 的冻结 `input_ids`，不重新分词或渲染 Prompt。按已确认方案，RULER 消费
前 6 个生成 Token、在 `s=5` 保存；GovReport/PG19 消费前 256 个、在 `s=31/255` 保存。
由于还需核对每次 Forward 的下一 Token 预测，来源分别至少要有 7/257 个生成 ID；不足则报错。

```bash
.venv/bin/python -m experiments.head_analysis.runners.kv_representation_runner \
  --run-set data/runs/b1_full_100_20261029_v1.manifest.json --source-root . \
  --model <snapshot> --tokenizer <snapshot> --device cuda:N --dtype bfloat16 \
  --sample-id govreport_8k_001 --run-dir data/runs/<new-v2-run>
```

多条输入重复传 `--sample-id`；`--block-size` 只控制 CPU 统计分块，默认 1024。
来源模型、dtype、SDPA 和 Tokenizer 必须与 B1 契约一致；沿用现有来源读取检查，不新增
Git 状态检查或 Hash 清单。V2 的 Forward/回放计时**包含**同步统计和 NPZ 写入；另存 CUDA
峰值显存（CPU 测试没有该项）。发生错误时停止后续样本，保留已完成及部分输出；已有目录
拒绝覆盖，重新尝试使用新目录并显式选择待完成样本。以上命令为用法示例，尚未运行真实模型。

新运行使用新目录。B1 工件由离线来源预检核验，B2 可调用 `validate_b2_run`；
不以进程退出码代替完整性验收。RULER 可用 `scripts/validate_b1_ruler.py` 检查已保存答案。

## 离线分析

离线入口只需 CPU，读取 Run Set 显式登记的正式批次。来源校验会读取 Trace，但不重新
推理；无效指标保留缺失值和有效数。各阶段中间包为 `partial`，最终整理后才发布整体
`completed`。输入参数与输出字段以代码和冻结结果包为准。

| 入口 | 主要参数与输出 |
|---|---|
| `scripts/analyze_page_rco.py` | 从完成的 B1 Run 生成双 Scope、多 k/Δ Page RCO 报告 |
| `scripts/analyze_b1_run_set.py` | `--run-stage3` 汇总 Query/Page；`--run-stage4` 计算 Token 稳定性；`--run-stage5 --stage3-dir ... --stage4-dir ...` 分析关联 |
| `scripts/analyze_b2_run_set.py` | 默认只做来源预检；`--run-stage2` 汇总覆盖；`--run-stage3 --stage2-dir ...` 分析 CV/覆盖；`--run-stage4 --stage2-dir ... --stage3-dir ...` 联合 B1 |
| `scripts/analyze_prefill_stability.py` | 以已有 B2 数值包探索 CV 对留出样本 Decode 稳定性的额外信息 |
| `scripts/finalize_b1_analysis.py`、`finalize_b2_analysis.py` | 整理已验收数值和报告，创建新的统一完成包 |

两个 Run Set 分析入口均需 `--run-set`、`--analysis-id` 和 `--output-dir`，可用
`--source-root` 指定来源目录；B1 还需为每批传入 `--page-rco-report`。
`--analysis-git-commit` 仅作可选版本记录，不检查 HEAD 或工作区，也不再需要 `--allow-dirty`。
B2 阶段二用 `--scratch-dir` 指定数据盘上的临时缓冲父目录。
整理 B2 分析时，指定已有三个阶段输出，可选 `--report` 加入研究报告：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.finalize_b2_analysis \
  --stage2-dir <stage2目录> --stage3-dir <stage3目录> --stage4-dir <stage4目录> \
  --analysis-id <结果名称> --output-dir <新目录> --report docs/research/v1_b2_prefill_consistency.md
```

整理过程检查阶段来源和样本坐标，将已有数值、参数和摘要复制到新目录，不重新计算指标。
它不依赖历史审计文件，也不预设研究结论；模型下载脚本尊重现有 Hugging Face 环境变量。
新结果使用 `headkv-v1-b2-numeric-bundle-v2` 标识，其文件布局和索引不同于历史冻结包。

在已整理的 B2 原数值包上运行探索性逐样本留出分析（无需 GPU，不覆盖原包）：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_prefill_stability \
  --bundle data/analysis/b2_full_100_completed_20261110_v1 \
  --output-dir data/analysis/<新分析目录> --bootstrap-replicates 2000
```

输出包括 `summary.json`、逐样本 `sample_metrics.jsonl` 和 `bootstrap.npz`。同一条件内
逐样本留出，PG19 按 Pair 留出；Bootstrap 按同一单位重采样并重新拟合，不将 Head 或 Step
视为独立样本。该分析使用已研究过的输入，不能代替独立新输入验证。

日常读取检查必需文件、字段、数组形状和样本/Step 对齐，不默认逐文件核对 Hash；B2 覆盖
分析也不依赖历史中断目录、核验报告或 B1 分析包，联合分析仍需 B1 数值包。旧格式的
来源标识仍保留，例如 B1 Page RCO 报告通过元数据 Hash 匹配批次；显式核验入口仍可检查
文件 Hash。已有冻结结果包不修改。
来源读取从 `analysis/b1_offline.py` 移至 `storage/b1_reader.py`；原有导入仍兼容，统计口径和
Prefill/Decode 采集行为不变。Page RCO 与 Token 稳定性的样本均值 Bootstrap 共用计算实现，
保留原有抽样与数值运算顺序。
统计提取保留原有 FP32/FP64、缺失值处理和随机抽样顺序。

方法定义和主结果直接阅读阶段报告；完整参数及数组轴在完成包的指标和数值索引中。
已冻结包中的旧文档和源码保持原样，重现旧版本行为使用包内源码快照。

## 测试

在项目根目录运行现有 CPU 回归测试：

```bash
.venv/bin/python -m pytest experiments/head_analysis/tests -q
```

测试以采集链路、指标公式和代表性的读写流程为主；改变数据或参数无需为每次实验新建测试。
共享的小型输入和轨迹构造放在 `tests/fixtures.py`、`tests/b2_fixtures.py`。

需要真实 GPU 时，可使用不下载模型权重的 Tiny Llama 检查：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.smoke_b2_tiny_model \
  --device cuda:N --dtype bfloat16
```

该烟测检查无 Probe/B2 回放的 Logits、Top-1 和阶段对齐，不能代替真实长输入验证。
