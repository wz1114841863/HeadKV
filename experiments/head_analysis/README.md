# Head Analysis 实验代码

本目录承载 HeadKV 的观测、采集、受控回放和离线分析。B1/B2 与 V2 的 100 条采集及分析已完成；V3 已完成表示重建、最小 Dense 存储模拟和量化/同字节组合对照。最新组合 Calib 的 K/V 均未通过第二关，本轮停止，不运行组合 Validation。

目标和状态见[研究总览](../../docs/research/研究总览.md)，方法与结果见 [B1](../../docs/research/v1_b1_attention_analysis.md)、[B2](../../docs/research/v1_b2_prefill_consistency.md)、[V2](../../docs/research/v2_kv_representation.md) 和 [V3-B 阶段小结](../../docs/research/v3_head_aware_kv_optimization.md#v3-b-阶段小结)。本文件维护代码组织、历史命令和结果入口，列出的已完成运行不代表待执行任务。

## 代码组织

| 目录 | 职责与主要入口 |
|---|---|
| `common/` | `io.py`：共用读写；`statistics.py`：均值、排名、相关性及 Bootstrap。直接提取已有实现，保持 NumPy-only |
| `runtime/` | `attention_probe.py`：Llama/SDPA 的只读 Q/K/V 及可选 Pre-RoPE K；`model.py`：共用模型加载、dtype、维度和 EOS 辅助函数 |
| `collectors/` | `attention_stats.py`：B1 基础统计；`prefill_consistency.py`、`b2_attention.py`：B2 参考及在线统计；`kv_representation.py`：V2 分块统计与位置抽样 |
| `storage/` | `b1_reader.py`：B1 来源读取；`trace_writer.py`、`b2_trace_writer.py`：Attention 数据写入；`kv_representation.py`：V2 逐层读写及 BF16 位保存 |
| `runners/` | `input_selection.py`：共用固定输入选择；`baseline_decode.py`、`b1_runner.py`、`b2_runner.py`、`kv_representation_runner.py`：模型调用与采集编排；`resource_metrics.py`：资源计量 |
| `analysis/` | B1 分布与稳定性、B2 覆盖与关联；`kv_representation_metrics.py`：谱/CKA；`kv_representation.py`：V2 范围比较与 B1 对齐；`head_identity.py`：固定 Head 与跨输入变化 |
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

多条输入重复传 `--sample-id`，或改用 `--sample-ids-file <json>` 读取按执行顺序排列的
样本 ID 字符串列表（两种选样方式互斥）；`--block-size` 只控制 CPU 统计分块，默认 1024。
来源模型、dtype、SDPA 和 Tokenizer 必须与 B1 契约一致；沿用现有来源读取检查，不新增
Git 状态检查或 Hash 清单。V2 的 Forward/回放计时**包含**同步统计和 NPZ 写入；另存 CUDA
峰值显存（CPU 测试没有该项）。发生错误时停止后续样本，保留已完成及部分输出；已有目录
拒绝覆盖，重新尝试使用新目录并显式选择待完成样本。进度输出包含样本序号、完成数/总数和耗时。
目前六条 Pilot 均已完成，剩余 94 条沿用同一采集协议，按六个条件串行执行；
结果位置、核验及资源开销见 [V2 报告](../../docs/research/v2_kv_representation.md)。

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
| `scripts/analyze_kv_representation.py` | 组合已完成 V2 Run，计算全量/抽样谱、448 位置 CKA 及置换参照，与已有 B1 RCO 对齐 |
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

V2 离线分析读取逐层 CPU 数值和 B1 完成包，无需重新推理或 GPU。重复传入 `--run-dir`
合并不同采集批次；默认全层、每层所有 KV Head 对、50 次单侧置换和 seed=20261111。

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_kv_representation \
  --run-dir data/runs/v2_kv_govreport_8k_001_20261111_v1 \
  --run-dir data/runs/v2_kv_pilot_remaining5_20261111_v1 \
  --b1-bundle data/analysis/b1_full_100_completed_20261103_v2 \
  --output-dir data/analysis/<new-v2-analysis>
```

输出 `summary.json`、`sample_summary.jsonl`、`permutations.npz` 和逐样本 NPZ；保留范围/N、
有效标记、完整谱、原始 RCO/有效数及全部置换值。计算使用 FP64，大数组发布为 FP32，
相关性输入矩阵保留 FP64 以精确重算；零方差保留无效，明显负谱报错。RULER 排除长期关系，
只另存短轨迹描述。现有完成数值包位于 `data/analysis/v2_kv_pilot_6_20261111_v1/`，无需重跑。

在该包上比较 Head 间差异与同一 Head 的输入变化，只读已有数值，不重算谱或 CKA：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_head_identity \
  --bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --output-dir data/analysis/<new-head-identity-analysis>
```

输出 `summary.json`、`profiles.npz` 和四个子集的 NPZ，保留 FP64 平方和分解、逐层份额、
排序及来源组留出结果。PG19 两长度成组留出，并分别保留短/长代表作敏感性检查；
原包和旧指标不变。当前结果为 `data/analysis/v2_head_identity_6_20261111_v1/`，
方法和“固定统计画像不等于固定 PCA 基”的边界见 [V2 报告](../../docs/research/v2_kv_representation.md)。

扩展验证先用上述谱/CKA 入口，仅传六个新增 Run（不含 Pilot），输出新的 94 条数值包。
然后固定六条 Pilot，执行问题 A；长度条件和 Pair 直接读取冻结输入元数据，不按实际长度猜测：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_head_identity \
  --bundle data/analysis/v2_kv_remaining94_20261112_v1 \
  --reference-bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --fixed-inputs data/fixed_v2/prompts.jsonl \
  --bootstrap-replicates 2000 --seed 20261112 \
  --output-dir data/analysis/v2_head_stability_a_94_20261112_v1
```

该模式只分析 K_post/V 的秩、E16、平均 Norm（完整中段主指标及同 448 位置敏感性检查），
输出固定参考评估、输入对排序、来源 Bootstrap 及分解数值，不更新参考、不覆盖原包。
PG19 两长度共用 Pair 抽样，参考不重采样；条件等权 CI 用于固定参考的逐输入评估，
输入对仅报告描述性分布。完整公式、统计范围和输出路径见 V2 报告第 11.5 节。

问题 B 直接使用这两个谱包，比较相同 448 位置上的 RoPE 秩变化，无需原始 K/V 或 GPU：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_rope_stability \
  --bundle data/analysis/v2_kv_remaining94_20261112_v1 \
  --reference-bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --sample-index data/analysis/b2_full_100_completed_20261110_v1/sample_index.json \
  --output-dir data/analysis/v2_rope_stability_b_100_20261113_v1
```

长度和 Pair 读取已有样本索引；输出逐输入秩变化、逐 Layer/Head 的跨输入统计、固定参考排序、
输入对排序、配对长度差和来源 Bootstrap。默认 2,000 次、seed=20261112，保留原包；
主要报告新增 94 条，Pilot 与合并 100 条分开保留。方法及结果见 V2 报告第 11.7 节。

问题 C 比较每层 28 个 Head 对的 CKA 排序与 Pilot 高相似配对的保持情况，仅读取已发布值：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_cka_stability \
  --bundle data/analysis/v2_kv_remaining94_20261112_v1 \
  --reference-bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --sample-index data/analysis/b2_full_100_completed_20261110_v1/sample_index.json \
  --output-dir data/analysis/v2_cka_stability_c_100_20261113_v1
```

默认固定 Pilot 均值前 7 对，Observed/Excess 分开；保留逐层排序、Top-7 保留率、输入对及三种表示比较。
三份 `matrices_k_pre/k_post/v.npz` 分别保存每层 8×8 对称关系矩阵，`mean/std/top_k_frequency`
轴为 `[cohort,kind,layer,head,head]`，`kind` 为 Observed/Excess，名称和坐标均在同文件；
对角线未测量，保持 NaN。Bootstrap 沿用按来源抽样、PG19 Pair 成组及六条件等权，不重算 SVD/CKA。
结果与限制见 V2 报告第 11.8 节。

问题 D 联合 V2 表示和 B1 Page RCO，使用已核验的 B2 数值包内的 B1 副本交叉核对：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_attention_kv_joint \
  --bundle data/analysis/v2_kv_remaining94_20261112_v1 \
  --reference-bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --b2-bundle data/analysis/b2_full_100_completed_20261110_v1 \
  --output-dir data/analysis/v2_attention_kv_joint_d_100_20261113_v1
```

全部 100 条对齐；长期关联适用 60 条，新增 56 条为主要报告对象，RULER 保留坐标及不适用状态。
输出 D1 的逐输入/逐层相关与方向、D2 各条件的固定画像/输入残差及分解数值。
默认 2,000 次来源 Bootstrap；D2 每次重新估计 Head 均值，PG19 两长度共用 Pair 抽样。
仅用 CPU 和已有数值，不重算模型、谱、CKA 或 RCO；方法及结果见 V2 报告第 11.9 节。

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

## V3-A 离线重建

读取远端既有 V2 原始工件，不运行模型。三个阶段依次执行：

```bash
OPENBLAS_NUM_THREADS=8 .venv/bin/python -m experiments.head_analysis.scripts.analyze_kv_reconstruction \
  prepare --run-dir data/runs/<V2批次1> --run-dir data/runs/<V2批次2> \
  --fixed-inputs data/fixed_v2/prompts.jsonl --output-dir data/analysis/<新目录>

OPENBLAS_NUM_THREADS=8 .venv/bin/python -m experiments.head_analysis.scripts.analyze_kv_reconstruction \
  validation --output-dir data/analysis/<同一新目录> --threads 8

OPENBLAS_NUM_THREADS=8 .venv/bin/python -m experiments.head_analysis.scripts.analyze_kv_reconstruction \
  test --output-dir data/analysis/<同一新目录> --threads 8
```

`prepare` 的 `--run-dir` 可重复，须合计覆盖固定 100 条；该阶段先冻结 `split.json`，
再实际读取所有原始层工件，检查两范围 moments 和 512 位置向量，不计算重建误差。
`validation` 只用 Calib 拟合，完成 A1～A3、同位置 BF16 及 RoPE 补充后写入
`validation_freeze.json`；`test` 必须在冻结后单独启动，不重新拟合或选择方法。
已有拟合、评估不能原地重跑，独立尝试使用新目录，避免覆盖已冻结结果。

`analysis/pca_reconstruction.py` 负责均值、基及误差公式，
`analysis/kv_representation_budget.py` 负责维度分配和完整字节账本，
`storage/representation_inputs.py` 负责已有输入的读取和划分，
`analysis/kv_reconstruction.py` 组装阶段流程。输出保留逐输入误差、来源 Bootstrap、
字节账本、实际 FP32 参数载荷及阶段报告；全量理想误差和 448 位置 BF16 实测分开。
方法、门槛和结果见 [V3 报告](../../docs/research/v3_head_aware_kv_optimization.md)。

方向一的子空间诊断读取原 V2 moments，复用冻结的 V3 均值、尺度和基，仅使用 CPU：

```bash
OPENBLAS_NUM_THREADS=8 .venv/bin/python -m experiments.head_analysis.scripts.analyze_kv_subspace_diagnostic \
  --bundle data/analysis/v3_a_20261113_v1 \
  --output-dir data/analysis/v3_a_subspace_diagnostic_<新版本> --threads 8
```

`analysis/pca_subspace.py` 负责几何量与有符号误差分解，
`analysis/kv_subspace_diagnostic.py` 负责逐层读取、A1 复现、输入对及来源 Bootstrap。
`layers/` 保留逐坐标数值与 G/M、G_C/C 的双端谱边界，`head_summary.npz` 保存各组
Head 汇总，`bootstrap.npz` 保存共用来源抽样和总体区间；理论非负损失的小负值原样保存并标记。
固定基相对逐输入 Oracle 的差距包含 Calib 内折中，不能全部称为泛化失败。说明见 V3 报告第 15 节。

## V3 Pre-RoPE Key 旁路编码

入口复用原 B1 固定输入/轨迹和 V3 冻结参数,模型继续使用 Dense Cache:

```bash
.venv/bin/python -m experiments.head_analysis.scripts.run_key_encoding \
  --bundle data/analysis/v3_a_20261113_v1 \
  --run-set data/runs/b1_full_100_20261029_v1.manifest.json --source-root . \
  --model <snapshot> --tokenizer <snapshot> --device cuda:4 \
  --cohort pilot --run-dir data/analysis/v3_pre_rope_pilot_<版本>/pilot
```

`pilot` 固定为原六条 Calib Pilot;核验其完成后,才使用 `--cohort validation`
和 `--pilot-dir <已完成的pilot目录>` 执行计划中的六条 Validation,结果写入另一目录.
已有目录不会覆盖,中断后保留失败现场;不自动转入 Validation 或更换输入.

`runtime/key_codec.py` 负责编码/恢复/字节账本,
`collectors/key_encoding.py` 负责完整中段/连续 Decode 的只读核验,
`runners/key_encoding_runner.py` 组装现有 Probe/回放/统计和通用落盘函数.
Probe 只补充实际 position_embeddings;没有改模型前向或 KV Cache 实现.
结果包括逐层 K_pre moments、两路径理想/BF16 误差、Decode BF16 系数位、检查点恢复成本及摘要.
恢复成本使用真实形状和重复的抽样系数做独立算子测量,不是压缩模型 Decode 延迟.

本地相关检查见 `tests/test_key_encoding.py`;远端 tmux 中可设置
`HEADKV_TEST_DEVICE=cuda:4` 运行同一测试,无需下载模型.
研究范围和精度区别见 [V3 报告第 16 节](../../docs/research/v3_head_aware_kv_optimization.md#16-方向二短期方案pre-rope-key-的完整编码与恢复已批准).

## Pre-RoPE Head 差异化预算离线探索

复用上述已完成的六条 Pilot/六条 Validation,不加载模型、不新增采集:

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_pre_rope_head_budget \
  --bundle data/analysis/v3_a_20261113_v1 \
  --codec-root data/analysis/v3_pre_rope_pilot_20261116_v1 \
  --output-dir data/analysis/v3_pre_rope_head_budget_<新版本> --threads 4
```

`analysis/pre_rope_head_budget.py` 组合已有投影误差、Head 画像、离散分配和编码函数。
先用 Calib 完整中段曲线冻结 Pre/Post 分配,再评估 Validation;均值/基保持不变。
输出完整中段理想误差、同 448 位置 BF16 误差、稳定性、实际参数载荷和字节账本。
两种范围分别报告,不作为实际 Cache 干预或推理性能结果;具体口径见 V3 报告第 17 节。

## Pre-RoPE 跨 Head 共享基对照

入口复用既有六条 Calib/六条 Validation 工件,输出独立的共享基分析数值包。
应在能读取原 `split.json` 所引用 V2 原始向量的主机上运行;目前这些路径位于远端。
仅需 CPU,无需加载模型:

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_pre_rope_shared_basis \
  --bundle data/analysis/v3_a_20261113_v1 \
  --codec-root data/analysis/v3_pre_rope_pilot_20261116_v1 \
  --cka-bundle data/analysis/v2_kv_pilot_6_20261111_v1 \
  --output-dir data/analysis/v3_pre_rope_shared_basis_20261116_v1 --threads 4
```

输出目录须不存在,重复运行请更换版本。
`analysis/pre_rope_shared_basis.py` 复用 PCA/编码/读取函数,仅用六条 Calib 拟合新逐 Head 基、
两 Head 共享基及索引/CKA/重建代价三种配对,冻结参数和分组后再读取 Validation 数值。
均值/归一化尺度沿用旧参数;旧逐 Head 基单列参照,不将重新拟合的变化归为共享收益。
共享基按组导出一次,实际 BF16 参考编码使用回读的参数和映射。
结果包含完整中段理想误差、同 448 位置最终 K_post 误差、配对诊断及模型/请求字节账本。
r128 只作数值检查,不参与候选选择;具体范围见 V3 报告第 18 节。

## Dense KV 存储模拟

按 [V3 第 20 节](../../docs/research/v3_head_aware_kv_optimization.md#20-最小-dense-存储模拟计划2026-11-17)
执行固定六条 Calib/六条 Validation,仅使用 CPU,不加载模型或改变 Attention:

```bash
.venv/bin/python -m experiments.head_analysis.scripts.simulate_dense_kv_memory \
  --run-set data/runs/b1_full_100_20261029_v1.manifest.json --source-root . \
  --bundle data/analysis/v3_a_20261113_v1 \
  --b1-bundle data/analysis/b1_full_100_completed_20261103_v2 \
  --codec-root data/analysis/v3_pre_rope_pilot_20261116_v1 \
  --output-dir data/analysis/v3_dense_memory_20261117_v1
```

来源路径须可读,当前完整 B1 数据在远端.输出目录必须为新目录.
`analysis/dense_kv_memory.py` 包含小序列逐请求参考和满足条件的精确扫描计数,
复用 B1 Reader、固定输入分组、条件权重和公共 I/O.默认比较六策略/三容量,
画像只来自原 36 条长文本 Calib,先核对 Calib 再保存冻结配置/配额,之后评估 Validation.
结果为配置、画像/uint32 配额、两阶段 CSV、紧凑逐 Head 整数计数及摘要,不展开巨量请求事件.
计数反映明确假设下的存储模型;参数化搬运时间不是真实模型延迟或硬件实测.
上述 v1 数值包已完成并核验,再次运行须使用新版本目录;首轮结果见报告第 20.10 节.

## Head 级量化敏感性

[V3 第 21 节](../../docs/research/v3_head_aware_kv_optimization.md#v3-b-阶段小结)
的量化、单 Head 干预、全 Head 局部统计及落盘链路已完成小模型验证；六条正式 Calib 已完成工件和数值一致性核验，
结果见 V3 报告第 21.11 节：局部 Head 差异明显，最终预测排序尚不稳定。
六条冻结 Validation 已在 GPU4 完成并核验，输出 `data/analysis/v3b_quant_validation_20261118_v1/`；局部画像保持，最终预测排序未保持，8 bit 非单调仍存在，结果见第 21.13 节。
`runtime/kv_quantization.py` 提供 K/V 分开的 4/8 bit 编码、实际打包/恢复与布局字节账本；
`runtime/head_intervention.py` 仅替换目标 Attention 输入副本，不改原 Cache 或只读 Probe。
`runners/quantization_runner.py` 的 `replay_quantization_checkpoints` 接受原 Prompt/生成 ID，
以及 `{decode_step: [HeadIntervention(...), ...]}`；复用原固定回放，只 Prefill 一次，
逐个执行并恢复检查点分支。`bits=None` 为旁路对照，`side` 为 `key` 或 `value`。
分支预测允许变化，原主回放继续核对 B1 预测。

`collectors/quantization_sensitivity.py` 逐 Head 计算全域 Attention 误差；
`storage/quantization_results.py` 保存每检查点的紧凑 NPZ、模型分支结果、字节及资源汇总，
不保存完整 Attention、Q/K/V、词表 Logits 或分支 Cache。
目前仅支持标准非 offload DynamicCache。
量化布局预算与旁路研究进程的实际显存分开；恢复引用不代表 Cache 追加没有分配开销。

正式运行入口如下，模型权重须已在本机；重复运行请换新输出目录：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.run_quantization_sensitivity \
  --bundle data/analysis/v3_a_20261113_v1 \
  --run-set data/runs/b1_full_100_20261029_v1.manifest.json --source-root . \
  --model /模型快照路径 --tokenizer /模型快照路径 \
  --device cuda:4 --cohort calib \
  --output-dir data/analysis/v3b_quant_calib_v1
```

六条 Calib 完成后保存配置和仅使用主检查点的 Head 画像。
Validation 使用相同参数，改为 `--cohort validation`，添加
`--calib-dir data/analysis/v3b_quant_calib_v1`，并换一个新的 `--output-dir`。
两阶段分开运行；Validation 在 Calib 不完整或冻结配置不一致时拒绝启动。
固定样本/层/位宽不可从 CLI 改写，`--block-rows` 仅控制分块大小，默认 1024。

可单独运行本轮 CPU 正确性测试，无需下载模型：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  experiments/head_analysis/tests/test_quantization_replay.py -q
```

D1 控制核心见 `runners/quantization_diagnostics.py`。`run_d1_checkpoint` 接受现成的原始
Cache、当前/下一 Token 与 `HeadIntervention(layer, head, side, None)`，按固定十二次序列
比较基线、匹配旁路、等值克隆、4/8 bit 换序重复与首次重建值直接注入。
`replay_d1_checkpoints` 接受原 Prompt/生成 ID 和 `{step: [上述坐标, ...]}`，复用一次
Prefill 与固定前缀回放；返回 `.baseline` 与 `.checkpoints`，每坐标包含 `.records`、
`.comparisons`，`capture_logits=True` 可保留 CPU 原生精度 Logits。
`check_inputs=False` 关闭新增诊断检查/快照，传入的原 `on_attention` Observer 仍在基线生效。
固定重建的测量复用首次编码信息，`encoding_performed=False` 表示本次未重新编码。
D1 控制核心已通过完整回归；落盘和 CLI 补充后，本地/远端各 40 项相关测试及远端
BF16、D=128 GPU 小模型端到端检查通过。真实模型运行状态见 V3 报告第 21.17 节。

固定两条输入的真实模型诊断入口（只运行 D1，输出目录必须不存在）：

```bash
.venv/bin/python -u -m experiments.head_analysis.scripts.run_quantization_diagnostics \
  --calib-dir data/analysis/v3b_quant_calib_20261117_v1 \
  --validation-dir data/analysis/v3b_quant_validation_20261118_v1 \
  --device cuda:0 --output-dir data/analysis/v3b_quant_diagnostic_20261118_v1/d1
```

`cuda:0` 对应启动时通过 `CUDA_VISIBLE_DEVICES` 指定的物理 GPU。
入口复用原配置、Prompt 和 B1 前缀，保存逐分支记录、原生 Logits 的精确 FP32 转存、
输入检查开/关比较及资源统计。GovReport 主坐标的原 KL/Top-1 未精确复现时，
仅触发一次预定的原顺序回放；这只是保守触发规则，不是质量阈值。

D2 的 `collectors/quantization_diagnostics.py` 在同一检查点分支上捕获真实 `o_proj`
输入/输出、目标及后续层输出、最终 Norm、`lm_head` 输入/输出，记录逐阶段 SSE、
参考能量、NMSE 和最大差，并区分目标 Query Head 组与其余组。
API 增加 `capture_propagation=True`，默认关闭；须保留 `check_inputs=True`。
逐分支记录位于 `.records[i]["propagation"]`，`.propagation.projection` 是 FP32
投影对照，`.propagation.arrays` 是紧凑数组。

CLI 增加 `--capture-propagation` 即启用 D2，须另选尚不存在的输出目录，不能使用已完成的
`d1/`。十二分支序列和 Forward 数不变；关闭检查的匹配序列也关闭 D2 hooks，保留原 Observer。
每个带检查坐标额外写 `*_checked_propagation.npz`：保存 B0/Q4/Q8 的最终隐藏向量、
真实 Attention 输出、原设备 FP32/CPU FP64 公式输出及 FP32 重投影 Logits；逐层追踪和
原生/FP32 的并列数、Top-1 间隔、KL64 位于同名 JSON。Q/K/V 只暂存选中 Head，计算后释放。
FP32 投影用原隐藏值和原权重分块重算，单独禁用 autocast/TF32 并恢复设置，
各分支相对其 FP32 基线比较；原 KL32 不变。已通过本地完整回归、远端 48 项相关测试和
CUDA 小模型验证；GPU4 的两条真实 D2 已完成工件和数值核验，输出已下载至本地同路径
`data/analysis/v3b_quant_diagnostic_20261118_v1/d2/`。运行记录和诊断结果见 V3 第 21.18～21.19 节；
原 D1 结果保留，FP32 参照不替换原指标。

## 同字节 Head 位宽分配

入口 `scripts/run_head_budget.py` 复用原量化、固定回放和 FP32 最终投影。
K/V 各比较统一 6 bit、Calib 拟合的每层半数 4/8 bit，以及固定循环错位一位的身份对照。
每个方案都计入实际码流、FP32 统计、另一侧 BF16、保护/残余/Decode、模型位图和头部对齐。
原 4/8 bit 行为及旧指标保持不变；新增 6 bit 是质量/存储对照，不表示已有高效压缩 Attention 内核。

本地已完成下面的拟合，复用现有目录即可；重新拟合须换新目录：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.run_head_budget fit \
  --source-calib-dir data/analysis/v3b_quant_calib_20261117_v1 \
  --output-dir data/analysis/v3b_head_budget_20261118_v1/allocation
```

完整模型 Calib 已在 GPU4、`zsy_2:head_budget_calib` 运行结束，原命令如下；该输出目录已使用，
不要重复启动覆盖，重跑须换新目录：

```bash
CUDA_VISIBLE_DEVICES=GPU-e8df4311-a158-0d38-d2de-0c096b18fd47 \
.venv/bin/python -u -m experiments.head_analysis.scripts.run_head_budget run \
  --allocation-dir data/analysis/v3b_head_budget_20261118_v1/allocation \
  --cohort calib --device cuda:0 \
  --output-dir data/analysis/v3b_head_budget_20261118_v1/calib
```

该半 4/8 组合没有稳定的同预算收益，本轮不运行其 Validation。历史 `run` 接口保留
`--cohort validation --calib-dir <已完成Calib目录>`，并要求方案/配置一致，但不作为本轮
下一步。模型路径和固定样本来自原配置，已有输出目录不覆盖。
该参数目录属于忽略的 `data/`，后续远端运行时需同步此冻结目录；只有代码同步不包含它。

每输入一次 Prefill，沿既定点位执行 Dense+六种组合；每点 JSON 保存预测、并列边界、
字节账本和投影设置，NPZ 保存七个原生/重投影词表向量、最终隐藏向量、位宽及逐 Head 重建误差。
汇总先平均输入内点位、再对条件等权，保留逐输入结果，报告 Head-aware 相对统一/错位的差值。
已通过远端 61 项相关测试及 BF16、D=128 CUDA 小模型检查。真实 Calib 已运行结束，日志/退出码
为同级 `calib.log/calib.exitcode`，已记录 `HEAD_BUDGET_CALIB_EXIT=0`；窗口退出后保留。
六条 Calib、10 个检查点已完成来源、数组、独立 Logits 指标与完整字节账本核验；原数值包
也已拉至本地同名目录。FP32 最终投影对照下，差异化 K/V 平均 KL 比统一 6 bit 高
55.14%/110.86%，当前规则未体现稳定的同预算收益。结果及数值口径见 V3 第 21.21 节。
Validation 尚未启动；若继续，须复用现有冻结方案，不能据本次结果调参后称同一验证。

第一关诊断复用旧 4/8 bit 包，仅新增六条 Calib 主点的 6 bit 局部统计及 H0 衔接锚点，
拟合每层至多一对 4/8 bit、其余 6 bit 的固定分配，并计算逐输入局部 Oracle：

```bash
CUDA_VISIBLE_DEVICES=GPU-e8df4311-a158-0d38-d2de-0c096b18fd47 \
.venv/bin/python -u -m experiments.head_analysis.scripts.run_head_budget diagnose-local \
  --source-calib-dir data/analysis/v3b_quant_calib_20261117_v1 \
  --allocation-dir data/analysis/v3b_head_budget_20261118_v1/allocation \
  --device cuda:0 \
  --output-dir data/analysis/v3b_budget_diagnostic_20261118_v1
```

新目录下 `samples/` 保存每条的 `local6.npz`、`anchor48.npz`、回放和资源信息，
`analysis.npz`/`summary.json` 保存固定/Oracle 与旧方案的比较和第一关判读，`allocation/`
保存固定表及可读取的二进制描述符。阈值沿 V3 第 21.22 节冻结，命令不自动启动第二关、
Validation 或 Test。只做第一关相关 CPU 测试可运行 `tests/test_local_budget_gate.py`。
该目录已在 GPU4、`zsy_2:budget_gate1` 完成并核验，不要重复覆盖；同级 `.log/.exitcode`
保存进度和返回码，已记录 `V3B_LOCAL_GATE_EXIT=0`，退出后窗口保留。启动前本地
356 项回归、远端 70 项相关测试及 BF16、D=128、GQA CUDA 小模型检查均通过。
六条 Calib 的 K/V 固定局部 J 分别改善 62.14%/32.09%，均通过第一关；新数值包已拉至
本地同名目录，详见 V3 第 21.24 节。第二关 Calib 的执行与结果见下，组合 Validation 和 Test 未启动。

第二关六条 Calib 的组合回放入口读取上面的冻结 F，生成固定 +1 错位，K/V 分开；
每点保留 Dense 前后对照及两种 Logits 路径，不重新拟合，不衔接 Validation：

```bash
CUDA_VISIBLE_DEVICES=GPU-e8df4311-a158-0d38-d2de-0c096b18fd47 \
.venv/bin/python -u -m experiments.head_analysis.scripts.run_head_budget replay-pairs \
  --local-gate-dir data/analysis/v3b_budget_diagnostic_20261118_v1 \
  --device cuda:0 \
  --output-dir data/analysis/v3b_budget_gate2_20261119_v1/calib
```

新目录 `allocation/` 保留实际采用的六份描述符，`samples/` 保存十个点的 JSON/NPZ、
回放及资源信息；`summary.json` 保留逐输入/总体/主点/晚期汇总、实测数值背景和逐侧
第二关判读。运行/核验记录见 V3 第 21.25 节，不能将其解释为持续压缩或部署性能实测。
该目录已在 GPU4、`zsy_2:budget_gate2_calib` 完成并核验，不要覆盖。日志/退出码位于
同级 `calib.log/calib.exitcode`，已记录 `V3B_GATE2_CALIB_EXIT=0`，窗口保留；六条 48 个
工件及日志已拉至本地同名目录。K/V 均未通过第二关，按冻结方案停止，不运行组合
Validation。详见 V3 第 21.25 节的结果段。启动前完整本地 366 项回归、远端 80 项相关
测试及 BF16、D=128、GQA CUDA 小模型检查通过。

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
