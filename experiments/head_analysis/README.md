# Head Analysis 实验代码

本目录承载 HeadKV 的研究专用代码；除非多个研究方向都需要同一抽象，否则不修改
`kvpress/`。核心库代码与命令行脚本分离，运行期观测、在线统计、落盘和离线分析也
必须保持单向依赖。

## 当前目录

```text
head_analysis/
├── runtime/                  # 模型运行期、只读的捕获与生命周期关联
│   └── attention_probe.py    # Llama/SDPA Attention function 边界 Probe
├── collectors/               # 在线基础统计与 Top-K/Page 聚合
│   └── attention_stats.py    # 分块精确 Softmax，输出紧凑 CPU 记录
├── storage/                  # NPZ/JSON 落盘与完整性校验
│   └── trace_writer.py       # 原子分片写入、manifest、跨层/Step 校验
├── runners/                  # 固定输入的 B1 推理编排
│   ├── baseline_decode.py    # 无 Probe 的 Greedy ID 基线工件
│   ├── b1_runner.py          # Manifest 校验、Greedy Decode、组件串联
│   ├── pilot_preflight.py    # 冻结 Pilot 的资源/版本预检与启动参数
│   └── resource_metrics.py   # Prefill/Decode 前向计时与资源记录契约
├── analysis/                 # 完成工件的只读离线分析
│   ├── page_rco.py           # 双 Scope、k/Δ 敏感性 Page RCO
│   ├── b1_offline.py         # 来源门、Query/Page 汇总、通用聚合与分析清单
│   └── token_stability.py    # Token Top-K 集合、Rank 与逻辑年龄稳定性
├── scripts/                  # 可直接执行的准备、下载和 smoke 脚本
├── tests/                    # 面向 runtime 与 scripts 的 CPU 回归测试
└── README.md
```

## B1 职责边界

尚未实现的目录仅在其首个实现加入时创建，避免空目录和未经实现的接口成为伪承诺。

| 目录 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `runtime/` | Q/K/Mask、阶段、位置等瞬态只读上下文 | 指标计算、持久化、策略干预 |
| `collectors/` | 在线基础统计、Token/Page Top-K、紧凑 CPU 结果 | 模型 Hook、文件格式细节 |
| `storage/` | NPZ 分片、原子写入、manifest、完整性检查 | GPU Tensor 指标计算、模型推理 |
| `runners/` | 固定 `input_ids`、模型调用、Probe/Collector/Writer 编排 | 指标公式与落盘实现 |
| `analysis/` | 读取完成分片、RCO 与汇总分析 | 再次运行模型或重新计算在线轨迹 |

`scripts/` 只作为人工或批处理入口，应导入上述可测试模块，不能承载核心研究逻辑。
每个新核心模块必须在 `tests/` 增加同名测试；运行期路径优先测试输出等价、注册表/Hook
恢复和阶段索引完整性。

## 当前使用方式

当前 Probe 的稳定导入路径为：

```python
from experiments.head_analysis.runtime.attention_probe import AttentionProbe
```

当前 B1 原型仅支持 Llama 的 `sdpa` 后端、batch size 1 和完整 Prefill 后的单 Token
Decode。长 Decode 验收先运行 `runners/baseline_decode.py`，它会用相同冻结
`input_ids` 保存无 Probe 的完整生成 ID；随后给 `b1_runner.py` 传入
`--verify-baseline <artifact.json>`。采集结果只有逐 ID 完全相等时才会以 completed
状态落盘，因而不将“能运行”误作观测等价性。Probe、Collector、Writer 与 Runner 已完成
CPU 小模型/构造输入验证；真实 GPU 长轨迹验收仍需单独记录。

完成 B1 运行后，可用 `scripts/analyze_page_rco.py` 对完成状态的 run 生成一个新的、
不覆盖原工件的 JSON 报告。RCO 仅在两步都具有完整 k 个有效 Page、且
`N_s >= k`、`N_t > k` 时计算；无效比较保留 `valid_pair_count=0`，绝不以零代替。
对 RULER 运行，`scripts/validate_b1_ruler.py` 从已保存的 generated IDs 解码并检查
冻结的 `reference_answer`；它只生成新的验证报告，不会重新运行模型或修改 B1 工件。

正式 B1 离线分析先运行 `scripts/analyze_b1_run_set.py`。当前第一阶段只读核对 Run Set、
冻结样本、每批 Schema、Manifest 文件集合/Hash、样本元数据、Decode 分片连续性及每批
Page RCO 坐标，成功后在全新目录中仅创建不可覆盖的 `analysis_plan.json`；它不会计算
指标，也不会提前生成 `status=completed` 的 `analysis_manifest.json`。正式 100 条输入时
`--expected-batches 10 --expected-samples 100` 为默认值，且必须显式记录与当前干净 `HEAD`
完全一致的分析代码提交：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_b1_run_set \
  --run-set data/runs/<run-set>.manifest.json --source-root . \
  --page-rco-report data/runs/<batch-01>.page-rco.json \
  --page-rco-report data/runs/<batch-02>.page-rco.json \
  --analysis-id <new-analysis-id> --analysis-git-commit <exact-commit> \
  --output-dir data/analysis/<new-analysis-id>
```

10 个批次的 Page RCO 报告均需逐项传入。来源验证会读取并哈希全部正式 Trace，因此这是
一次有意的完整输入门，不应把耗时误认为重新运行模型。

第 2 阶段的只读核心接口为
`summarize_query_and_page_metrics(index)`：每条 Trace 只读取一次，同时返回 Query Head 28 项
标量和 KV Head 双 Scope Page Mass/Coverage/有效率。样本内保存 mean、median、P10、P90 和
有效数；条件级 Profile 只对样本均值等权。Decode 仅在单样本 `C>=32` 时按其自身归一化进度
生成四段，RULER 等短轨迹的分段数组保持 `NaN` 且有效数为零。该接口当前只返回 NumPy
结构，尚未接入正式 completed 分析工件的 CLI。

第 3 阶段的 `aggregate_page_rco` 读取既有 Page RCO 报告，在每个固定 Head 坐标上输出样本
等权主 Profile 和 `valid_pair_count` 加权敏感性 Profile；条件级标量先在每条样本内等权汇总
Head，再执行 PCG64 固定种子的样本级 Bootstrap。PG19 仅按冻结 `pair_id` 计算
`32K-8K` 配对差异。CLI 增加 `--run-stage3` 后会写出 `stage3_profiles.npz`、
`stage3_summary.json` 和 `status=partial` 的 `stage3_manifest.json`，但仍不会创建
`analysis_manifest.json`：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_b1_run_set \
  <上述来源参数> --run-stage3 \
  --bootstrap-replicates 2000 --bootstrap-seed 20261103
```

这些是第 1～3 阶段的可审计中间工件；Token 稳定性和多指标关联完成前不能标记正式分析完成。

第 4 阶段的 `token_stability.py` 从 Decode Top-64 位置计算 `k={16,32,64}`、九档 Δ 的精确
集合重合、Token RCO、更替率均值/P90、共同 Token Rank 位移，以及逻辑年龄中位数/P90。
每个 Step 的 Sentinel、重复位置、候选数和可见范围均重新检查；零共同 Token 只令 Rank 位移
为空，不会丢弃其他有效指标。条件级主结果按样本等权并执行固定种子 Bootstrap，PG19 仍按
冻结 pair 输出 `32K-8K`。CLI 使用 `--run-stage4` 创建独立的
`stage4_token_profiles.npz`、`stage4_token_summary.json` 和 partial Manifest：

```bash
.venv/bin/python -m experiments.head_analysis.scripts.analyze_b1_run_set \
  <上述来源参数> --run-stage4 --token-comparison-chunk-steps 16 \
  --bootstrap-replicates 2000 --bootstrap-seed 20261103
```

Token 比较采用受控 Step 分块，只保存样本/Head 汇总，不保存数亿条 Step-pair 明细。

## 资源计量边界

`runners/resource_metrics.py` 由 Baseline 与 B1 Runner 共同使用。它在标准 Greedy
生成中将第一个根模型 `forward` 记为 Prefill、其后每个根模型 `forward` 记为 Decode；CUDA
使用同一设备流上的 Event，并在样本边界同步后报告 Decode 的总时间、均值、p50/p95、最小和
最大值。CPU 回归测试使用 `perf_counter`，不应把它当作 GPU 性能结果。

每条样本另保存：`generation_end_to_end_seconds`（包围 `model.generate` 的 host wall time，
不含 B1 写盘）、峰值显存，以及 B1 的 `trace_write_wall_seconds`、样本逻辑数组字节数和
实际样本文件字节数。根模型前向时间不包含 Python 生成控制、token 选择或写盘；这些范围不得
混为吞吐量或硬件加速结论。

## 7.5 Pilot 运行前检查

`scripts/prepare_b1_pilot.py` 固定 10 条 Pilot 输入和当前 B1 配置。它在不加载模型的前提下
检查冻结集、精确 Git 提交、模型/Tokenizer 快照、目标 GPU 的空闲显存和计算进程、运行盘空间
以及所有输出路径尚不存在；通过后只写出一个不可覆盖的 JSON 启动计划。计划中的 Baseline 与
B1 参数可以直接用于远端 tmux 运行，其中 B1 会携带批次预算和 `--stop-on-failure`。

```bash
.venv/bin/python -m experiments.head_analysis.scripts.prepare_b1_pilot \
  --input data/fixed_v2/prompts.jsonl --manifest data/fixed_v2/manifest.json \
  --model <local-model-snapshot> --tokenizer <local-model-snapshot> \
  --device cuda:N --expected-commit <exact-headkv-commit> \
  --baseline-output data/runs/<run-id>.baseline.json \
  --run-dir data/runs/<run-id> --output data/runs/<run-id>.plan.json
```
