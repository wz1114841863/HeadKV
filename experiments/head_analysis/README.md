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
│   └── resource_metrics.py   # Prefill/Decode 前向计时与资源记录契约
├── analysis/                 # 完成工件的只读离线分析
│   └── page_rco.py           # 双 Scope、k/Δ 敏感性 Page RCO
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

## 资源计量边界

`runners/resource_metrics.py` 由 Baseline 与 B1 Runner 共同使用。它在标准 Greedy
生成中将第一个根模型 `forward` 记为 Prefill、其后每个根模型 `forward` 记为 Decode；CUDA
使用同一设备流上的 Event，并在样本边界同步后报告 Decode 的总时间、均值、p50/p95、最小和
最大值。CPU 回归测试使用 `perf_counter`，不应把它当作 GPU 性能结果。

每条样本另保存：`generation_end_to_end_seconds`（包围 `model.generate` 的 host wall time，
不含 B1 写盘）、峰值显存，以及 B1 的 `trace_write_wall_seconds`、样本逻辑数组字节数和
实际样本文件字节数。根模型前向时间不包含 Python 生成控制、token 选择或写盘；这些范围不得
混为吞吐量或硬件加速结论。
