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
│   └── b1_runner.py          # Manifest 校验、Greedy Decode、组件串联
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
