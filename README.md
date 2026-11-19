# HeadKV

HeadKV 是一个面向大语言模型 KV Cache 的研究项目，研究不同 KV Head 的 Attention 访问行为、K/V 表示结构及这些差异的可利用性。项目基于 KV-Base，技术来源为 NVIDIA KVPress → KV-Base → HeadKV。

研究从 Head 行为与表示差异的发现，推进到这些差异能否转化为质量可接受的 KV 编码、存储和搬运收益，最终面向边缘 NPU/ASIC。

## 当前进展

- **数据准备**：100 条固定输入，覆盖 RULER、GovReport 和 PG19 的六个任务与长度条件。
- **V1-B1/B2**：完成 Attention 分布、Token/Page 时间稳定性、Prefill 一致性及跨阶段关系分析。
- **V2**：完成 100 条 K/V 表示采集及跨输入分析，支持 Head 表示、RoPE 响应和跨 Head 关系存在可复现结构。
- **V3**：完成固定 PCA/Pre-RoPE 重建、Head 预算与共享、Dense 存储模拟，以及量化敏感性和同字节组合对照。局部误差可以优化，但最新组合 Calib 的 K/V 均未通过冻结门槛，该轮停止，不运行组合 Validation。

已有证据支持 Head 异质性及局部画像的用途；尚无可靠的同预算组合预测优势、持续压缩任务质量或实际系统收益证据。结果限于当前模型和输入，未完成跨模型验证。

研究主线、状态和报告入口见[研究总览](docs/research/研究总览.md)；当前阶段的集中结论见 [V3-B 阶段小结](docs/research/v3_head_aware_kv_optimization.md#v3-b-阶段小结)。各阶段的方法、数值和限制保留在对应报告中。

## 项目结构

```text
kvpress/                         KV-Base 框架及 KVPress 代码
experiments/head_analysis/       HeadKV 专用采集、运行、存储和离线分析
  common/                        共用读写与统计
  runtime/                       模型加载与 Attention 捕获
  collectors/                    在线指标采集
  runners/                       实验运行与数据选择
  storage/                       结果读写
  analysis/                      离线指标分析
  scripts/                       数据准备、分析与结果整理入口
  tests/                         研究代码的回归测试
docs/research/                   研究总览与各阶段报告
docs/base/                       KV-Base 架构与扩展说明
data/                            本地或远程数据、模型相关工件和实验结果
```

`data/` 已加入 Git 忽略规则，数据和模型需在实际运行环境中单独准备。已有实验结果保留在对应运行环境；克隆代码不会自动获得这些文件。

## 环境与运行

项目使用 Python 3.10 及以上版本，依赖由 `pyproject.toml` 和 `uv.lock` 管理。使用 `uv` 安装：

```bash
uv sync
```

真实模型采集需要准备模型、固定数据及可用 GPU。当前 Attention 采集链路已验证的范围是 Llama、SDPA、Batch=1；其他模型或 Attention 后端需要分别验证。离线分析只需 CPU，不需要再次运行模型。

数据准备、模型采集、离线分析的具体命令与参数见 [Head Analysis 实验说明](experiments/head_analysis/README.md)。KV-Base 的架构与扩展方式见 [docs/base](docs/base/architecture.md)。

## 测试

运行 Head Analysis 的 CPU 回归测试：

```bash
.venv/bin/python -m pytest experiments/head_analysis/tests -q
```

测试覆盖采集、指标、读写及代表性的运行链路；真实模型和 GPU 行为仍需在目标环境中验证。
