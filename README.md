# HeadKV

HeadKV 是一个面向大语言模型 KV Cache 的研究项目，主要研究不同 Attention Head 的访问行为和潜在规律。项目基于 KV-Base，技术来源为 NVIDIA KVPress → KV-Base → HeadKV。

当前研究关注三个问题：不同 Head 的行为有何差异；这些差异在时间、输入和任务之间是否稳定；Prefill 阶段的特征与后续 Decode 行为之间是否存在关联。

## 当前进展

- **数据准备**：构建并验证 100 条固定输入，覆盖 RULER、GovReport 和 PG19 的六个数据集与长度条件。
- **V1-B1：Attention 分布与时间行为**：完成相同输入上的 Attention 统计、Token/Page 时间稳定性和跨条件分析。
- **V1-B2：Prefill 一致性与跨阶段关系**：回放 B1 生成轨迹，分析 Prefill 频次集中度、Decode 覆盖及其与时间稳定性的关系。

现有结果表明，在当前模型和输入条件下，Head 的时间行为存在差异，部分 Prefill 特征与 Decode 稳定性相关。这些发现仍限于当前模型、输入和生成轨迹；项目尚未完成跨模型验证、因果干预或压缩质量与系统效率评估。

研究目标、实验设置和阶段结论见[研究总览](docs/research/研究总览.md)；具体方法、数据和结果见[数据准备](docs/research/data_preparation.md)、[B1 报告](docs/research/v1_b1_attention_analysis.md)和 [B2 报告](docs/research/v1_b2_prefill_consistency.md)。

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
