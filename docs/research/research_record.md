# 实验目标
目标一:研究 Head 的分布异质性
分析不同 Head 的 Attention 和 KV 表示在统计特征上是否存在显著差异.
例如,Attention 集中程度/历史 Token 访问位置/KV 数值分布和有效秩等.

目标二:研究 Head 特征的稳定性
判断这种差异是模型相对固有的特征,还是受输入任务/Prompt 内容和 Decode 阶段影响.
重点研究跨输入稳定性/短期与长期时间稳定性,以及 Prefill 对 Decode 的预测能力.

目标三:研究不同 Head 特征之间的关联
例如,Attention 集中程度是否影响时间稳定性?具有相似 KV 表示的 Head 是否具有相似的 Attention 行为?
重点判断不同指标之间是否存在稳定的统计关系,而非预先将 Head 划分为固定类别.

这三个目标分别对应 分布差异 → 分布稳定性 → 分布间关系.

# 实验规划
基于之前开发的KV-Base框架, 针对HeadKV进行开发.

## 实验基础
模型选择: 当前以 Llama-3.1-8B-Instruct 作为验证模型.
数据集处理结果见: [数据集准备](data_preparation.md).当前固定集位于远端 `data/fixed_v2/`,共 100 个输入.后续 HeadKV 实验应直接读取其中的
`prompts.jsonl`;如需按任务分别运行,则读取对应数据集的 JSONL.

## V1实验方案设计
V1方案计划参看: docs/research/research_plan_v1.md

## V1-B1 当前进展

2026-10-29 至 2026-11-03 已完成 100 条固定输入的 B1 正式采集、Baseline/B1 一致性、
原始工件核验及完整 CPU 离线分析。分析覆盖 Query/Page 基础统计、双 Scope 多 k/多 Δ
Page RCO、Token Top-K 稳定性、样本/Pair Bootstrap、GQA 对齐关联、Layer 去中心化关系和
跨任务/长度 Head 身份比较。

权威完成包位于远端：

```text
data/analysis/b1_full_100_completed_20261103_v2/
```

其 completed Manifest SHA-256 为
`e1b0b4fc89b50b7979458b6416dd9b203e046320bdab3c503f7666854f59fd0d`。完整运行来源、阶段
工件 Hash、统计结果、图表、论文对应边界和未验证事项见：
[《V1-B1 全量采集与完整离线分析结果》](v1_b1_full_analysis_results_v1.md)。

当前证据支持固定模型内 Page/Token 时间稳定性异质性、跨条件 Head 身份一致性及二者的显著
关联；不支持固定二值 Head 分类、CateKV Prefill-to-Decode 预测或任何系统效率收益。离线
口径与接口见[《V1-B1 离线分析指标与汇总接口规范》](v1_b1_offline_analysis_spec_v1.md)，
该规范已执行完成并冻结。下一阶段需另行审核 B2 或连续稳定性干预实验，不能静默扩展 B1
结论。
