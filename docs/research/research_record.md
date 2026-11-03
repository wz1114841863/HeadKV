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

2026-10-29 至 2026-10-30 已完成 100 条固定输入的 B1 正式采集、Baseline/B1 一致性与
工件完整性核验，以及双 Scope、多 k、多 Δ 的 Page RCO 离线计算。当前已得到 Page 时间
稳定性随 Δ 衰减、不同 Layer/KV Head 稳定性明显异质、Head 排名在 GovReport/PG19 及
不同长度间较一致的描述性证据。

完整运行来源、派生报告 Hash、统计方法、主要结果、论文对应边界与下一阶段计划见：
[《V1-B1 全量采集与 Page RCO 初步分析结果》](v1_b1_full_analysis_results_v1.md)。

当前状态不是 B1 全部研究完成：Entropy、Top-K Mass、位置/距离、Token Top-K 与 RCO 的
联合分析及样本级统计不确定性尚未完成；CateKV Prefill CV 属于后续 B2，不能由当前结果
替代。

B1 后续离线分析的样本权重、统计单位、Token Top-K 稳定性、GQA 映射、派生工件和代码接口
见：[《V1-B1 离线分析指标与汇总接口规范》](v1_b1_offline_analysis_spec_v1.md)。当前为待审核
草案，审核前不据此生成正式分析工件。
