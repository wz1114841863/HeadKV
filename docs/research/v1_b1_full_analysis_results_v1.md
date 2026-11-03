# V1-B1 全量采集与 Page RCO 初步分析结果（v1）

## 1. 状态与范围

**状态：全量采集与 Page RCO 离线计算已完成；B1 多指标联合分析尚未完成。**

本文记录 2026-10-29 至 2026-10-30 完成的 100 条固定输入 B1 正式采集、工件完整性核验、
双 Scope 多尺度 Page RCO 离线计算及第一轮描述性分析。本文不把描述性结果解释为稳定/
不稳定 Head 的冻结分类，也不证明任何 KV 压缩、Page 调度、推理加速或硬件收益。

指标定义以
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md)
为准；运行批次与保留规则以
[《V1-B1 全量 100 条运行计划与运行集合接口》](v1_b1_full_run_plan_v1.md)为准。

## 2. 运行与数据来源

| 项目 | 冻结值 |
|---|---|
| 运行集合 | `data/runs/b1_full_100_20261029_v1.manifest.json` |
| 运行状态 | `completed` |
| 样本 | 100 条唯一输入：RULER 40、GovReport 40、PG19 20 |
| 模型 | Llama-3.1-8B-Instruct |
| 代码提交 | `8721b46cd5713d285dfb0cf88d1a53a468d5bc71` |
| 固定数据 SHA-256 | `30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee` |
| 精度与 Backend | BF16、SDPA |
| 生成 | Greedy、Batch=1、标准连续 KV Cache |
| Page 配置 | Page size=16、双 Scope Top-32 |
| RCO 参数 | `k={8,16,32}`；`Δ={1,2,4,8,16,32,64,128,256}` |

全量采集共 10 个串行批次，100/100 样本完成。独立复核通过 778 个原始工件的大小与
SHA-256，约 12.27 GB；每条输入 Hash、Baseline/B1 generated IDs、
`decode_trace_length=generated_length-1` 及 Decode Step 连续性均通过。RULER 40/40 条参考
答案匹配。原始 B1 Trace 约 11.427 GiB，逻辑数组约 21.438 GiB，累计 B1 生成时间约
16.927 GPU 小时，最大 reserved 显存约 29.197 GiB，未触发 32 GiB 停止线。

## 3. Page RCO 派生工件完整性

离线任务于 2026-10-30 串行运行，耗时约 2 小时 56 分钟，全部退出码为 0。10 份报告合计
486,771,555 B，包含 1,382,400 条
`(sample,scope,k,delta,layer,kv_head)` 记录及 639,459,840 个有效 Step 对。

后一个数量是跨 Scope、k、Δ、Head 重复统计的计算配对数，**不是独立样本量**，不得据此
构造虚假的超大统计显著性。

| 批次 | 样本 | 记录 | 零有效配对记录 | 报告 SHA-256 |
|---|---:|---:|---:|---|
| GovReport 8K b01 | 10 | 138,240 | 0 | `0a68fe3777048b1fc9247fc2329cc4f9f96815c77fbda24f32d6cc7982e4e4ab` |
| GovReport 8K b02 | 10 | 138,240 | 0 | `57e9e7c5b6f1513beeb1a86e9669af7b89dbea98fc3c57d889395df592246897` |
| GovReport 16K b01 | 10 | 138,240 | 0 | `73546e0445120de5d5ca9592636bb2d9269bc9fa0152a48407537be8aad655e7` |
| GovReport 16K b02 | 10 | 138,240 | 0 | `6ef65660067cb60579f1551d2e25a1d1656e54ea3869700142595a82a4c87a9d` |
| PG19 8K b01 | 5 | 69,120 | 0 | `c07501e846cd8e6f2ded24e0dcefb2687a54396191fbc5026ed02fb6f5630138` |
| PG19 8K b02 | 5 | 69,120 | 0 | `829f2db422a360337cd492519ea35ac3ca082bca7a53fc4f480ca2e517386f30` |
| PG19 32K b01 | 5 | 69,120 | 0 | `a4155bec783a4e45d28e757729b6e35fd2b23677e7398dc8dce4317420890f6e` |
| PG19 32K b02 | 5 | 69,120 | 0 | `934a841c4c9fddae9b54f313f4702cd7048630564d62a80b5cac67ebef1562c2` |
| RULER 8K b01 | 20 | 276,480 | 184,320 | `77447955029ffda44ed6f0767949dfccad128d8107de80d81e668a0f6ae73381` |
| RULER 32K b01 | 20 | 276,480 | 184,320 | `f197dbe343c1cb8f25fd25fd303aff7397802871e8d6a037bac349a4d5613ac9` |

所有报告的 Schema、来源 `run_metadata` Hash、k/Δ、Layer、KV Head、Scope、组合唯一性、
空值规则和指标范围均通过核验。RULER 每条只生成 7 Token，具有 6 个 Decode Step，因此
仅 `Δ=1/2/4` 有效；`Δ>=8` 的零配对记录是正确的不可用结果，不是工件缺失。

## 4. 汇总方法

主表使用 `k=32`，对每条 Head 记录的 `rco_clipped_mean` 按 `valid_pair_count` 加权。另行
检查了样本/Head 等权均值，二者最大差异约 0.009，主要趋势不是个别长输出样本单独造成的。

Head 排名分析先在同一条件内对每个 `(layer,kv_head)` 聚合，再计算跨条件 Spearman 排名
相关。PG19 8K/32K 使用相同书目及续写起点的配对样本；GovReport 8K/16K 不是配对内容，
其差异只能描述，不能解释为上下文长度因果效应。

## 5. 主要结果

### 5.1 多时间尺度 RCO

| 条件 | Scope | Δ=1 | Δ=32 | Δ=128 | Δ=256 |
|---|---|---:|---:|---:|---:|
| GovReport 8K | all-visible | 0.748 | 0.490 | 0.364 | 0.259 |
| GovReport 8K | prompt-only | 0.648 | 0.327 | 0.248 | 0.223 |
| GovReport 16K | all-visible | 0.744 | 0.500 | 0.375 | 0.275 |
| GovReport 16K | prompt-only | 0.620 | 0.308 | 0.227 | 0.201 |
| PG19 8K | all-visible | 0.779 | 0.570 | 0.444 | 0.316 |
| PG19 8K | prompt-only | 0.660 | 0.383 | 0.315 | 0.290 |
| PG19 32K | all-visible | 0.771 | 0.590 | 0.472 | 0.343 |
| PG19 32K | prompt-only | 0.647 | 0.410 | 0.350 | 0.321 |

所有长 Decode 条件均表现出明显的时间衰减：相邻 Step 稳定不代表长时间稳定。PG19 的
长时间 all-visible RCO 高于 GovReport，说明续写与摘要任务存在可见的任务差异，但当前不
把它推广为其他模型或数据集的一般规律。

### 5.2 Scope Mass 与 Top-32 Coverage

| 条件 | Prompt Scope Mass | all-visible Coverage | prompt-only Coverage |
|---|---:|---:|---:|
| GovReport 8K | 0.718 | 0.897 | 0.877 |
| GovReport 16K | 0.714 | 0.884 | 0.858 |
| PG19 8K | 0.689 | 0.918 | 0.900 |
| PG19 32K | 0.702 | 0.896 | 0.875 |

长 Decode 中 all-visible RCO 通常高于 prompt-only，`Δ=32` 时差距约 0.16 至 0.19。Prompt
仍承载约 69% 至 72% Attention Mass，且 prompt-only Top-32 Coverage 为 0.858 至 0.900，
因此较低的 prompt-only RCO 不能简单归因于 Scope 无效或 Coverage 太低。它更接近“原始
上下文中的重要 Page 排名持续变化”，但仍不等价于这些 Page 的实际 Cache 保留收益。

### 5.3 PG19 配对长度结果

| Scope | Δ=1 | Δ=32 | Δ=128 | Δ=256 |
|---|---:|---:|---:|---:|
| all-visible：32K - 8K | -0.006 | +0.020 | +0.027 | +0.033 |
| prompt-only：32K - 8K | -0.013 | +0.023 | +0.027 | +0.024 |

32K 没有提高相邻 Step 稳定性，但较长时间间隔上的 RCO 略高。`Δ=128/256` 时约 66% 至
76% 的配对 Head 记录为正差值。该效应较小，后续需以样本为重采样单位给出置信区间或配对
置换检验，不能把 2,560 个 Head 记录视为独立样本。

### 5.4 Head 空间异质性

聚合 GovReport 与 PG19，在 `k=32, Δ=32` 下：

| Scope | Head P10 | 中位数 | Head P90 |
|---|---:|---:|---:|
| all-visible | 0.322 | 0.526 | 0.722 |
| prompt-only | 0.196 | 0.326 | 0.520 |

`L00H2`、`L01H7`、`L27H3` 在多个时间尺度上相对稳定；`L02H5`、`L05H2`、`L25H5`
相对不稳定。这里的 `H` 是 KV Head。上述名称仅用于定位连续分布两端的候选案例，不构成
冻结的 Stable/Unstable 类别。

### 5.5 跨任务与跨长度排名

在 `k=32, Δ=32` 下：

- GovReport 8K/16K 的 Head 排名相关为 all-visible 0.993、prompt-only 0.972；
- PG19 8K/32K 为 all-visible 0.983、prompt-only 0.957；
- GovReport 与 PG19 跨任务相关为 all-visible 0.884 至 0.907、prompt-only 0.838 至 0.880。

这说明 Page 稳定性存在较强的 Layer/KV-head 结构性成分，但排名并非完全不受任务影响。
`Δ=1` 与 `Δ=128` 的排名相关在 all-visible 为 0.75 至 0.86，在 prompt-only 为 0.43 至
0.64，说明短期稳定性尤其不能充分预测 Prompt Page 的长期稳定性。

### 5.6 Top-k 敏感性

长 Decode 汇总中，all-visible 在 `Δ=1` 时从 `k=8` 的 0.773 降至 `k=32` 的 0.756，
而在 `Δ=128` 时从 0.244 升至 0.401；prompt-only 的变化更非单调。RCO 对预算敏感，后续
报告必须保留多 k 结果，不能把单一 k 的排序直接外推到其他预算。

## 6. 与参考论文的对应边界

本次结果较强支持 FlexiCache 所描述的三个定性现象：KV Head 的 Page 时间稳定性存在明显
差异、RCO 随时间间隔衰减、Head 相对稳定性具有跨任务一致性。它不是 FlexiCache 系统复现：
B1 按真实 Attention Page Mass 排序，而不是论文系统的 MinMax/Quest 类估计 Page 分数；本次
也没有执行论文的 bottom-25% 固定分类、稀疏 Attention、Page 搬移或性能评估。

结果只与 CateKV 的“部分 Head 更一致、部分 Head 更动态”这一宽泛观点相容。B1 没有计算
CateKV 的 Prefill CV、二值化阈值、关键 Token、Prefill-to-Decode Recall 或分类跨样本重合，
因此不能证明 CateKV 的 Prefill 分类能预测 Decode，也不能把高 RCO KV Head 直接称为
CateKV consistent Head。

RULER 40/40 是无压缩完整模型的检索结果，不能证明任何压缩方法仍能保持检索准确率。

## 7. 当前验收边界

当前可以确认：

1. B1 全量 100 条采集及原始工件完整性通过；
2. 双 Scope、多 k、多 Δ Page RCO 离线计算完整；
3. 已获得支持 Page 时间稳定性异质性和跨任务排名一致性的描述性证据。

当前不能确认：

1. Entropy、Top-K Mass、位置偏好和距离与 RCO 的关系；
2. Token Top-K 的时间稳定性和位置迁移；
3. 样本级统计不确定性及预注册阈值下的 Stable/Unstable 集合；
4. CateKV 的 Prefill-to-Decode 顺序一致性；
5. 任何 KV 压缩、推理精度、吞吐、延迟或硬件收益。

## 8. 下一阶段：B1 离线分析冻结

下一阶段仍属于 B1，不立即进入 B2，建议按以下顺序执行：

具体指标、样本权重、统计不确定性、Token Top-K 公式、派生工件和函数接口已在
[《V1-B1 离线分析指标与汇总接口规范》](v1_b1_offline_analysis_spec_v1.md)形成第一版草案，
需审核后再实现。

1. **冻结离线汇总接口。** 明确样本等权、有效 Step 对加权、Head 聚合、GQA Query/KV Head
   映射、Bootstrap/置换检验单位及派生工件 Schema，实现可测试、不可覆盖的汇总脚本；
2. **完成单指标图谱。** 汇总 Prefill/Decode 的 Entropy、Normalized Entropy、Top-Mass、
   Sink/Recent/Prompt/Generated Mass、Distance，以及 Page Scope Mass/Coverage，输出条件分布和
   Layer×Head 热力图；
3. **正式化时间稳定性分析。** 输出多 k/多 Δ 曲线、Head 热力图、样本级置信区间、PG19
   配对检验和 RULER 短轨迹适用性说明；
4. **完成多指标关联。** 在正确的 GQA 映射下研究 Entropy/Top-Mass/Distance 与短期、长期
   RCO 的关系，进行层内和跨任务分析，避免 Layer 与长度混杂；
5. **补充 Token Top-K 分析。** 在编码前冻结 Token 集合重合、位置迁移和突发切换的公式，
   再从已保存的 Decode Top-64 轨迹离线计算；
6. **冻结 B1 结果。** 生成带来源 Hash 的 JSON/CSV/图表和正式研究报告，审核后再决定是否
   增加仅用于论文对照的 bottom-25% Head 集合分析；
7. **另行设计 B2。** B2 开始前冻结 CateKV CV、观察窗口、排除区域、阈值、参考/评估样本
   划分和新 Schema。现有 B1 Trace 不含恢复 CateKV Prefill 观察矩阵所需的完整信息，B2 将
   需要新的受控采集，不能由本次工件直接补算。
