# V1-B1 全量采集与完整离线分析结果（v1）

## 1. 状态、范围与结论摘要

**状态：B1 全量采集、来源核验、单指标分析、Token 稳定性、多指标关联、统一数值包和首批
图表均已完成。**

本文记录 2026-10-29 至 2026-11-03 对 100 条固定输入完成的 B1 正式采集与离线分析。正式
完成包为：

```text
data/analysis/b1_full_100_completed_20261103_v2/
```

最主要的结果是：固定 Llama-3.1-8B-Instruct 中不同 Head 的 Page 与 Token 时间稳定性存在
显著异质性；同一 Layer/Head 的相对稳定性身份在不同任务和输入长度间具有较高一致性，且
Layer 去中心化后仍然存在；Token 与 Page 稳定性高度相关但并不等价；Entropy、Top-Mass、
Prompt Mass、Attention Distance 或 Page Coverage 中不存在一个能跨任务普遍替代时间稳定性
指标的简单标量。

这些结果直接支持本模型与固定数据上的 Head 行为异质性和跨条件结构性，但不构成固定
Stable/Unstable 二值分类，也不验证 CateKV 的 Prefill-to-Decode 顺序一致性，不证明任何
KV 压缩、推理精度、吞吐、延迟或硬件收益。

指标定义以
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md)
为准；离线统计口径以
[《V1-B1 离线分析指标与汇总接口规范》](v1_b1_offline_analysis_spec_v1.md)为准。

## 2. 冻结运行与原始工件

| 项目 | 冻结值 |
|---|---|
| 运行集合 | `data/runs/b1_full_100_20261029_v1.manifest.json` |
| 运行状态 | `completed` |
| 样本 | 100 条：RULER 40、GovReport 40、PG19 20 |
| 模型 | Llama-3.1-8B-Instruct |
| 采集代码提交 | `8721b46cd5713d285dfb0cf88d1a53a468d5bc71` |
| 固定数据 SHA-256 | `30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee` |
| Run Set SHA-256 | `039afecab0135169ccac0e96050a414b3e313d37d6108d7d648cae80bbcbd85d` |
| 精度与 Backend | BF16、SDPA |
| 生成 | Greedy、Batch=1、标准连续 KV Cache |
| Page 配置 | Page size=16、双 Scope Top-32 |
| RCO 参数 | Page `k={8,16,32}`；Token `k={16,32,64}`；九档 Δ |

全量采集共 10 个串行批次，100/100 样本完成。独立复核通过 778 个原始工件的大小与
SHA-256，约 12.27 GB；每条输入 Hash、Baseline/B1 generated IDs、
`decode_trace_length=generated_length-1` 及 Decode Step 连续性均通过。RULER 40/40 条参考
答案匹配。原始 B1 Trace 约 11.427 GiB，逻辑数组约 21.438 GiB，累计 B1 生成时间约
16.927 GPU 小时，最大 reserved 显存约 29.197 GiB，未触发 32 GiB 停止线。

## 3. Page RCO 来源与完整性

10 份只读派生报告合计 486,771,555 B，包含 1,382,400 条
`(sample,scope,k,delta,layer,kv_head)` 记录及 639,459,840 个有效 Step 对。后一个数量在
Scope、k、Δ 和 Head 间重复统计，**不是独立样本量**。

| 批次 | 样本 | 记录 | 零有效记录 | 报告 SHA-256 |
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

所有报告的 Schema、来源 metadata Hash、k/Δ、Layer、KV Head、Scope、组合唯一性、空值规则
和数值范围均通过。RULER 每条只有 6 个 Decode 观测，因此仅 `Δ=1/2/4` 有效；更长 Δ 的
空记录是正确的不可用结果。

## 4. 离线阶段与正式完成包

| 阶段 | 内容 | 代码提交 | 主要工件 SHA-256 |
|---|---|---|---|
| Stage 3 | Query/Page 标量、Page RCO、Bootstrap | `b07c7208...` | `stage3_profiles.npz`: `e69659dd...a8560` |
| Stage 4 | Token Top-K 稳定性 | `cc6a9cb2...` | `stage4_token_profiles.npz`: `8c0694c8...e38ad9` |
| Stage 5 | GQA 对齐关系与跨条件身份 | `0e55d4de...` | `stage5_relationships.json`: `aaed4d6c...112d57` |
| Stage 5.5 v2 | 完成包、CSV 与图表 | `e760d168...` | `analysis_manifest.json`: `e1b0b4fc...9fd0d` |

Stage 3、4、5 的正式运行时间分别约为 6 分 26 秒、2 小时 12 分 44 秒和 15 分 36 秒；
Stage 5.5 v2 约 27 秒。Stage 5 共生成 480 条关系，其中 321 条具有完整 256 Head，159 条
空关系均对应 RULER 长 Δ 等不适用组合。

正式 v2 包的 Manifest SHA-256 为：

```text
e1b0b4fc89b50b7979458b6416dd9b203e046320bdab3c503f7666854f59fd0d
```

它登记 16 个输出、共 231,338,595 B；所有文件已独立复算大小和 SHA-256。主要文件包括：

- `head_profiles.npz`：228,553,124 B，50 个数组，SHA-256
  `e2a794c73d78e69f4750fb9ed20291e9ec21ead56634d7959256fb88364d17dd`；
- `correlation_profiles.npz`：74,914 B，20 个数组，SHA-256
  `33ea94b0dd981584abfa46bb6959bea9d09264dcd50f36e09494169371bcbbfa`；
- 四份 CSV：Page RCO 1,296 行、Token 稳定性 972 行、关系 480 行、条件指标 336 行；
- 四张 PNG：Page/Token 衰减、跨条件 Head 身份矩阵、Token/Page 与短长期关系图。

首个 `...completed_20261103_v1` 包的数值正确，但跨条件图存在标签重叠，并把无效的 RULER
长 Δ 对角线显示为 1。该包保留为诊断工件，不覆盖；v2 修正图表后成为权威包。v1/v2 的
NPZ 与 CSV 逐字节相同，数值结论没有改变。

## 5. 统计口径

- 条件级主结果先在单一样本内汇总，再对样本等权；Step/有效配对加权只作敏感性分析；
- 置信区间以样本或 PG19 `pair_id` 为重采样单位，PCG64、2,000 次、seed `20261103`；
- Query Head 与 Page 指标关联时，先按 `q_head // 4` 将四个 Query Head 等权聚合到 KV Head；
- 相关性以 Spearman 为主，Pearson 为敏感性，并同时报告全模型、Layer 去中心化和逐 Layer；
- GovReport 8K/16K 不是内容配对，只作描述性比较；PG19 8K/32K 使用 10 个冻结 pair；
- 不把 Step、Step 对、Layer 或 Head 当作独立样本制造统计显著性。

下文除特别说明外，稳定性主表使用 `k=32` 和样本等权的 `rco_raw_mean`。

## 6. 单指标结果

### 6.1 Decode Attention 基础统计

以下为样本等权后再跨 Layer/Query Head 的均值：

| 条件 | Normalized Entropy | Top-64 Mass | Prompt Mass | Mean Distance |
|---|---:|---:|---:|---:|
| GovReport 8K | 0.297 | 0.856 | 0.718 | 5,339 |
| GovReport 16K | 0.286 | 0.848 | 0.714 | 10,234 |
| PG19 8K | 0.276 | 0.873 | 0.694 | 5,257 |
| PG19 32K | 0.260 | 0.857 | 0.702 | 19,548 |
| RULER 8K | 0.263 | 0.903 | 0.904 | 5,157 |
| RULER 32K | 0.251 | 0.884 | 0.903 | 20,072 |

输入变长时平均距离随可见范围增加，不应把该绝对量直接解释为稳定性。RULER 的 Prompt
Mass 明显更高，但其轨迹太短，只支持短期比较。

### 6.2 Page Scope Mass 与 Top-32 Coverage

| 条件 | Prompt Scope Mass | all-visible Coverage | prompt-only Coverage |
|---|---:|---:|---:|
| GovReport 8K | 0.718 | 0.897 | 0.877 |
| GovReport 16K | 0.714 | 0.885 | 0.858 |
| PG19 8K | 0.694 | 0.918 | 0.899 |
| PG19 32K | 0.702 | 0.898 | 0.875 |
| RULER 8K | 0.904 | 0.924 | 0.915 |
| RULER 32K | 0.903 | 0.898 | 0.887 |

所有 Scope 有效率均为 1。高 Coverage 表示 Top-32 Page 捕获较多 Scope Attention Mass，
但不等价于这些 Page 的保留收益或压缩后精度。

### 6.3 Page Top-K 时间稳定性

| 条件 | Scope | Δ=1 | Δ=32 | Δ=128 | Δ=256 |
|---|---|---:|---:|---:|---:|
| GovReport 8K | all-visible | 0.749 | 0.490 | 0.364 | 0.257 |
| GovReport 8K | prompt-only | 0.649 | 0.328 | 0.247 | 0.222 |
| GovReport 16K | all-visible | 0.745 | 0.501 | 0.376 | 0.275 |
| GovReport 16K | prompt-only | 0.621 | 0.310 | 0.230 | 0.204 |
| PG19 8K | all-visible | 0.779 | 0.572 | 0.445 | 0.310 |
| PG19 8K | prompt-only | 0.662 | 0.388 | 0.323 | 0.298 |
| PG19 32K | all-visible | 0.773 | 0.591 | 0.472 | 0.342 |
| PG19 32K | prompt-only | 0.649 | 0.411 | 0.350 | 0.322 |

所有长 Decode 条件都随 Δ 明显衰减：相邻 Step 稳定不代表长时间稳定。all-visible 通常比
prompt-only 更稳定；Prompt 仍承载约 69%～72% Attention Mass 且 Coverage 较高，因此差异
不能简单归因于 Prompt Scope 无效。

Page Head 的空间分布也很宽。在 `all-visible,k=32,Δ=32` 的长任务汇总中，Head P10、
中位数、P90 约为 0.322、0.526、0.722；这支持连续异质性，不等价于冻结二值分类。

PG19 配对的 all-visible `32K-8K` 差异在 `Δ=1/32/128/256` 分别为 -0.0063、+0.0198、
+0.0273、+0.0328。短期差异的置信区间跨零，较长时间尺度略高，但幅度仍小；不能把该结果
推广为未配对的 GovReport 长度因果效应。

### 6.4 Decode Token Top-K 时间稳定性

| 条件 | Δ=1 | Δ=32 | Δ=128 | Δ=256 |
|---|---:|---:|---:|---:|
| GovReport 8K | 0.605 | 0.210 | 0.111 | 0.086 |
| GovReport 16K | 0.598 | 0.214 | 0.113 | 0.087 |
| PG19 8K | 0.614 | 0.221 | 0.116 | 0.087 |
| PG19 32K | 0.594 | 0.234 | 0.139 | 0.110 |
| RULER 8K | 0.713 | — | — | — |
| RULER 32K | 0.722 | — | — | — |

Token Top-32 具有约 0.60 的相邻 Step 稳定性，但长任务到 `Δ=256` 只剩约 0.09～0.11；
对应成员进入率接近 0.89～0.91。共同 Token 的成员保留和内部 Rank 位移并不等价，K=16/32/64
也会改变绝对数值，因此不能只用一个 K 或只看成员交集。

PG19 配对的 `32K-8K` Token RCO 差异为：`Δ=1` -0.0200，95% CI
[-0.0522, 0.0026]；`Δ=32` +0.0133，CI [0.0030, 0.0238]；`Δ=256` +0.0230，
CI [0.0158, 0.0308]。32K 的优势只出现在中长时间尺度，且效应较小。

## 7. Head 身份与指标关系

### 7.1 跨任务、跨长度 Head 身份

all-visible Page RCO 的跨条件 Head Profile Spearman：

| 条件对 | Δ=1 | Δ=32 | Δ=256 |
|---|---:|---:|---:|
| GovReport 16K vs 8K | 0.990 | 0.993 | 0.990 |
| PG19 32K vs 8K | 0.965 | 0.983 | 0.963 |
| GovReport 16K vs PG19 32K | 0.935 | 0.905 | 0.766 |
| GovReport 8K vs PG19 8K | 0.907 | 0.888 | 0.744 |
| RULER 32K vs 8K | 0.851 | — | — |

GovReport 16K vs PG19 32K 在 `Δ=256` 的 95% Bootstrap CI 为 [0.636, 0.871]，Layer
去中心化后为 0.812；逐 Layer 32 层均为正，29/32 层高于 0.5。每层只有 8 个 KV Head，
逐层数值只作描述性证据。

Token RCO 的同类关系更高：

| 条件对 | Δ=1 | Δ=32 | Δ=256 |
|---|---:|---:|---:|
| GovReport 16K vs 8K | 0.998 | 0.996 | 0.990 |
| PG19 32K vs 8K | 0.982 | 0.980 | 0.925 |
| GovReport 16K vs PG19 32K | 0.974 | 0.932 | 0.857 |
| GovReport 8K vs PG19 8K | 0.975 | 0.945 | 0.909 |
| RULER 32K vs 8K | 0.975 | — | — |

GovReport 16K vs PG19 32K 的 Token `Δ=256` 关系在 Layer 去中心化后为 0.942。这直接支持
“哪些 Head 更稳定”具有固定模型内的跨条件结构，不只是某些 Layer 整体偏高造成的。

### 7.2 Token 与 Page 稳定性

all-visible Token RCO × Page RCO 的 Head Profile Spearman：

| 条件 | Δ=1 | Δ=32 | Δ=128 | Δ=256 |
|---|---:|---:|---:|---:|
| GovReport 16K | 0.844 | 0.918 | 0.766 | 0.731 |
| GovReport 8K | 0.829 | 0.909 | 0.737 | 0.696 |
| PG19 32K | 0.746 | 0.858 | 0.639 | 0.649 |
| PG19 8K | 0.755 | 0.862 | 0.667 | 0.641 |
| RULER 32K | 0.634 | — | — | — |
| RULER 8K | 0.662 | — | — | — |

二者在 `Δ=32` 最一致，但长期下降到约 0.64～0.73。Page 聚合会平滑部分 Token 迁移，
因此 Token 与 Page 稳定性反映相关但不相同的时间行为，不能互相替代。

### 7.3 短期能否预测长期

all-visible Page `Δ=1` 与后续稳定性的 Head Profile Spearman：

| 条件 | 对 Δ=32 | 对 Δ=128 | 对 Δ=256 |
|---|---:|---:|---:|
| GovReport 16K | 0.893 | 0.862 | 0.828 |
| GovReport 8K | 0.885 | 0.850 | 0.816 |
| PG19 32K | 0.854 | 0.763 | 0.631 |
| PG19 8K | 0.847 | 0.756 | 0.638 |

相邻 Step 稳定通常能部分预测长期稳定，但能力随时间跨度下降。prompt-only 的 `Δ=1` 对
`Δ=256` 仅为 GovReport 0.571～0.611、PG19 0.402～0.443，说明 Prompt Page 的长期身份
更动态。

### 7.4 基础标量能否预测稳定性

结果不支持一个跨任务通用的简单代理：

- GovReport 短期 Normalized Entropy 与 Page RCO 约 -0.27，Top-64 Mass 约 +0.30；
- RULER Top-64 Mass 与短期稳定性约 +0.24～+0.41；
- PG19 短期关系接近零，长期可出现 Entropy 正相关、Top-Mass 负相关；
- Prompt Mass 与 prompt-only RCO 通常较弱或为负，Mean Distance 通常接近零；
- Page Coverage 在 GovReport/RULER 中为中等正相关，在 PG19 长期则较弱或为负。

因此不能使用“高集中度、高 Top-Mass、高 Prompt Mass 或远距离访问 ⇒ 高时间稳定性”的统一
规则。时间稳定性应作为相对独立的行为维度直接测量。

## 8. 与参考论文的对应边界

本次结果直接支持 FlexiCache 所描述的三类定性现象：KV Head 的 Page 时间稳定性存在明显
差异、RCO 随时间间隔衰减、Head 相对稳定性具有跨任务一致性。Stage 5 还表明这种身份在
Layer 去中心化后仍然存在，并扩展到了 Token Top-K。

它不是 FlexiCache 系统复现：B1 按真实 Attention Page Mass 排序，而不是论文系统的
MinMax/Quest 类估计 Page 分数；没有执行 bottom-25% 固定分类、稀疏 Attention、Page 搬移
或性能评估。

结果与 CateKV 的 Head 行为异质性观点相容，但 B1 没有计算 CateKV 的 Prefill CV、二值化
阈值、关键 Token、Prefill-to-Decode Recall 或分类跨样本重合。因此不能把高 RCO Head 直接
称为 CateKV consistent Head，也不能声称 Prefill 能预测 Decode。该问题属于 B2 新采集。

RULER 40/40 是无压缩完整模型的检索结果，不能证明任何压缩方法仍保持检索准确率。

## 9. 当前验收与后续

当前可以确认：

1. B1 100 条正式采集、Baseline 等价性和原始工件完整性通过；
2. Query/Page 基础统计、双 Scope 多 k/多 Δ Page RCO、Token Top-K 稳定性均已完成；
3. 样本等权、样本/Pair Bootstrap、GQA 映射和 Layer 去中心化关系分析已完成；
4. 统一 JSON/NPZ/CSV/PNG 已绑定到同一个 `completed` Manifest 并通过独立哈希核验；
5. 已获得固定模型内 Head 时间稳定性异质性、Token/Page 关联和跨条件身份一致性的证据。

当前仍不能确认：

1. 可跨模型、跨模型规模或跨更多数据集推广的普遍规律；
2. 一个冻结的 Stable/Unstable 或 Consistent/Adaptive 二值 Head 分类；
3. CateKV 的 Prefill-to-Decode 顺序一致性；
4. KV 表示数值分布、有效秩与 Attention 稳定性的关系；
5. 任何 KV 压缩、推理质量、吞吐、延迟、显存流量或硬件收益。

B1 的核心数值分析到此可以冻结。后续若只增加 Layer×Head 热力图、案例轨迹或论文版排版，
应直接读取 v2 完成包，不改变统计口径。下一研究阶段可单独设计 B2：冻结 CateKV CV、观察
窗口、阈值、训练/评估划分和新 Schema 后进行受控重采集；或先设计基于当前连续稳定性指标的
干预实验，但两者都必须建立新的问题定义与验收门，不能把 B1 观察性证据直接当作系统收益。
