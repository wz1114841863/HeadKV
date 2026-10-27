# Llama Attention Head 分布特性研究__V1 实验方案报告

**版本:V1.1（B1/B2 分段方案，待审核）**

**基础框架:** KV-Base

**实验模型:** Llama-3.1-8B-Instruct

**研究对象:** Attention Head 的空间分布/时间演化及跨输入稳定性

**研究基础:** CateKV/FlexiCache (文件位于docs/paper)

**当前状态:** 100 条固定 Prompt 已准备完成并通过数据校验。已在
`experiments/head_analysis/runtime/attention_probe.py` 实现仅内存捕获的 Llama/SDPA
`AttentionProbe` 原型、在线指标 Collector、分片 Writer、固定输入 ExperimentRunner 及
CPU 小模型/构造输入回归测试；真实 GPU 验收尚未完成。基础指标与数据接口由
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md) 单独冻结。

---

## 1. 研究背景与目标

### 1.1 研究背景

前期分析发现，CateKV 和 FlexiCache 分别从不同角度讨论了 Attention Head 的行为差异。

- **CateKV** 研究 Attention 的跨 Query 一致性,区分具有不同历史 Token 关注行为的 Head.
- **FlexiCache** 研究 Decode 阶段重要 KV Page 集合的时间稳定性,发现不同 Head 的稳定程度存在差异.

两项工作的研究对象、观测阶段及指标定义有所不同，尚不能直接认定它们发现的是相同的 Head 特征。

因此，V1 阶段希望在统一模型、输入和实验环境下，对 Head 的 Attention 行为进行系统统计与比较。

V1 研究的是上述现象，而不是复现 CateKV 的 KV 压缩方法或 FlexiCache 的 GPU Page
管理机制。CV 和 RCO 是可复算的参考量化方法；压缩、估计排序、调度和驱逐均不属于
本阶段交付物。

### 1.2 核心研究目标

**目标一:Head 的 Attention 分布异质性**

研究不同 Layer/不同 Head 在注意力集中程度/历史 Token 位置偏好等方面是否存在显著差异.

**目标二:Head 的 Attention 时间稳定性**

研究同一个 Head 在 Decode 不同时间尺度上的注意力变化,包括短期稳定性/长期稳定性及阶段性变化.

**目标三:Head 特征的跨输入稳定性**

研究同一 Head 的行为是否随 Prompt/任务类型/输入长度和生成阶段变化.

**目标四:不同 Attention 指标之间的关联**

研究 Attention 集中性、跨 Query 一致性、Decode 时间稳定性和 Prefill 预测能力之间是否存在稳定关系。

### 1.3 V1 不研究的内容

本阶段暂不包括:

- KV Cache 的数值分布/PCA/有效秩和跨 Head 表示冗余.
- KV Cache 量化/压缩/驱逐或其他干预策略.
- Attention 或 KV Cache 的计算加速算法.
- 专用加速器设计.
- 完整 Attention Matrix 和完整 KV Cache 的常规持久化保存.

KV 表示分析将在后续 V2 阶段开展.

---

## 2. 实验模型与输入配置

### 2.1 模型配置

| 项目 | 配置 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct |
| Transformer Layers | 32 |
| Query Heads / Layer | 32 |
| KV Heads / Layer | 8 |
| Attention 机制 | GQA |
| KV Cache | 完整/无压缩 |
| 推理方式 | 自回归生成 |
| Decode 策略 | Greedy Decoding |
| Sampling | 关闭 |
| Batch Size | 1 |
| 模型状态 | `eval()`,无梯度 |

实验固定模型权重/Tokenizer/Chat Template/计算精度及 Attention Backend,并保存实际运行版本.

### 2.2 固定实验输入

本阶段使用已经筛选并验证的 100 条 Prompt.

| 数据集 | 输入长度 | 样本数 | 任务 |
|---|---|---:|---|
| GovReport | 8K | 20 | 长文本摘要 |
| GovReport | 16K | 20 | 长文本摘要 |
| PG19 | 8K | 10 | 文本续写 |
| PG19 | 32K | 10 | 文本续写 |
| RULER | 8K | 20 | 长文本检索 |
| RULER | 32K | 20 | 长文本检索 |
| **合计** | | **100** | |

所有实验直接读取已固定的 `input_ids`,不重新构造 Prompt 或再次添加 Chat Template.

RULER 的具体任务类型以已验证的 Manifest 为准,不能把不同 Haystack 类型混记为同一任务.

### 2.3 Decode 长度

初步设置:

| 数据集 | 最大生成长度 |
|---|---:|
| RULER | 128 Token |
| GovReport | 1024 Token |
| PG19 | 1024 Token |

允许模型正常通过 EOS 结束生成.

记录实际生成长度;只有具备相应 Decode 长度的样本才能参与较长时间间隔的稳定性分析.

### 2.4 实验可复现性

每次推理保存:

- Prompt 样本 ID 和输入 Token 哈希.
- 完整生成 Token ID 序列.
- 实际生成长度和结束原因.
- 模型/Tokenizer/软件与 Attention Backend 版本.
- 计算精度和生成配置.
- KV-Base Git Commit 与采集配置.

后续 KV 表示实验可以依据相同 Token 序列进行确定性回放,并核对计算结果.

---

## 3. V1 总体技术方案与分段实施

V1 采用 **在线基础统计 + 离线系统分析** 的混合方案，并分为 B1/B2 两个连续的
实现批次。该划分降低初次开发和正确性验证复杂度，但不预设 B1 的在线计算开销已经
可接受；开销必须由真实模型小规模试运行测量。

| 批次 | 研究重点 | 主要指标 | 可回答的问题 |
|---|---|---|---|
| B1：Attention 基础分布与时间演化 | Decode Attention 的集中性、位置偏好和 Page 稳定性 | Entropy、Top-K Mass、Position Mass、Distance、Token/Page Top-K、`RCO_exact_page` | 哪些 Head 集中、偏向何种历史位置、重要 Page 是否在不同时间尺度稳定？ |
| B2：Prefill 跨 Query 一致性与跨阶段关系 | Prefill 中重要位置的重复出现及其对 Decode 的预测能力 | `CV_prefill`、Prefill 关键 Token、Prefill-to-Decode 覆盖率 | 哪些 Head 可由 Prefill 预测，且这些 Head 是否也具有 Decode 稳定性？ |

B1/B2 属于同一 V1 研究计划，不是两套互不相关的实验。B2 必须沿用 B1 已冻结的
样本、逻辑位置、Head 映射、生成轨迹和基础记录格式；B2 专属数据通过新增文件保存，
不得改变 B1 基础指标的含义。

整体流程为：

```text
固定 Prompt / input_ids
          |
          v
Llama 推理:Prefill + Decode
          |
          v
Attention 只读观测接口
          |
          v
在线统计与特征提取
    |
    |-- 分布特征:Entropy / Top-K Mass
    |-- 位置特征:Distance / Position Mass
    |-- 重要对象:Top-K Token / Page
    |-- B2：Prefill 跨 Query 统计信息
          |
          v
紧凑结构化数据存储
          |
          v
离线统计分析
    |
    |-- 不同 Head 的分布差异
    |-- 多尺度时间稳定性 / RCO
    |-- B2：CV_prefill 与 Prefill 预测能力
    |-- 跨任务 / 跨样本一致性
    |-- 不同指标之间的关联
          |
          v
研究现象总结与假设提出
```

### 3.1 在线部分的职责

在线部分仅负责获得当前 Attention 所需的基础统计和必要轨迹。

其目标是尽可能保留后续离线分析所需的信息，同时避免保存完整 Attention。

在线部分不执行跨任务比较、Head 聚类或显著性检验等研究分析。

### 3.2 离线部分的职责

离线部分基于已经保存的统计数据,研究不同 Head/不同时间段和不同任务的分布规律.

离线分析不依赖模型推理和 GPU，也不需要重新构建完整 Attention Matrix。

当某个新研究问题无法由已保存的数据回答时，允许通过固定 Prompt 和生成轨迹进行针对性重采集。

---

## 4. B1 在线采集与统计方案

B1 的精确公式、坐标、数据类型、文件 Schema、Probe 输入和验证条件以
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md)
为准。本节只保留研究层面的取舍与范围说明。

### 4.1 采集时机

**Prefill 阶段（B1）：**

- 观测每个 Layer 最后 64 个有效 Query.
- 分析这些 Query 对可见历史 Token 的 Attention 分布.
- 记录基础标量特征和查询逻辑位置；不在 B1 建立 Prefill 关键 Token 集合。

**Decode 阶段（B1）：**

- 每个 Decode Forward 均进行观测.
- 覆盖全部 32 个 Layer/32 个 Query Head.
- 记录当前 Query 对所有可见历史 Token 的 Attention 分布特征.
- 保存 Page Top-K 轨迹，并按配置保存 Token Top-K 轨迹。

需区分 Prefill 最后 Query 产生首个新 Token/以及后续 Decode Forward 消费生成 Token 的过程,避免 Step 与 Token 位置错位.

### 4.2 B1 必须指标：Attention 集中性

| 指标 | 统计含义 | 保存方式 |
|---|---|---|
| Attention Entropy | 衡量注意力分布的集中程度 | 标量 |
| Normalized Entropy | 消除可见 Token 数量不同造成的部分影响 | 标量 |
| Top-K Attention Mass | 前 K 个 Token 的注意力质量 | 标量数组 |

Top-K Mass 计算预算：

`K = 16/64/256/1024`

如果可见 Token 数不足 K,则使用实际可见数量,并记录有效 K.

Entropy 必须基于完整可见 Attention 分布在线计算,而不是由保存的 Top-K 近似.

### 4.3 B1 必须指标：Attention 位置分布

记录:

- Sink Token Attention Mass.
- Recent Token Attention Mass.
- Prompt Token Attention Mass.
- Generated Token Attention Mass.
- 平均 Attention Distance.
- 历史距离分桶统计.

初始设置:

- Sink 区域:前 4 个 Token.
- Recent 区域:最近 64 个可见 Token.

不同位置统计可能存在重叠.例如 Sink 和 Prompt 不是互斥类别,不应将全部位置指标直接相加解释为 100%.

位置距离以原始逻辑 Token 位置计算.

### 4.4 B1 可配置指标：重要 Token 轨迹

对于每个 Query Head/每个 Decode Step:

- 计算当前 Attention 权重最高的 Token.
- 保存排序后的 Top-K Token 绝对位置.
- 必要时保存对应权重.

初始建议保存：

`Top-64 Token`

该轨迹用于离线研究 Token 集合重合度、时间漂移、位置切换及长期稳定性。它默认
启用但必须允许以配置关闭；关闭后不影响 B1 的 Page RCO 主指标。

Top-64 可以离线得到较小 K 的集合，但无法恢复 Top-128 或完整 Attention 分布。
全量运行默认仅保存位置，不保存对应权重；权重仅作为可选诊断数据保存。

### 4.5 B1 必须指标：重要 KV Page 轨迹

为了对应 FlexiCache,需要支持 KV Head 粒度的 Page 稳定性分析.

初始设置:

- Page Size:16 Token.
- 每个 KV Head 保存 Top-32 Page.
- 使用对应 Query Group 的 Attention 信息计算 Page 重要性.

V1 优先采用**真实 Attention 概率的 Page 聚合分数**。应先对每个 Query Head 的 Attention 概率进行归一化，再在同一 KV Group 内汇总；不能直接平均未归一化的 Attention Logits。

该结果统一命名为 `RCO_exact_page`，是基于精确 Attention Page Mass 的离线行为
观测指标，不等同于 FlexiCache 的 min/max Key 估计排序，也不是可直接部署的低成本
Page 选择算法。B1 必须同时保存 `all_visible` 与 `prompt_only` 两套 Top-32 Page ID：
前者用于整体 Decode Page 稳定性，后者用于考察对原始 Prompt 的持续关注。两套 Scope
分别计算 RCO，不得混用其 Page ID 或候选 Page 数。全量运行不保存每页分数；Page Mass
仅作为可选诊断数据保存。

### 4.6 B1 数据优先级

| 数据 | 优先级 | 全量运行默认策略 |
|---|---|---|
| Entropy、Top-K Mass | 必须 | 保存标量 |
| Position Mass、Distance | 必须 | 保存标量和距离分桶 |
| KV Head 双 Scope Top-32 Page ID | 必须 | 每个 Decode Step 保存 `all_visible` 与 `prompt_only` ID |
| KV Head Page Scope Mass、Coverage、有效标记 | 必须 | 每个 Decode Step、每个 Scope 保存 |
| Query Head Top-64 Token ID | 可配置 | 默认启用，可关闭 |
| Token/Page 权重 | 可选 | 默认不保存，仅诊断模式可保存 |
| 完整 Attention | 仅 Debug | 仅限指定样本、层、Head、Step |

这一区分只限制数据落盘范围，不减少为计算 B1 必须指标所需的在线 Attention 统计工作。

## 5. B2 预留：Prefill 跨 Query 一致性与跨阶段分析

### 5.1 `CV_prefill` 与 Prefill 关键 Token

为了支持跨 Query 一致性，B2 将在 Prefill 最后 64 个 Query 上建立频次统计。

具体包括:

- 固定的观察 Query 窗口.
- 论文所定义的注意力二值化规则.
- 被判定为重要的历史 Token 位置.
- 各 Token 跨 Query 的出现频次.
- 必要的 Head 聚合结果和统计参数.

由于 Llama 使用 GQA,需要明确区分 Query Head 的统计与将同一 KV Group 内多个 Query Head 聚合后的 KV Head 统计.

`CV_prefill` 用于描述 Prefill 重要性频次是否集中在少数历史位置；较高 CV 不应
直接解释为较高的 Decode 时间稳定性。阈值、排除区域、Query/KV Head 聚合、关键
Token 选择及敏感性分析均在 B2 开始前冻结。CateKV 的原始配置可作为参考条件，
但不自动构成 V1 的默认定义。

### 5.2 Prefill-to-Decode 覆盖率

在 Decode 每一步,利用已经建立的 Prefill 关键 Token 集合,计算其当前 Attention Mass.

建议分别记录:

1. 关键 Token 占当前全部 Attention Mass 的比例.
2. 关键 Token 占当前指向 Prompt 的 Attention Mass 的比例.

两者分别反映跨阶段持续有效性和排除新增 Token 注意力占比变化后的条件覆盖能力.

### 5.3 B2 与 B1 的兼容性

B2 可以为固定 Prompt 和固定生成 Token 序列重新执行模型推理，但必须在相同的
逻辑位置与 GQA 映射体系下增加 B2 专属数据文件。不得重写 B1 的标量定义、Page
ID、Token ID 或其既有文件 Schema。

## 6. 在线计算要求

在线计算应满足:

- 不改变原始 Q/K/V/Attention Output 和 KV Cache.
- 不长期持有原始 GPU Tensor.
- 不构建完整的 Prefill Attention Matrix.
- 对观测所需的 Q/K 采用分块计算,控制临时显存.
- 使用稳定的 Softmax 统计方法,避免长序列数值溢出.
- 采集计算与模型真实 Attention 的 Mask/RoPE/Scaling/GQA 语义保持一致.

在线采集会引入额外计算开销，尤其是在高效 Attention Backend 下重新计算统计量时。
分批实现不等于降低 B1 的计算量；B1 必须在真实模型的两条 8K 样本上记录相对基线的
运行时间、峰值显存、逻辑数组字节数和实际落盘字节数，再决定全量运行配置。

---

## 7. 离线统计分析方案

离线分析是 V1 的主要研究环节.

### 7.1 B1 单指标分布分析

研究不同 Head 的:

- Attention Entropy 分布.
- Top-K Attention Mass 分布.
- 历史位置偏好.
- Attention Distance.

主要输出:

- Layer × Head 热力图.
- Head 指标分布直方图.
- 不同任务和长度条件下的统计分布.
- 典型 Head 的时间变化曲线.

第一步只描述连续分布,不预先进行 Stable/Unstable 或 Consistent/Adaptive 二分类.

### 7.2 B1 时间稳定性分析

基于在线保存的 Top-K Token/Page 轨迹,计算:

- 相邻 Step 的重要集合重合度.
- 不同时间间隔的重合度.
- `RCO_exact_page` 随时间间隔变化的曲线.
- 稳定性随时间间隔变化的曲线.
- 重要位置是否存在突发性切换.

初始时间间隔:

`Δ = 1/2/4/8/16/32/64/128/256`

RCO 应明确固定 Top-K 预算/Page 数量与随机重合基线.可见 Page 数不足预算的 Step 不参与不适用的计算.

### 7.3 B2：跨 Query 一致性与 Decode 稳定性对照

在同一模型/相同样本/对应 KV Head 上比较:

- `CV_prefill`.
- Decode 短期 Page 稳定性.
- Decode 长期 Page 稳定性.
- Prefill 关键 Token 对 Decode 的预测能力.

主要研究:

1. CV 高的 Head 是否具有更高的时间稳定性?
2. 短期稳定的 Head 是否也长期稳定?
3. Prefill 一致性与 Decode 稳定性是否存在对应关系?
4. 哪些 Head 不符合上述直觉关系?

重点分析连续指标的相关性,而不是只比较预设比例的 Head 分类重合率.

### 7.4 跨输入稳定性分析

以固定的 Layer 和 Head 为单位,统计同一指标在不同样本/任务和上下文长度中的变化.

分析内容包括:

- Head 指标均值与方差.
- Head 排名的跨样本一致性.
- 不同任务下的分布差异.
- 不同生成时间区间的特征变化.
- 上下文长度改变对 Head 特征的影响.

由于 GovReport 的 8K/16K 样本不是严格配对的同一内容,长度比较不能简单解释为上下文长度的因果效应.

对于 PG19 配对样本,可以进一步分析保持续写起点相同条件下,增加较早历史上下文是否影响 Head 行为.

### 7.5 多指标关联分析

重点分析以下关系:

| 指标组合 | 主要问题 |
|---|---|
| Entropy × RCO | 注意力集中程度是否与时间稳定性相关? |
| Top-K Mass × RCO | 稀疏注意力是否更稳定? |
| `CV_prefill` × `RCO_exact_page`（B2） | 跨 Query 一致性与 Decode 稳定性是否相关？ |
| 短期 RCO × 长期 RCO | 短期稳定能否预测长期稳定? |
| Prefill Recall × RCO | Prefill 预测能力与 Decode 稳定性是否相关? |
| Distance × RCO | 远距离访问行为与稳定性是否存在联系? |

分别进行全模型/层内和跨任务统计,避免 Layer 或输入长度造成的混杂影响.

本阶段以相关性和分布解释为主,暂不强行进行 Head 聚类.

---

## 8. 数据保存与复现方案

### 8.1 存储原则

采用 **元数据 + 标量数组 + Top-K 轨迹 + 少量 Debug 数据** 四类记录.

默认不保存:

- 完整 Prefill Attention Matrix.
- 所有 Decode Step 的完整 Attention Weight.
- 完整 KV Cache.
- 逐 Step 重复的历史 K/V.
- 与已保存数据重复的中间 Tensor.

### 8.2 B1 建议数据结构

对于 Decode,逻辑上采用:

| 数据 | 维度 |
|---|---|
| Attention 标量特征 | `[Step, Layer, QueryHead, Feature]` |
| Top-K Token 位置 | `[Step, Layer, QueryHead, K]` |
| `all_visible` Top-K Page 位置 | `[Step, Layer, KVHead, P]` |
| `prompt_only` Top-K Page 位置 | `[Step, Layer, KVHead, P]` |
| Page Scope Mass、Coverage、有效标记 | `[Step, Layer, KVHead, Scope]`，`Scope=2` |
| Query 绝对位置 | `[Step]` |
| 生成 Token 序列 | `[GeneratedLength]` |

其中，Token K 默认 64，Page P 默认 32，Page Scope 的固定顺序为
`[all_visible, prompt_only]`。两套 Page 数组及其 Scope 级数据均为 B1 必选；完整
字段名、dtype、无效值和候选 Page 数规则以指标规范为准。

需要保存有效 KV 长度、Head 映射以及必要的数组有效性标记。B2 的 Prefill 关键 Token
和频次统计另存为新增 B2 文件，不写入或重定义 B1 文件。

### 8.3 存储文件

建议每个 Sample 独立保存:

```text
runs/<run_id>/
├── run_metadata.json
├── config_snapshot.yaml
└── samples/<sample_id>/
    ├── sample_metadata.json
    ├── generated_ids.json
    ├── prefill_stats.npz
    ├── decode_stats_000.npz
    ├── decode_stats_001.npz
    └── debug/                 # 可选
```

Decode 轨迹按固定 Step 数分片,初始建议每 128 Step 一个文件.

JSON 用于元数据,压缩数组用于高频统计.最终格式必须具有明确的 Schema Version,保证不同运行结果可合并分析.

### 8.4 调试采集模式

单独提供 Debug 模式,允许选择特定:

- Sample.
- Layer.
- Query Head.
- Prefill Query 或 Decode Step.

仅对这些位置保存完整 Attention 或必要的 Q/K,以便验证统计正确性和绘制热力图.

Debug 模式不作为常规 100 样本实验的默认设置.

---

## 9. KV-Base 接口修改方案

### 9.1 总体原则

保留 KV-Base 原有推理和 Cache 管理机制.

V1 主要在 `experiments/head_analysis/` 中开发独立的采集和分析模块.只有现有接口无法获得必要的 Attention 中间数据时,才修改 `kvpress` 的通用接口.

### 9.2 新增模块及职责

| 模块 | 职责 |
|---|---|
| `AttentionProbe` | 接入实际 Attention 计算,提供只读 Q/K/Mask 和位置信息 |
| `AttentionStatsCollector` | 执行在线 Entropy/Top-K/位置分布等统计 |
| `AttentionTraceWriter` | 将统计量与 Top-K 轨迹分块写入磁盘 |
| `AttentionAnalyzer` | 离线计算 B1 的 RCO/Head 稳定性及指标关联 |
| `ExperimentRunner` | 读取固定 Prompt,运行推理并记录生成轨迹 |

`PrefillAccumulator` 及 B2 的关键 Token/覆盖率逻辑在 B2 单独加入。

### 9.3 现有模块复用范围

**`KVContext`**

继续负责 Layer/Prefill/Decode/Step/Cache 长度等元数据.

**`KVObserver`**

保留原有只读观测功能,不直接承担复杂的 Attention 分布统计.

**`KVRecorder`**

继续负责实验配置/样本元数据和运行事件.

**Attention Function 接口**

根据实际 Llama Attention Backend 选择最小侵入的观测接入方式,避免与现有全局 Attention Patch 冲突.

需要先在 CPU 小模型上确认能够获得实际 Attention 计算使用的 Q/K 和位置信息,再确定是否需要改动 `kvpress/attention_patch.py`.

---

## 10. B1 正确性验证与验收标准

### 10.1 CPU 基础测试

GPU 不可用期间,使用随机初始化的小型 Llama GQA 模型验证:

1. Observer 开启和关闭后的模型输出等价.
2. Prefill/Decode 阶段及 Query 位置识别正确.
3. Query Head 与 KV Head 映射正确.
4. 在线 Attention Entropy 与完整参考 Attention 结果一致.
5. Top-K Token/Top-K Mass 和 Page 聚合结果正确.
6. 分块计算与非分块参考计算在数值容差内一致.
7. 数据保存与重新加载无索引丢失.
8. Hook 退出后模型原有计算行为得到恢复.

### 10.2 GPU 小规模验证

GPU 恢复可用后:

- 先对两条 8K Prompt 测试完整 B1 采集流程.
- 检查 Greedy 生成轨迹是否与无采集运行一致.
- 验证真实 Attention Backend 下的数值一致性.
- 测量在线统计增加的时间和显存开销.
- 检查单样本实际落盘数据量.
- 只有资源测量满足预先确认的运行预算后，才扩展至 16K/32K 和全部 100 条输入.

若某些较长生成样本自然提前结束,应记录实际可观察的 Step 范围,而不是强行补齐.

### 10.3 B1 验收条件

B1 开发完成的标准:

- 能稳定采集全部 Layer/Head 的 Prefill 末端和 Decode Attention 基础统计.
- 无须保存完整 Attention 即可离线分析时间稳定性.
- 能计算 `RCO_exact_page`，并重新加载不同任务结果开展 Head 分布与时间稳定性研究.
- 记录完整生成 Token 轨迹,支持后续 KV 特征回放.
- 在线采集不会改变模型语义,并具有明确的资源开销报告.

---

## 11. 实施计划

### 阶段 A：冻结 B1 规范

审核并冻结 B1 数据结构、在线计算规则、Head/Token 位置约定和文件格式；本阶段不修改模型代码。

### 阶段 B1：基础 Attention 行为采集

实现 Prefill/Decode 基础统计、Top-K Token/Page 轨迹、TraceWriter 和 CPU 精确性测试。

### 阶段 B1-分析：基础行为图谱

实现单指标分布、多尺度 `RCO_exact_page`、跨样本与跨任务分析。

### 阶段 B1-试运行：资源测量与扩展

待 GPU 可用后，先对两条 8K 固定 Prompt 测量资源开销，再据结果决定是否扩展运行。

### 阶段 B2：Prefill 一致性及跨阶段分析

在不修改 B1 基础定义和文件 Schema 的前提下，冻结 B2 参数，增加 `CV_prefill`、
Prefill 关键 Token 和 Prefill-to-Decode 覆盖率，并开展与 B1 时间稳定性的联合分析。

---

## 12. V1 预期成果

V1 预计形成以下研究结果:

1. **Attention Head 空间分布图谱**:各层 Head 的集中性/稀疏性及位置偏好.
2. **Attention Head 时间稳定性图谱**:短期与长期稳定性曲线,以及典型变化模式.
3. **CateKV 与 FlexiCache 联合分析**:CV/RCO/Prefill Recall 等指标之间的关系.
4. **Head 跨任务稳定性报告**:分析哪些统计特征较为稳定,哪些明显依赖输入或生成阶段.
5. **V2 研究问题清单**:根据 V1 结果提出需要进一步通过 K/V 表示特征验证的假设.

V1 不预设一定存在固定的 Head 类别,也不要求统计关联必然显著.未发现预期关联,同样属于有效的实验结果.

---

## 13. V1 方案结论

本阶段采用以下确定的技术路线:

**固定 Prompt → Attention 在线基础统计 → 保存紧凑统计和关键轨迹 → 离线分析 Head 分布/稳定性及指标关联 → 提出下一阶段研究假设.**

先研究 Attention,再研究 KV 表示.

在线部分重点保证统计正确性/信息充分性和存储可控性;离线部分重点保证研究分析的可复现性与指标之间的可比性.

**本报告经确认后,再进入 KV-Base 代码实现阶段.**
