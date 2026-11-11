# V1-B2：Prefill 一致性与跨阶段关系——计划与结果

状态：已完成。正式采集与分析时间为 2026-11-03 至 2026-11-10。
共用实验设置见[研究总览](研究总览.md)，Decode 时间稳定性定义见[B1](v1_b1_attention_analysis.md)。

## 1. 研究问题与计划

B2 参考 CateKV 的跨 Query 重要性频次，研究 Prefill 特征能否描述后续 Decode 行为：
频次是否集中于少量位置；这些位置能覆盖多少后续 Attention；频次集中程度与集合覆盖、
时间稳定性是否相关。

原计划依次完成指标定义、小模型与真实 GPU 验证、100 条采集、基础分布和覆盖分析、
CV 与覆盖关联、B1/B2 联合分析及结果整理。以上均已完成。

B2 对同一批输入回放 B1 的生成 Token，采集独立的 Prefill/Decode 统计，不改写 B1。
六条 Pilot 验证无 Probe 对照；正式集合逐 Forward 检查 Top-1 与 B1 一致。
采集链路通过不代表研究假设成立，结论由离线分析给出。

## 2. 方法与关键口径

### 2.1 Prefill 频次与 CV

统计单位为 KV Head。取 Prompt 最后 `W=64` 个 Query，候选历史区间为
`E=[I,T0−R)`，保证所有观察 Query 都能看到全部候选位置。
每个 Query Head 先在 E 内 Softmax，再对共享 K/V 的四个 Query Head 求概率均值，
得到观察矩阵 A，形状为 `[64,|E|]`。这一概率是候选区间内的相对值，不是完整 Attention
的绝对质量。

$$
\tau=\alpha\,\operatorname{Quantile}_{q}(\operatorname{vec}(A)),\quad
B_{ij}=\mathbf{1}[A_{ij}\ge\tau],\quad C_j=\sum_i B_{ij},\quad
CV=\frac{\operatorname{std}_{ddof=0}(C)}{\operatorname{mean}(C)}.
$$

Quantile 使用 FP32、linear 插值；均值为零时标记无效。高 CV 表示频次更集中于少数
Token，不直接表示这些 Token 在 Decode 中仍重要。

| 参数条件 | I | R | q | α |
|---|---:|---:|---:|---:|
| `paper_token_primary`，主条件 | 64 | 64 | 0.984 | 0.8 |
| `paper_token_alpha1` | 64 | 64 | 0.984 | 1.0 |
| `paper_token_k099` | 64 | 64 | 0.990 | 0.8 |
| `paper_token_recent256` | 64 | 256 | 0.984 | 0.8 |

只对 `C>0` 的位置排序，依次按频次降序、概率和降序、位置升序打破并列。
预算为 128/512/2048，实际集合大小为预算与非零频次位置数的较小者，不补零频次位置。
四组参数和三档预算均保留，不根据结果重选主条件。

### 2.2 集合、Recall 与参照

`learned_middle` 为上述排名前缀；`augmented_prompt` 额外加入固定 Initial `[0,I)`
和 Recent `[T0−R,T0)`。两者总大小不同，后者高覆盖不能全部归功于中段排名。

Decode 概率在完整可见位置上 Softmax 后再作 GQA 概率均值。令集合 Mass 为 A、Prompt
总 Mass 为 P，则 `Recall_all=A`，`Recall_prompt=A/P`。P=0 时后者无效；主分析先逐
Step 求 A/P，再样本内汇总，不能用 mean(A)/mean(P) 替代。Recall 指 Attention Mass
覆盖，不是答案准确率。

为控制固定区和有效集合大小，计算：

```text
fixed_mass = augmented_mass − learned_mass
middle_mass = prompt_mass − fixed_mass
b_eff = min(budget, critical_token_valid_count)
uniform_expected_mass = b_eff / candidate_token_count × middle_mass
learned_excess_mass = learned_mass − uniform_expected_mass
```

均匀参照是从同一候选区间抽取等大小集合的期望质量，不提供随机集合方差。
Prompt 条件参照及超额覆盖均逐 Step 除以 P。固定区只保存 Initial/Recent 合计贡献，
当前不能进一步拆分两者。

### 2.3 聚合、时间与关联

每条样本内按有效 Step 汇总，再跨样本等权；保留实际集合大小、有效数和缺失值。
长轨迹 `C≥32` 分四段和 20 个进度区间，另计算归一化时间斜率、AUC 和末段减首段；
RULER 六 Step 不补长轨迹，也不参与长期衰减判断。

主关系预先指定为主参数、预算 512、Learned Prompt Recall 与 CV 的 Layer 去中心化
Spearman。联合分析另关联 B1 的双 Scope Page/Token RCO 和 GQA 对齐后的基础统计。
GQA 时间聚合及去中心化沿用 B1，B2 CV 则直接从组内概率均值计算，不能平均四个 CV。

Bootstrap 使用 PCG64、seed=20261110、2,000 次，完整样本重采样后重建 Profile；PG19
两长度共用十个 pair 的抽样。CI 是逐项 95% 百分位区间，没有同时置信带或 FDR 声明。
逐 Layer 的联合 CI 只计算主配置，其他配置保留逐层点估计和有效数。

## 3. 主要结果

### 3.1 Prefill 集合包含后续访问线索，但固定区贡献很大

主条件、预算 512；覆盖为样本等权、KV Head 等权后的 Prompt 条件 Recall：

| 条件 | CV 均值 | Learned | 等大小均匀期望 | Augmented |
|---|---:|---:|---:|---:|
| GovReport 8K | 4.347 | 5.69% | 1.53% | 81.49% |
| GovReport 16K | 4.399 | 3.87% | 0.79% | 79.17% |
| PG19 8K | 4.113 | 6.34% | 1.09% | 89.05% |
| PG19 32K | 4.436 | 6.14% | 0.31% | 86.12% |
| RULER 8K | 3.873 | 11.26% | 1.18% | 92.75% |
| RULER 32K | 4.089 | 9.60% | 0.33% | 88.45% |

Learned 平均覆盖在六条件均高于均匀期望。Initial/Recent 合计承载约 75%～83% 的
Prompt Attention，因此 Augmented 的高覆盖不能全部解释为 Prefill 排名有效。
名义 2048 预算在 8K 下存在有效排名长度饱和，预算比较必须结合 b_eff。

### 3.2 CV 与覆盖的关系依赖任务和预算

下表均为主条件、预算 512、Prompt 口径的 Layer 去中心化 Spearman [95% CI]：

| 条件 | CV × Learned Recall | CV × Learned 超额覆盖 |
|---|---:|---:|
| GovReport 8K | 0.042 [−0.071,0.160] | 0.177 [0.047,0.301] |
| GovReport 16K | 0.186 [0.089,0.278] | 0.318 [0.211,0.403] |
| PG19 8K | 0.143 [0.058,0.233] | 0.184 [0.103,0.277] |
| PG19 32K | 0.135 [−0.021,0.254] | 0.149 [−0.007,0.267] |
| RULER 8K | −0.339 [−0.356,−0.312] | −0.335 [−0.352,−0.310] |
| RULER 32K | −0.444 [−0.455,−0.424] | −0.440 [−0.450,−0.418] |

高 CV 不是通用的高 Recall 标记。真实文本部分关系较弱或 CI 跨零，RULER 则为负。
集合整体超过均匀期望，与 CV 不能统一排序覆盖能力，可以同时成立。

预算也改变关系：GovReport 8K 的 CV×Learned Recall 从预算 128 的 0.202，变为预算
2048 的 −0.273 [−0.366,−0.162]。加入固定区后六条件 CV×Augmented 相关均为正，
但集合大小和含义已改变。参数敏感性同样不支持重新挑选一个通用正向规则。

### 3.3 CV 与 Decode 时间稳定性的关系更清晰

主参数、k=32；下表为 Layer 去中心化 Spearman [95% CI]：

| 条件 | CV × Token RCO，Δ=1 | CV × Prompt Page RCO，Δ=32 |
|---|---:|---:|
| GovReport 8K | 0.636 [0.540,0.707] | 0.772 [0.696,0.818] |
| GovReport 16K | 0.645 [0.556,0.714] | 0.769 [0.696,0.825] |
| PG19 8K | 0.734 [0.663,0.790] | 0.894 [0.874,0.901] |
| PG19 32K | 0.706 [0.642,0.765] | 0.906 [0.879,0.916] |
| RULER 8K | 0.634 [0.620,0.646] | 不适用 |
| RULER 32K | 0.724 [0.713,0.736] | 不适用 |

四参数、全部有效 k/Δ 的 CV×RCO 方向均为正，逐项 CI 下界也均大于零；这是完整敏感性
网格的方向一致性，不是多重比较校正结论。Layer 去中心化后关系仍明显。
它支持把 CV 作为时间行为的候选信号，尚未测试留出输入上的预测准确率。

### 3.4 覆盖与稳定性不能合并解释

主条件下，Learned Recall 与短间隔 Token RCO 的六条件相关约为 −0.28～−0.50，
Augmented 则约为 +0.45～+0.71。但 PG19 的长间隔 Prompt Page 关系可转为正，
不能把所有粒度、时间间隔统一描述为负向。

Learned 高覆盖通常伴随更高 Decode entropy、更低 Top-64 Mass；Augmented 的关系
相反。这是解释线索，当前没有偏相关控制或因果证据。

### 3.5 时间变化与配对长度

主条件、512 预算的 Learned Prompt Recall 末段减首段，单位为百分点：

| 条件 | 变化 [95% CI] |
|---|---:|
| GovReport 8K | +0.39 [−0.34,1.31] |
| GovReport 16K | −0.12 [−0.42,0.23] |
| PG19 8K | −4.49 [−5.49,−3.54] |
| PG19 32K | −4.26 [−6.04,−2.79] |

PG19 平均覆盖随 Decode 下降，GovReport 未呈现明确统一方向。绝对覆盖下降不意味着
Head 覆盖排序的相关性也下降；PG19 的生成重复限制继续有效。

PG19 `32K−8K` 的 CV 均值差为 +0.323 [0.214,0.431]，Learned Recall 差为
−0.20 [−1.12,0.78] 个百分点；CV×RCO 的主 k=32、Δ=32 配对差 CI 也跨零。
当前不能确认 CV 增加伴随覆盖或相关性提高，跨零也不能证明两长度等效。

## 4. 结论与限制

B2 将三种属性区分开：Prefill 频次集中、固定集合的后续质量覆盖、Decode 重要集合的
时间稳定性。CV 与后者的关联更清晰，但不是通用覆盖分数。

这些结果补充了 B1 对 Head 异质性的观察，并建立 CateKV 式 Prefill 指标与 FlexiCache
式稳定性指标之间的实测关联；不是论文系统或全部指标的严格复现。
单模型、100 条固定输入及已有生成轨迹限制仍然存在，尚无留出预测、跨模型、因果、
固定二值 Head 分类、压缩质量或系统效率结论。

## 5. 数据入口

以下路径均相对项目根目录：

| 入口 | 内容 |
|---|---|
| `data/runs/b2_full_100_completed_20261110_v1.manifest.json` | 正式 100 条采集集合 |
| `data/analysis/b2_full_100_completed_20261110_v1/` | 本地与远端的统一数值和文字包 |
| 包内 `source_index.json`、`sample_index.json` | 阶段来源与样本身份 |
| 包内 `metric_registry.json`、`array_catalog.json` | 精确口径、参数、字段、形状和有效数 |
| 包内 `primary_results.json`、`stages/` | 主结果及全部预算、参数和联合数值 |

100 条采集的输入、回放和连续性检查通过；离线各阶段已完成独立数值核验。
统一包保存 145 个数组，主结果与原数组坐标核对一致。完整核验、Hash 和历史规范见包内
`analysis_manifest.json`、`validation/`、`documents/`，研究报告不再重复流水。
整理时 Head Analysis 全套已有 180 项测试通过；本报告合并不重算任何指标。
