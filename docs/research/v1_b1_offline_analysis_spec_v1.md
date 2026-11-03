# V1-B1 离线分析指标与汇总接口规范（v1）

## 1. 状态、目的与边界

**状态：v1 已审核冻结；正在按第 12 节分阶段实现，尚未生成正式全量分析工件。**

本文冻结 V1-B1 全量 Trace 的离线读取、聚合、统计不确定性、派生工件和函数接口。它承接
[《V1-B1 Attention 指标定义与数据接口规范》](v1_b1_attention_metric_interface_spec.md)
已经冻结的在线指标，不重新定义原始 Attention、Page 排名或落盘数组。

本文适用于以下工作：

- Prefill/Decode Query Head 28 项标量的空间分布和时间变化；
- Decode Token Top-64 的集合稳定性和位置迁移；
- KV Head 双 Scope Page Mass、Coverage 与多尺度 Page RCO；
- 同一 B1 指标在样本、任务、长度、Layer/Head 间的差异；
- Query Head 指标与 KV Head Page RCO 的 GQA 对齐关联。

本文不包含：

- CateKV 的 Prefill CV、关键 Token 或 Prefill-to-Decode Recall；
- Stable/Unstable、Consistent/Adaptive 的主分析二值分类；
- KV 压缩、驱逐、稀疏 Kernel、吞吐、延迟或硬件结论；
- 从紧凑 Trace 恢复完整 Attention Matrix；
- 为追求更强相关性而事后改变样本、指标、k、Δ 或 Head 聚合方式。

## 2. 冻结输入与分析门

### 2.1 权威输入

首个正式分析对象为：

```text
data/runs/b1_full_100_20261029_v1.manifest.json
```

其固定数据 SHA-256、代码提交、模型、生成和采集配置以运行集合 Manifest 为权威。当前已
完成的 Page RCO 报告及 Hash 记录于
[《V1-B1 全量采集与 Page RCO 初步分析结果》](v1_b1_full_analysis_results_v1.md)。

离线工具不得根据目录名猜测数据集、长度或配对关系。`sample_id`、`dataset`、`task`、
`pair_id`、Prompt 长度和生成长度必须来自冻结输入 Manifest 与样本元数据，并交叉核验。

### 2.2 启动前强制检查

正式汇总仅接受满足以下条件的输入：

1. 运行集合状态为 `completed`，10 个批次、100 个唯一 `sample_id` 与冻结集合完全一致；
2. 各批 `run_metadata.json` 状态为 `completed`，运行集合记录的 metadata/file-manifest
   SHA-256 与实物一致；
3. 被读取的 NPZ/JSON 必须在对应 `file_manifest.json` 中，并在读取时核对文件大小与
   SHA-256；
4. `schema.json` 必须为 `headkv-v1-b1-attention-v1`，Feature 顺序、维度、Scope、Page/Token
   Top-K 和距离桶完全一致；
5. 每个样本状态为 `completed`，Decode Step 连续，Trace 长度与生成长度一致；
6. Page RCO 报告的来源 run、metadata Hash、k/Δ、Scope 和记录组合通过校验；
7. 任一来源缺失、Hash 失败、Schema 不兼容或坐标重复时，整次正式分析失败关闭，不静默
   跳过批次、样本、Layer、Head 或 Step。

无效的局部指标按原始 Schema 保留为无效/空值；“输入工件损坏”与“指标在该坐标不可用”
必须分开报告。

## 3. 分析坐标与统计单位

### 3.1 坐标层次

| 层次 | 主键 | 用途 |
|---|---|---|
| Query 观测 | `(sample,phase,step/query,layer,q_head)` | 28 项标量、Token Top-K |
| KV Head Page 观测 | `(sample,decode_step,layer,kv_head,scope)` | Scope Mass、Coverage、Page Top-K |
| RCO Step 对 | `(sample,s,t,layer,kv_head,scope,k,delta)` | Page 时间稳定性 |
| 样本-Head 汇总 | `(sample,condition,layer,head[,scope,k,delta])` | 统计推断的最小 Head 记录 |
| 条件-Head Profile | `(condition,layer,head[,scope,k,delta])` | 热力图和跨任务排名 |

`condition` 至少区分 `dataset × input_length`，即 RULER 8K/32K、GovReport 8K/16K、PG19
8K/32K；不得先合并长度再分析。

### 3.2 独立样本单位

统计不确定性的独立重采样单位是**样本或已冻结的 pair_id**，不是 Decode Step、Step 对、
Query Head、KV Head 或 Layer。数亿个有效 RCO Step 对只是计算配对数，不能作为数亿个独立
观测进行显著性检验。

### 3.3 GQA 映射

Llama 的固定映射为：

```text
kv_head = floor(q_head / 4)
Q(v) = {4v, 4v+1, 4v+2, 4v+3}
```

Query Head 指标保持 `[L,Hq]` 粒度用于自身分布分析。只有在与 KV Head Page 指标关联时，
才构造：

$$
X^{KV}_{i,l,v}=\frac{1}{4}\sum_{h\in Q(v)}X^{Q}_{i,l,h}.
$$

必须先在各 Query Head 上按本文规定完成时间/样本汇总，再进行四个 Query Head 的等权平均；
不得把同名 Query Head ID 与 KV Head ID 直接对应，也不得在原始 Attention Logit 层聚合。

## 4. 通用聚合与不确定性规则

### 4.1 主分析：样本等权

对于标量指标 `x`，先在单一样本、Phase、Layer/Head 内对有效 Query/Step 求均值，并同时保存
时间中位数、P10、P90 和有效数。条件级 Head Profile 对样本等权：

$$
\bar{x}_{c,l,h}=\frac{1}{|I_c|}\sum_{i\in I_c}\bar{x}_{i,l,h}.
$$

每个样本无论实际生成 512、800 或 1024 Token，在条件级主分析中权重相同。跨全部 Step 的
曝光加权均值可以作为诊断项输出，但字段名必须带 `step_weighted`，不能替代主结果。

Page RCO 报告已经在每个 `(sample,layer,kv_head,scope,k,delta)` 内对有效 Step 对求均值。
正式条件级 RCO 主结果对这些**样本级 RCO 均值等权**；按 `valid_pair_count` 加权的结果仅作为
`pair_weighted` 敏感性结果。无有效配对的样本-Head 记录保留为空，不当作零。

### 4.2 Head 分布

每个条件同时输出两种互不替代的视图：

1. `head_profile_distribution`：先对样本平均，再对固定的 256 个 KV Head 或 1,024 个 Query
   Head 报告 P10/P25/P50/P75/P90、均值、标准差、最小和最大；
2. `sample_head_distribution`：所有样本-Head 记录等权，用于观察样本内/样本间混合分布。

不以极值 Head 的单次结果代替总体分布。典型 Head 仅作为后续案例轨迹候选，并记录其选择
规则，不能按图形效果手工挑选。

### 4.3 Decode 时间分段

长 Decode 的阶段变化使用每条样本自己的归一化进度四分段：

```text
segment = min(3, floor(4 * decode_step / C))
```

其中 `C` 是该样本 Decode Trace 长度。仅 `C>=32` 的样本进入四分段比较；更短样本只保留
完整轨迹汇总，不重复填充到四个阶段。RULER 当前 `C=6`，不进入阶段或长期 Δ 分析。

### 4.4 Bootstrap 与配对检验

正式置信区间使用 NumPy `PCG64`、固定 seed `20261103`、2,000 次样本级 Bootstrap，输出
2.5%/97.5% percentile 置信区间及有效样本数。每次重采样必须从原始样本级记录重新计算 Head
Profile 和目标统计量，不能直接重采样 Step 或 Head。

PG19 8K/32K 按冻结 `pair_id` 配对。主效应先在每个 pair 内对固定 Head 集求差，再对 10 个
pair 等权；同时输出 pair-level Bootstrap CI。若报告 p 值，使用 10 个 pair 的完整 `2^10`
符号翻转检验；同一指标族内使用 Benjamini-Hochberg FDR 0.05，并保留原始与校正 p 值。

GovReport 8K/16K 不是内容配对，只报告条件差异及样本级不确定性，不使用“长度导致”之类
因果措辞。RULER 8K/32K 同样不得因样本编号相似而自动视作配对。

### 4.5 相关性

连续指标关系以 Spearman 相关为主、Pearson 为敏感性结果。每个相关项至少输出：

- 相关系数、Bootstrap 95% CI、有效样本/Head 数；
- 全部 256 KV Head 的条件 Profile 相关；
- Layer 去中心化结果：对每层减去该层 Head 均值后再相关，检查 Layer 混杂；
- 每层独立相关的分布，不用单一全模型系数掩盖层间差异。

不得把同一 Head 的大量 Step 当作独立点计算极小 p 值。

## 5. B1 单指标分析

### 5.1 Query Head 28 项标量

Prefill 和 Decode 分开分析，不直接合并 Query 轴与时间轴。至少输出：

- `entropy`、`entropy_normalized`；
- `top_mass_16/64/256/1024`；
- `sink_mass_4`、`recent_mass_64`、`prompt_mass`、`generated_mass`；
- `mean_distance` 及 17 个 Distance Mass 桶。

Prefill 使用实际保存的末端 `W_eff<=64` 个 Query，先在样本内按 Query 等权。Decode 先按
Step 汇总，并额外输出第 4.3 节的四分段结果。

读取时重新检查以下不变量，容差沿用原始指标规范与 float32 精度：

- `0<=entropy_normalized<=1`；
- `top_mass_16<=top_mass_64<=top_mass_256<=top_mass_1024<=1`；
- `prompt_mass+generated_mass≈1`；
- 17 个 Distance Mass 之和≈1；
- 所有普通标量有限，距离非负。

不变量失败属于分析输入错误，不能通过裁剪或删点修复。

### 5.2 KV Head Page 质量

对 `all_visible` 和 `prompt_only` 分别汇总：

- `page_scope_mass`；
- `page_top32_coverage`；
- `page_scope_valid` 有效率；
- 完整 Decode 与四分段的样本等权统计。

`prompt_only` Coverage 必须与 Prompt Scope Mass 联合展示。高 Coverage 或高 RCO 只描述排序
或质量集中程度，不自动等价于保留这些 Page 的推理收益。

## 6. 时间稳定性指标

### 6.1 Page RCO

Page RCO 继续使用已经冻结的公式、两个 Scope、`k={8,16,32}` 和
`Δ={1,2,4,8,16,32,64,128,256}`。正式输出包括：

- 每条件/Scope/k/Δ 的样本等权均值与 Bootstrap CI；
- `pair_weighted` 敏感性均值和有效配对数；
- Layer×KV-head Profile、Head 分布和多 Δ 衰减曲线；
- 短期 `Δ=1` 与中长期 `Δ=32/128/256` 的 Head 排名相关；
- PG19 配对长度差异；
- Scope Mass/Coverage 联合表。

RULER 仅报告实际有效的 `Δ=1/2/4`，不得补齐 `Δ>=8` 曲线或用零代替。

### 6.2 Decode Token Top-K 稳定性

Token Top-K 是 Query Head 指标。对存储的 Top-64 位置恢复 `k={16,32,64}`，时间间隔沿用
Page RCO 的九档 Δ。设 `S_h(s;k)` 为 Step `s` 的前 k 个 Token 位置集合，`N_t` 为 Step `t`
的 `visible_token_count`：

$$
O^{token}_k(s,t)=\frac{|S_h(s;k)\cap S_h(t;k)|}{k},
$$

$$
RCO^{token}_{raw}=\frac{O^{token}_k-k/N_t}{1-k/N_t},\qquad
RCO^{token}_{clipped}=\max(0,RCO^{token}_{raw}).
$$

有效条件为两步均有完整 k 个非 Sentinel 且不重复的位置、`N_s>=k`、`N_t>k`。无效配对保留
零有效数和空指标。该量命名为 `RCO_exact_token`，不得与 Page RCO 混用。

同时输出：

- `token_entry_rate = 1 - O_token`，描述 Top-k 集合更替；
- 共同 Token 的归一化平均 Rank 位移
  `mean(|rank_s-rank_t|)/(k-1)`；无共同 Token 时为空；
- Top-k Token 的逻辑年龄 `query_position-token_position` 的中位数和 P90；
- 更替率的时间分布和高分位数，不在 v1 中预设“突发切换”二值阈值。

Token Top-K 未保存权重，因此这些指标只能描述集合/排名变化，不能恢复 Top-k Attention Mass。

## 7. 多指标关联

正式联合分析至少包括：

| 关系 | Head 粒度 | 主问题 |
|---|---|---|
| Normalized Entropy × Page RCO | Q Group 聚合到 KV Head | 注意力集中是否对应 Page 稳定 |
| Top-Mass × Page RCO | Q Group聚合到 KV Head | 稀疏性是否对应短期/长期稳定 |
| Prompt Mass × prompt-only RCO | Q Group 聚合到 KV Head | 持续关注 Prompt 的质量与排序稳定是否一致 |
| Mean Distance × Page RCO | Q Group 聚合到 KV Head | 远距离访问是否更稳定 |
| Page Coverage × Page RCO | KV Head | 高覆盖集合是否也更稳定 |
| Token RCO × Page RCO | Q Group 聚合到 KV Head | Token 与 Page 时间行为是否一致 |
| 短期 RCO × 长期 RCO | KV Head | 相邻 Step 能否预测长期稳定 |

每项分别按条件、Scope 和必要的 k/Δ 报告；主配置为 Page `k=32`、短期 `Δ=1`、中期
`Δ=32`、长期 `Δ=128/256`，其余 k/Δ 用于敏感性分析。不能在查看结果后更换“主配置”。

### 7.1 Stage5 精确聚合与不确定性接口

Stage5 指标核心只消费已经通过 Hash 审查的 Stage3/Stage4 数值工件，不重新读取模型或重算
在线 Attention；CLI 仍按正式输入门重新核验冻结 Run Set/Trace。两个来源必须具有完全相同的
Run Set Hash、十个 Page RCO 报告 Hash、样本与条件
轴、Bootstrap 参数及主 k/Δ；任一项不一致均失败关闭。

Query Head 指标和 Token RCO 先在**单条样本、固定 Layer** 内按连续 GQA 映射
`kv_head=floor(q_head/4)` 对四个 Query Head 等权平均，再对同一条件的样本等权形成
`[Layer,Hkv]` Profile。不得先跨样本平均 Query Head 后再做结果选择，也不得用 Step 数或
有效 Pair 数给主结果加权。Page 指标和 Page RCO 已是 KV Head 粒度，不重复执行 GQA 聚合。

每条关系固定输出：

- 全部 256 个 `(Layer,KV Head)` 坐标的 Spearman 主相关和 Pearson 敏感性相关；
- 分别在每层减去该层 Head 均值后的全模型相关；
- 32 个逐 Layer 相关及各自有效 Head 数；
- 样本级 2,000 次 Bootstrap 的全模型和 Layer 去中心化 95% percentile CI；
- 关系 ID、两个输入 Profile、条件、有效 Head 数和 Bootstrap 模式。

同一条件内的两个指标使用同一组样本重采样索引。跨条件 Head 身份比较中，PG19 8K/32K
按冻结 pair 联合重采样；其他条件分别独立重采样。Bootstrap 只表达输入样本变化造成的不确定
性，Head 坐标本身不重采样，也不据 256 个 Head 计算伪独立显著性。相关不足 3 个有效 Head、
任一侧零方差或不适用的 RULER 长 Δ 输出空值和实际有效数，不改写为零。

跨条件主分析仅使用 Page/Token `k=32`、`Δ={1,32,128,256}` 的连续 Head Profile；所有
15 个条件对均保留记录，不适用组合保持空值。该接口直接检验“同一 Head 是否跨任务保持
Page/Token 稳定身份”，但不包含 B2 的 Prefill CV 或 Prefill-to-Decode Recall。

## 8. 跨任务排名与论文对照

跨任务/长度稳定性使用连续 Head Profile 的 Spearman 相关与 Bootstrap CI。主分析不按结果
二值分类 Head。

如需与 FlexiCache 的 bottom-25% 不稳定 Head 集合进行次级对照，必须先在独立附录中冻结
论文的 Page 评分来源、k、窗口 W、窗口内 Δ 聚合和频次规则。B1 的
`RCO_exact_page` 基于真实 Attention Page Mass，不能把其 bottom-25% 数值冒充论文 MinMax
评分系统的复现。

CateKV 的 CV 与 Prefill-to-Decode 预测不属于本文。当前 Prefill Trace 只有 28 项标量，不能
恢复 CateKV 的二值 Observation Matrix；B2 必须使用新 Schema 和受控重采集。

## 9. 派生工件接口

### 9.1 目录与不可覆盖规则

正式分析使用新目录：

```text
data/analysis/<analysis_id>/
├── analysis_plan.json
├── analysis_manifest.json
├── condition_summary.json
├── head_profiles.npz
├── page_rco_summary.json
├── token_stability_summary.json
├── correlations.json
├── tables/
│   └── *.csv
└── figures/                 # 图表阶段生成
    └── *.png
```

`analysis_id` 必须新建，目录或任一目标文件已存在时拒绝启动，不覆盖、不追加旧结果。原始
`data/runs/` 全程只读。图表只读取已经完成的数值汇总，不再次定义或重算指标。

### 9.2 `analysis_plan.json`

运行前生成且不可覆盖，至少包含：

```json
{
  "schema_version": "headkv-v1-b1-offline-plan-v1",
  "analysis_id": "...",
  "source_run_set": {"path": "...", "sha256": "..."},
  "source_page_rco_reports": [{"path": "...", "sha256": "..."}],
  "source_schema_version": "headkv-v1-b1-attention-v1",
  "analysis_schema_version": "headkv-v1-b1-offline-analysis-v1",
  "git_commit": "...",
  "parameters": {
    "page_k": [8, 16, 32],
    "token_k": [16, 32, 64],
    "deltas": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20261103,
    "primary_weighting": "sample_equal"
  }
}
```

计划保存完整参数及其规范化 JSON SHA-256。路径可因机器不同而变化，但来源内容 Hash、
Schema 和分析参数必须完全记录。

### 9.3 `analysis_manifest.json`

仅在全部数值分析成功后原子创建，至少包含：

```json
{
  "schema_version": "headkv-v1-b1-offline-analysis-v1",
  "analysis_id": "...",
  "status": "completed",
  "plan_sha256": "...",
  "source_run_set_sha256": "...",
  "sample_count": 100,
  "condition_counts": {"...": 0},
  "outputs": [
    {"relative_path": "...", "sha256": "...", "file_bytes": 0}
  ],
  "warnings": []
}
```

若分析中断，保留计划、日志和已生成临时/诊断文件，但不得创建 `status=completed` 的
Manifest。JSON 中无效数值使用 `null`；NPZ 中允许 `NaN`，并必须带有效数数组。正式 JSON
禁止写出非标准 `NaN/Infinity` 字面量。

### 9.4 数值输出约定

- `condition_summary.json`：样本数、生成长度、28 项标量、Page Mass/Coverage 的条件汇总；
- `head_profiles.npz`：固定顺序的条件、Layer、Query/KV Head 连续 Profile 和有效样本数；
- `page_rco_summary.json`：样本等权主结果、pair-weighted 敏感性、Bootstrap CI；
- `token_stability_summary.json`：Token RCO、更替率、Rank 位移和年龄；
- `correlations.json`：全模型、Layer 去中心化、分层相关及 Bootstrap CI；
- CSV 仅作为便于审阅的扁平导出，JSON/NPZ 为权威数值来源。

所有数组的轴名称、坐标值、dtype 和单位必须写入 NPZ 配套 JSON 或 Manifest，不能依赖代码
中的隐含顺序。

## 10. 代码组织与函数接口

离线分析代码继续独立放在 `experiments/head_analysis/`，不修改 `kvpress/`：

```text
experiments/head_analysis/
├── analysis/
│   ├── page_rco.py              # 已有单运行 Page RCO
│   ├── b1_offline.py            # 读取、验证、聚合与统计核心
│   └── token_stability.py       # Token Top-K 集合指标
├── scripts/
│   └── analyze_b1_run_set.py    # 薄 CLI 与不可覆盖工件编排
└── tests/
    ├── test_b1_offline.py
    └── test_token_stability.py
```

建议的最小核心接口：

```python
load_completed_run_set(path) -> RunSetIndex
validate_analysis_sources(index, page_rco_reports) -> ValidationReport
summarize_query_metrics(index, config) -> QueryMetricSummary
summarize_page_metrics(index, config) -> PageMetricSummary
compute_token_stability(sample_trace, k_values, deltas) -> list[TokenStabilityRecord]
aggregate_page_rco(reports, config) -> PageRCOSummary
analyze_metric_relationships(inputs, config) -> CorrelationSummary
write_analysis_bundle_new(output_dir, plan, outputs) -> AnalysisManifest
```

核心模块只返回 Python/NumPy 数据，不写文件；CLI 负责路径、计划、日志和原子落盘。第一版
核心分析仅依赖现有 NumPy/标准库。若图表需要 Matplotlib，应作为后续显式依赖变更单独审核，
不能在核心统计模块中隐式导入。

## 11. 验证与验收

实现阶段至少覆盖以下 CPU 测试：

1. 构造多样本、不同 Step 长度，验证样本等权与 step/pair-weighted 结果不同且字段不混淆；
2. 验证 Query Head 到 KV Head 的四对一映射和先汇总后聚合顺序；
3. 验证 Token RCO、Sentinel、候选不足、重复位置、零共同 Token 和 Rank 位移；
4. 验证 RULER 短轨迹不会产生 `Δ>=8` 的伪数据；
5. 验证 PG19 pair 对齐、缺 pair 拒绝、Bootstrap seed 可复现和完整符号翻转检验；
6. 验证 Layer 去中心化相关与跨条件 Head 排名；
7. 验证 28 项 Feature 不变量和 NaN/null 规则；
8. 验证来源 Hash/Schema/样本集合失败时整次分析失败关闭；
9. 验证输出目录、计划和结果文件不可覆盖，失败时不生成 completed Manifest；
10. 用小型构造 Run Set 完成端到端 CLI 测试，再读取 10 条 Pilot 或全量工件做只读 smoke。

第一步实现验收命令建议为：

```bash
.venv/bin/python -m pytest \
  experiments/head_analysis/tests/test_b1_offline.py \
  experiments/head_analysis/tests/test_token_stability.py \
  experiments/head_analysis/tests/test_page_rco.py -q
.venv/bin/python -m compileall -q experiments/head_analysis
```

正式全量分析只有在以下条件同时成立时完成：

1. 来源与所有消费文件 Hash 通过；
2. 100 条样本及六个条件完整；
3. 所有主/敏感性权重字段明确，无 Step/Head 伪独立推断；
4. 数值输出、CSV 和图表均绑定同一 completed Manifest；
5. RULER、GovReport、PG19 的适用边界分别报告；
6. 结果报告明确区分描述性证据、统计不确定性、论文对照和未验证系统收益。

## 12. 实施顺序

本文审核后按以下顺序推进：

1. 实现来源索引、Hash/Schema 验证、样本等权通用聚合和分析计划/Manifest；
2. 实现 Query Head 标量和 KV Head Page Mass/Coverage 汇总；
3. 接入既有 Page RCO 报告并生成正式 Bootstrap 汇总；
4. 实现 Token Top-K 稳定性；
5. 实现 GQA 对齐的多指标关联；
6. 在构造数据和 10 条 Pilot 上验收后，运行全量 100 条 CPU 离线分析；
7. 审核数值工件后再增加图表和正式 B1 结果报告。

### 12.1 当前实现进度

第 1 项已完成本地代码与构造工件验收：`analysis/b1_offline.py` 实现只读 Run Set 索引、
冻结样本/PG19 pair 映射、Schema、文件集合/Hash、样本元数据、生成长度和 Decode 分片检查，
并校验每批 Page RCO 坐标；同时提供样本等权主聚合、明确命名的 observation-weighted
敏感性聚合，以及分析计划/完成 Manifest 的纯构造接口。`scripts/analyze_b1_run_set.py` 仅在
来源全部通过后创建新目录和 `analysis_plan.json`，并核对分析代码为指定的干净 Git HEAD；
此阶段不会创建 completed Manifest，也没有读取远端正式 100 条工件。

第 2 项核心汇总也已完成本地构造验收：`summarize_query_and_page_metrics` 单次读取每条 Trace，
返回 Query Head 28 项标量和 KV Head 双 Scope Page Mass、Top-32 Coverage、Scope 有效率；
样本内同时保留 mean/median/P10/P90/有效数，条件级 Profile 对样本均值等权。Decode 四分段
严格使用第 4.3 节公式且只接受 `C>=32`，短轨迹保持空值。读取时重新核验 Entropy、Top-Mass、
Prompt/Generated Mass、Distance Mass、Page Mass/Coverage 的范围、单调性、和约束与 NaN
规则。当前只形成 Python/NumPy 汇总对象，尚未写正式远端数值工件。

第 3 项已完成本地构造验收：`aggregate_page_rco` 在固定
`(condition,layer,kv_head,scope,k,delta)` 上分别生成样本等权主 Profile 与
`valid_pair_count` 加权敏感性 Profile；条件级标量先在单一样本内等权汇总 Head，再使用固定
PCG64 seed 对样本执行 2,000 次 Bootstrap。PG19 按冻结 `pair_id` 输出 `32K-8K` 的 Head 与
标量配对差异及 pair-level Bootstrap CI。`--run-stage3` 只创建
`stage3_profiles.npz`、`stage3_summary.json` 和 `status=partial` 的
`stage3_manifest.json`；不得创建正式 completed Manifest。

第 4 项已完成本地构造验收：`analysis/token_stability.py` 对已保存 Top-64 位置计算
`k={16,32,64}` 与九档 Δ 的精确集合重合、Token RCO、更替率均值/P90、共同 Token 的归一化
Rank 位移及逻辑年龄中位数/P90。实现按 Step 分块执行精确 `k×k` 比较，不保存 Step-pair
明细；Sentinel、重复位置、候选不足、越过可见位置和零共同 Token 均有独立测试。条件级结果
按样本等权并使用冻结 Bootstrap 参数，PG19 输出 `32K-8K` pair-level 差异。远端入口只写
`stage4_token_profiles.npz`、`stage4_token_summary.json` 和 `status=partial` 的
`stage4_manifest.json`，仍不创建 completed Manifest。

第 5 项已开始实现：`analysis/metric_relationships.py` 从 Stage3/Stage4 NPZ 恢复样本级
Profile，执行四对一 GQA 聚合、样本级 Bootstrap、全模型/Layer 去中心化/逐 Layer 相关以及
Page/Token 稳定性的跨条件 Head 身份比较。`--run-stage5` 在读取数值前重新核验两个 partial
Manifest、计划与所有消费文件 Hash，只写 `stage5_relationship_profiles.npz`、
`stage5_relationships.json` 和 partial Manifest。当前仅完成本地构造与 CLI 回归测试，尚未
执行 10 条 Pilot 或远端正式 100 条 Stage5，因此仍不得创建 completed Manifest。

验证命令及当前结果：

```bash
.venv/bin/python -m pytest experiments/head_analysis/tests -q
# 56 passed
.venv/bin/python -m compileall -q experiments/head_analysis
```

下一验收项为第 5 项 10 条 Pilot 与远端正式 100 条关系工件；正式结果通过审查前，不得把
阶段性工件命名为完整 B1 离线分析结果。
