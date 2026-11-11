# V1-B1：Attention 分布与时间行为——计划与结果

状态：已完成。正式采集与分析时间为 2026-10-29 至 2026-11-03。
项目背景和共用实验设置见[研究总览](研究总览.md)。

## 1. 研究问题与计划

B1 研究不同 Head 的注意力集中程度、位置偏好和时间稳定性，以及这些特征在任务和输入
长度间是否一致。参考 FlexiCache 的 Page 时间稳定性，进一步比较 Token 与 Page 粒度。

实施顺序为：建立只读捕获与紧凑记录；完成小模型、真实模型和长 Decode 验证；采集
100 条固定输入；离线计算基础分布、多尺度稳定性、跨条件 Head 身份和指标关联。
以上工作均已完成，模型生成与 Baseline 的 Token 序列一致。

## 2. 方法与关键口径

### 2.1 观测与在线统计

AttentionProbe 在 Llama/SDPA 实际 Attention Function 边界读取 RoPE 后 Q 和已更新
Cache 的 K；只读统计路径以 FP32 分块重算概率，不改变模型 Attention 或 Cache。
Prefill 观察最后 64 个 Query，Decode 观察每个 Forward。不保存完整 Attention 矩阵。

第一个生成 Token 来自 Prefill 的最后一个 Query。Decode Step 从零开始，Step `s`
的 Query 位置为 `T0+s`，消费 `generated_ids[s]`，预测下一 Token；成功轨迹长度为
`generated_length−1`。这一区分用于避免首 Token 和时间索引错位。

| 统计层次 | 保存内容 |
|---|---|
| Query Head | Entropy、Normalized Entropy、Top-16/64/256/1024 Mass、Sink/Recent/Prompt/Generated Mass、距离均值与分桶、Token Top-64 ID |
| KV Head | 逻辑 Page size=16 的双 Scope Top-32 ID、Scope Mass、Top-32 Coverage 与有效性 |

Entropy 为 `−Σp log(p)`，Normalized Entropy 除以 `log(N_visible)`；Top-K Mass 为
前 K 个 Token 概率和。距离是 Query 与 Key 的逻辑位置差。

Page 分数为同组四个 Query Head 的概率在 Page 内求和后取平均，按精确 Page Mass
排序；同分时较小 ID 优先。两个 Scope 分别是 `all_visible` 与 `prompt_only`。
Coverage 为所选 Page Mass 除以对应 Scope Mass，不能将 Prompt 条件 Coverage 当成
占全部 Attention 的比例。Prompt 边界 Page 只计入该 Scope 内 Token 的 Mass。

### 2.2 时间稳定性

设两时刻 Top-K 集合为 S(s;k)、S(t;k)，`t=s+Δ`，较晚候选数为 N(t)：

$$
O_k(s,t)=\frac{|S(s;k)\cap S(t;k)|}{k},\qquad
RCO_{raw}=\frac{O_k(s,t)-k/N(t)}{1-k/N(t)}.
$$

主分析使用未截断的 `rco_raw_mean`，保留负值；截为零的 RCO 只作独立辅助量。
随机基线是均匀集合参照，不是显著性检验。两步必须都有完整 K 个对象，且
`N(s)≥k、N(t)>k`；无效比较保留缺失值与有效数，不补零。

Page 使用 `k={8,16,32}`，Token 使用 `k={16,32,64}`；
`Δ={1,2,4,8,16,32,64,128,256}`。另分析 Token 成员更替、共同成员 Rank 位移和年龄。
RULER 每条只有六个 Decode Step，仅 `Δ=1/2/4` 有效。

### 2.3 聚合与统计

先在每条样本内汇总，再对样本等权形成条件级 Head Profile。Step/Step 对加权结果仅作
敏感性比较。长轨迹按自身进度分四段，`C<32` 不做阶段比较。

关联使用 Spearman，Pearson 为补充；同时检查全模型、Layer 去中心化和逐 Layer 结果。
Query Head 指标先在样本内完成时间汇总，再按 `kv_head=floor(q_head/4)` 等权聚合。
去中心化在每层内减去 Head 均值，之后在共同有效坐标计算相关。

Bootstrap 以完整样本为单位，PG19 长度差使用十个冻结 pair；PCG64、2,000 次、
seed=20261103，报告百分位 95% CI。Head、Layer 和 Step 均不充当独立实验样本。

## 3. 主要结果

### 3.1 基础分布

下表先对样本等权，再对 Query Head 等权：

| 条件 | Normalized Entropy | Top-64 Mass | Prompt Mass |
|---|---:|---:|---:|
| GovReport 8K | 0.297 | 0.856 | 0.718 |
| GovReport 16K | 0.286 | 0.848 | 0.714 |
| PG19 8K | 0.276 | 0.873 | 0.694 |
| PG19 32K | 0.260 | 0.857 | 0.702 |
| RULER 8K | 0.263 | 0.903 | 0.904 |
| RULER 32K | 0.251 | 0.884 | 0.903 |

注意力集中在较少 Token，Prompt 仍承载大量质量，但高集中度或高 Prompt Mass 本身
不能代表时间稳定。

### 3.2 稳定性随时间间隔衰减

下表为 `k=32、all_visible`、样本等权后的平均 RCO：

| 条件 | Page Δ=1 | Page Δ=32 | Page Δ=256 | Token Δ=1 | Token Δ=32 | Token Δ=256 |
|---|---:|---:|---:|---:|---:|---:|
| GovReport 8K | 0.749 | 0.490 | 0.257 | 0.605 | 0.210 | 0.086 |
| GovReport 16K | 0.745 | 0.501 | 0.275 | 0.598 | 0.214 | 0.087 |
| PG19 8K | 0.779 | 0.572 | 0.310 | 0.614 | 0.221 | 0.087 |
| PG19 32K | 0.773 | 0.591 | 0.342 | 0.594 | 0.234 | 0.110 |

相邻 Step 稳定不代表长期稳定。Page 聚合会平滑部分 Token 迁移，两种粒度不能替代。
仅 Prompt 的 Page 稳定性通常更低；例如 GovReport 8K 的 Δ=32 为 0.328，而全部可见
Page 为 0.490。分析时不能混用两个 Scope。

### 3.3 Head 差异及跨条件身份

长任务 `all_visible、k=32、Δ=32` 的 Head RCO 分布 P10/中位数/P90 约为
0.322/0.526/0.722，说明 Head 间差异明显。

同一 Head 的相对稳定性在不同输入条件间较一致。Page Δ=32 的 Profile Spearman：
GovReport 16K 对 8K 为 0.993，PG19 32K 对 8K 为 0.983，GovReport 16K 对 PG19 32K
为 0.905。这是 Head 的相对排序相似，不是不同输入关注相同 Token/Page。

跨任务长间隔仍保留结构：GovReport 16K 对 PG19 32K 的 Page Δ=256 相关为
0.766 [0.636,0.871]，Layer 去中心化后为 0.812；相应 Token 去中心化相关为 0.942。
这些结果不能只用不同 Layer 整体偏高来解释。

### 3.4 指标关联与长度比较

Token 与 Page 稳定性在 Δ=32 的 Head Profile 相关约为 0.86～0.92，Δ=256 降至
0.64～0.73。短期稳定性与长期稳定性也有关联，但随跨度增大而减弱。

Entropy、Top-Mass、Prompt Mass、距离和 Page Coverage 与稳定性的关系随任务、Scope
和 Δ 改变，未找到跨任务通用的简单替代量。

PG19 的配对 Token RCO `32K−8K` 在 Δ=32 为 +0.0133 [0.0030,0.0238]，
Δ=256 为 +0.0230 [0.0158,0.0308]；Δ=1 的 CI 跨零。长度优势只出现在部分时间尺度，
幅度较小，不能外推为所有任务的长度效应。

## 4. 结论与限制

B1 支持固定模型内 Head 时间稳定性异质性、随间隔衰减及跨任务相对身份一致性，
与 FlexiCache 描述的现象对应。这里使用真实 Attention Page Mass，不是论文系统的
估计 Page 分数；没有复现其 Page 管理、稀疏计算或性能收益。

B1 没有测量 Prefill CV 或跨阶段覆盖，该问题由[B2](v1_b2_prefill_consistency.md)
接续。现有结果不构成固定二值分类、跨模型规律、KV 表示分布或压缩质量结论。
RULER 40/40 检索正确只说明完整模型的任务表现；PG19 原有重复限制仍需保留。

## 5. 数据入口

以下路径均相对项目根目录，完整 B1 数据保留在远端：

| 入口 | 内容 |
|---|---|
| `data/runs/b1_full_100_20261029_v1.manifest.json` | 100 条正式采集集合 |
| `data/analysis/b1_full_100_completed_20261103_v2/` | 权威完成包 |
| 包内 `head_profiles.npz`、`correlation_profiles.npz` | 单指标、稳定性及关联数值 |
| 包内 `analysis_manifest.json` | 来源、参数和文件索引 |

正式采集、Baseline 输出一致性、分片完整性和离线数值核验均通过。完整字段、执行版本及
核验细节见完成包和 B2 完成包内保存的历史规范快照，不再另设运行、预算或逐步验收文档。
