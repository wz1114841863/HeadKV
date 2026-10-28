# HeadKV V1-B1 Attention 指标定义与数据接口规范

> **文档版本：** V1-B1 Final v1.0（十节整合审定稿）
> **Schema 版本：** `headkv-v1-b1-attention-v1`
> **研究阶段：** V1-B1（Attention 基础分布与时间演化）；V1-B2 仅预留接口
> **模型与框架：** Llama-3.1-8B-Instruct；KV-Base；Transformers 5.2.0 作为当前代码审阅基准
> **状态：** 指标及接口设计稿；Llama/SDPA 的只读内存捕获 Probe、基础统计 Collector、
> 分片 Writer 与固定输入 ExperimentRunner 已完成 CPU 小模型/构造输入回归；真实 GPU
> 验收尚未完成
> **设计依据：** `research_plan_v1.md` 与 `v1_b1_attention_metric_interface_spec.md`，并合并逐节审阅决议

**核心原则：** 固定输入与生成轨迹；在实际 Attention 调用边界只读观察；在线分块计算基础统计并紧凑落盘；离线分析 Head 空间分布和时间稳定性；不在 B1 采集完整 KV 或实现 CateKV 的 CV。B1 同时保留 `all_visible` 与 `prompt_only` 两套 Page Top-K。

---

## 1. 目的、范围与术语

### 1.1 目的与范围

本规范定义 V1-B1 中 **Prefill 末端** 与 **逐 Token Decode** 阶段的 Attention 基础统计、轨迹采集、接口及持久化协议。在线阶段从当前 Attention 概率分布提取标量统计和少量重要位置；离线阶段用这些记录分析不同 Head 的注意力集中性、位置偏好、重要 Token/Page 的时间稳定性和跨输入差异。**紧凑记录不承诺还原完整 Attention 概率分布。**

B1 研究的是 CateKV、FlexiCache 提出的 Head 行为异质性现象，**不复现**前者的 KV 压缩机制或后者的 Key min/max Page 评分及 GPU Page 调度。B2 才研究 Prefill 跨 Query 一致性、关键 Token 与 Prefill-to-Decode 覆盖率；KV 表示的数值特征、PCA、量化和硬件方案留到后续阶段。

### 1.2 术语

| 术语 | V1-B1 中的准确定义 |
|---|---|
| 精确 Attention | 利用实际 Attention 调用边界的 Q、K、Scaling 和**等效有效可见性约束**，按完整 Softmax 概率公式重算；不做 Top-K 概率近似，不要求与后端计算逐 bit 相同。 |
| Query Head | Llama 每层 32 个 Query Head；Token 概率、Entropy、Top-K Mass、位置特征的统计单位。 |
| KV Head | Llama 每层 8 个共享 K/V 的 Head；Page Mass、Page Coverage、Page Top-K、Page RCO 的统计单位。 |
| 逻辑 Token 位置 | 从 0 开始的输入/生成序列绝对位置，不等于任意物理缓存槽位。 |
| 逻辑 Page | 以 16 个连续逻辑 Token 为一组，`page_id = floor(token_position / 16)`；是**分析单位**，不代表 Cache 物理分页。 |
| `all_visible` | 当前 Query 可见的全部 Token 所在的 Page 范围。 |
| `prompt_only` | 当前 Query 可见、且逻辑位置小于 Prompt 长度的 Token 所在 Page 范围。 |
| `RCO_exact_page` | 根据精确 Attention Page Mass 排序得到的 Page 集合，离线计算随机重合校正值；不同于 FlexiCache 的估计 Page 分数。 |

“精确”指概率定义及全候选范围，不代表 GPU 浮点运算顺序逐位相同。必须记录实际 Attention Backend，并使用独立 eager 参考实现验证定义。

## 2. B1 的输入、运行与不变量

### 2.1 固定输入

输入取自已冻结且通过独立校验的 **100 条 Prompt**。`fixed_dataset.path` 是**配置参数**：由运行者指向真实 `prompts.jsonl`（原草案给出的 `data/fixed_v2/prompts.jsonl` 仅为当前环境示例），不得在采集端复制或重写数据集构建逻辑。

| 数据集 | Prompt 目标长度 | 数量 | 任务 | 每条最大生成长度 |
|---|---:|---:|---|---:|
| GovReport | 8K / 16K | 20 / 20 | 长文摘要 | 1024 |
| PG19 | 8K / 32K | 10 / 10 | 文本续写 | 1024 |
| RULER | 8K / 32K | 20 / 20 | Needle 检索 | 128 |
| **合计** | | **100** | | |

读取固定集已有的 `input_ids`，**不得再次添加 Chat Template 或特殊 Token**。记录 `sample_id`、`dataset`、`task`、`pair_id`、`actual_prompt_length`、`max_new_tokens`、`input_sha256` 等；原始哈希规则沿用固定集 Manifest，不在采集阶段重新定义。RULER 的 `niah_single_1` / `niah_single_2` 任务身份以实际 Manifest 为准，不能改名冒充。

模型采用 `eval()`、无梯度、Batch Size=1、完整连续无压缩 KV Cache、Greedy Decode（`do_sample=false`、`num_beams=1`）。固定并记录模型/Tokenizer revision、计算 dtype、实际 Attention Backend、完整 generation config、Transformers/PyTorch 版本、KV-Base Git Commit 和随机种子。自然 EOS 允许提前终止，不强行填充生成长度。

### 2.2 运行不变量

1. **纯观测：** 不修改模型权重、Q/K/V、Attention 输出、Mask、KV Cache、生成参数或 Backend；模型实际 Attention 函数仅执行一次。
2. **完整 Cache：** 仅支持 Token 逻辑位置从 0 连续增长、能够验证 Key 轴到逻辑位置一一对应的 Cache。压缩、Gather/重排、滑窗截断、不可识别的 Cache/Mask 运行必须拒绝。
3. **执行边界：** 单样本、标准完整 Prefill 加逐 Token Decode；不支持 packed batch、Chunked Prefill、Speculative Decode、并行采集或不受支持的特殊 Attention。
4. **共享 Page 候选集：** 同一 Decode Step 的 `all_visible`/`prompt_only` 候选逻辑 Page 集必须在全部 Layer、Query Head 与 KV Head 间一致；B1 v1 仅接受满足该条件的标准连续 Causal 可见性。出现 Head/Layer 专属的有效可见集合时必须拒绝采集，而非把不同候选数压缩写入单一 `[C]` 数组。
5. **资源边界：** 不长期保留 GPU Q/K/V 引用，不保存完整 Prefill Attention Matrix，不默认写出完整 Attention 或完整 KV。
6. **可追溯性：** 每个样本保存完整 `generated_ids` 及哈希；后续 B2 可按相同 Token 历史回放，并验证数值一致性。

## 3. 坐标体系、生命周期与 GQA 映射

### 3.1 固定维度与索引

| 符号 | 说明 | 默认值 |
|---|---|---:|
| `L` | Transformer 层数 | 32 |
| `Hq` | 每层 Query Head 数 | 32 |
| `Hkv` | 每层 KV Head 数 | 8 |
| `G` | 每个 KV Head 共享的 Query Head 数，即 `Hq/Hkv` | 4 |
| `D` | Head Dimension | 128 |
| `T0` | Prompt 实际 Token 数 | `actual_prompt_length` |
| `W` | Prefill 观察末端 Query 窗口 | 64 |
| `P` | 逻辑 Page Size | 16 |
| `K_token` | Decode Token Top-K 记录预算 | 64 |
| `K_page` | Decode Page Top-K 记录预算 | 32 |

Layer、Head、Step 和 Token 位置全部从 0 开始。Query Head `h` 对应的 KV Head：

$$
v(h)=\left\lfloor \frac{h}{G}\right\rfloor=\left\lfloor\frac h4\right\rfloor.
$$

四个 Query Head 虽共享 K/V，却拥有各自的 Q 和 Attention 概率；B1 的 Query Head 标量和 KV Head Page 指标不能仅按同名 Head ID 直接比较。

### 3.2 Prefill 与 Decode 的时间定义

**Prefill：** 仅记录逻辑位置 `max(0,T0-W)` 至 `T0-1` 的 `W_eff=min(W,T0)` 个 Query。标准完整因果注意力下，位于 `q` 的 Query 可见 `[0,q]`（**包含自身**）；即使 Prefill 最后 64 行，可见 Key 数也各不相同。若实际 Backend 存在其他 Mask，必须使用真实的有效可见集合。

**Decode：** 第一个生成 Token `generated_ids[0]` 由 Prefill 最后一个 Query 的 Logits 产生；第一个 Decode Forward 消费该 Token。对标准 Greedy 逐 Token 流程中的 `decode_step=s`：

$$
\begin{aligned}
\mathrm{query\_position}(s)&=T_0+s,\\
\mathrm{input\_token}(s)&=\mathrm{generated\_ids}[s],\\
\mathrm{predicted\_token}(s)&=\mathrm{generated\_ids}[s+1].
\end{aligned}
$$

在**成功完成的标准流程**中，令实际生成 Token 数为 `G_out`（与 GQA 组数 `G` 不同），则：

$$
\mathrm{decode\_trace\_length}=\max(0,G_{out}-1).
$$

生成 1 个 Token 时 Decode Trace 为 0 条。中断/失败时必须根据实际已采集 Step 记录部分完成状态，不能用上述公式伪称采集已完成。

**跨层对齐：** `decode_step` 是**样本级 Forward 序号**，不是每层 Hook 的累计次数。每个成功 Forward 的全部 32 层必须对应同一 `query_position`、一致的阶段与正确的 Key 逻辑映射；事件主键为 `(sample_id, forward_id, layer_id, phase, query_position)`，Decode 还带 `decode_step`。

## 4. 精确 Attention 概率与在线分块计算

### 4.1 Attention 数学定义与有效可见性

对 Layer `l`、Query Head `h`、Query `q`、可见 Key 逻辑位置 `j∈M(q)`，定义：

$$
z_{l,h,q,j}=s_l\,Q_{l,h,q}^{\mathsf T}K_{l,v(h),j}+b_{l,h,q,j},\qquad j\in M(q),
$$

$$
p_{l,h,q,j}=\frac{\exp(z_{l,h,q,j})}{\sum_{r\in M(q)}\exp(z_{l,h,q,r})}.
$$

- `Q/K` 必须取**模型实际传入 Attention Function 的、完成投影和 RoPE 且 K 已完成 Cache Update** 的 Tensor，不自行重建投影或 RoPE。
- `s_l` 是该 Attention 调用实际使用的 Scaling；`b` 仅表示可见位置上有限的加性 Attention Bias。
- `M(q)` 独立表达排除性 Mask。标准 Causal Attention 下为 `[0,q]`；`attention_mask=None` **绝不自动等于全 Key 可见**。
- 若 Backend 原始 additive mask 已包含有限 Bias，不能在 `b` 中再次相加；Probe 必须能提供等效可见性与 Bias 语义。缺失或无法判定时**拒绝采集**。
- 不允许对每个 Key Block 单独执行 Softmax 后直接拼接为全局概率。

### 4.2 在线双遍分块算法

默认采用分块两遍计算，`key_block_size=1024` 为可调资源参数，不属于指标定义。

**第一遍：** 只选 Prefill 末端 `W_eff` 个 Query 或当前 Decode Query；按 Key Block 计算有效 Logits；以稳定的块级最大值与 Log-Sum-Exp 累积每个 Query Head 的**全局** `logZ`；同时维护最大 `K=max(1024, K_token)` 个 Logit 候选用于 Top-K Mass，以及排序后的 Token Top-K 位置。

**第二遍：** 使用全局 `logZ` 将每个 Key Block 转换为全局概率；计算 Entropy、位置质量、Attention Distance/分桶以及每个 Query Head 的 Page Mass；最后在同 KV Group 内汇总四个 Query Head 的 Page Mass，得到两个 Scope 的 Page 排名、覆盖率和总质量。

每块处理结束即释放临时张量，必要的紧凑统计结果移至 CPU Writer。不得构造全量 Prefill `[T0,T0]` 矩阵，不得因采集而改变模型 Forward 的计算范围。分块控制临时显存，**并不会自动降低重复 QK 的计算量**。

### 4.3 统计精度与数值约束

模型原始 Q/K dtype 不改变；观察路径默认采用 FP32 Logits、Softmax 归一化与统计累积；标量以 `float32` 落盘。FP32 重算是完整概率数学定义，不保证与 BF16/FlashAttention 内部逐位相等。Top-K 并列排序按**统计路径的实际数值**先降序，精确相同再以较小 Token/Page ID 优先。

CPU 参考与 Debug 验证：有效概率和约等于 1；`0≤Entropy≤log(N)`；Top-K Mass 随 K 非降且在 `[0,1]`；距离分桶质量和约等于 1；`all_visible` Page 总质量约等于 1。正式 GPU 不应在每 Step 强制同步 CPU 执行这些断言；错误通过事件/分片级检查报告。

## 5. B1 Attention 指标定义

### 5.1 指标层次与通用约定

**Query Head 指标：** Entropy、Token Top-K Mass、Position Mass、Distance、Token Top-K。
**KV Head 指标：** Page Mass（在线临时）、Page Top-K、Page Coverage、Page Scope Mass；Page RCO（离线）。

记 `N(q)=|M(q)|`。所有标量均由第四节的**完整可见** Attention 概率计算，不由截断 Top-K 近似推断。`N(q)=0` 为非法观测。Prefill B1 只保存 Query Head 基础标量；Decode B1 保存基础标量、Token Top-K（可配置）及两套 Page 数据。B1 **不计算** CV、关键 Token 频次或 Prefill Recall。

### 5.2 Query Head 集中性与稀疏性

**Entropy：**

$$
H_{l,h,q}=-\sum_{j\in M(q)}p_{l,h,q,j}\log p_{l,h,q,j},\qquad
H_{\mathrm{norm}}=\begin{cases}H/\log N(q),&N(q)>1,\\0,&N(q)=1.\end{cases}
$$

同时保存 `entropy`、`entropy_normalized` 和 `visible_token_count`。归一化熵有利于长度对照，但不能消除语义及输入分布混杂。

**Top-K Token Attention Mass：** `S_k(q)` 为当前 Query Head 最高概率的 `min(k,N)` 个可见 Token：

$$
\operatorname{TopMass}_{k}(q)=\sum_{j\in S_k(q)}p_{l,h,q,j},\quad k\in\{16,64,256,1024\}.
$$

保存 `top_mass_16`、`top_mass_64`、`top_mass_256`、`top_mass_1024`。当 `N<k`，实际有效 k 为 N，质量理论上为 1。**计算 Top-1024 Mass 不依赖是否保存 Token Top-64 ID。**

### 5.3 Query Head 位置分布与历史距离

令 Query 逻辑位置为 `a(q)`、Key 逻辑位置为 `a(j)`，只在 `M(q)` 上求和：

| Feature | 位置集合 / 计算方式 | 注意 |
|---|---|---|
| `sink_mass_4` | `a(j)<4` | Sink 与 Prompt 可重叠 |
| `recent_mass_64` | `a(q)-63 ≤ a(j) ≤ a(q)` | 包含当前 Query 自身位置 |
| `prompt_mass` | `a(j)<T0` | 原始 Prompt 区域 |
| `generated_mass` | `a(j)≥T0` | Prefill 时为 0 |
| `mean_distance` | `Σ_j p_j[a(q)-a(j)]` | 逻辑位置的绝对 Token 距离 |

各质量区间可重叠，**只有**标准完整 Cache 下的 `prompt_mass+generated_mass≈1` 是互补分区校验；Sink/Recent 不能再与之求和。

**Distance Histogram：** 对距离 `d=a(q)-a(j)`，以下 17 桶分别保存概率质量：

`0`、`1`、`2–3`、`4–7`、`8–15`、`16–31`、`32–63`、`64–127`、`128–255`、`256–511`、`512–1023`、`1024–2047`、`2048–4095`、`4096–8191`、`8192–16383`、`16384–32767`、`≥32768`。

相对距离可在离线分析中由记录的 Query 位置计算；不单独增加在线字段。

### 5.4 Token 级 Top-K 轨迹（可配置）

对于每个 **Decode Query Head**，将可见 Token 按概率降序、逻辑位置升序打破并列，保存 Top-64 逻辑绝对位置：`token_topk_positions[C,L,Hq,64]`，`uint32`。候选不足时尾部填 `UINT32_MAX`，有效数为 `min(64,visible_token_count)`。

`save_token_topk=true` 为初始默认，可因正式资源测量显式关闭；关闭不影响 Query Head 基础标量或 Page RCO。默认**不保存** Token 权重；如启用诊断，写入 `token_topk_weights`，不能与 `top_mass_64` 混淆。B1 不据 Prefill Token Top-K 构建 CateKV 关键 Token 集。

### 5.5 KV Head 级 Page Mass 与双 Scope Top-K

**逻辑 Page：** `page_id(j)=floor(a(j)/P)`，`P=16`；不足 16 Token 的末页保留，按实际可见 Token 的概率求和，**不对 Page 长度做平均或补零**。

对 KV Head `v`，令 `\mathcal H(v)` 为其对应的四个 Query Head，Page `p` 的重要性：

$$
B_{l,v,q,p}=\frac1G\sum_{h\in\mathcal H(v)}\ \sum_{\substack{j\in M(q)\\\lfloor a(j)/P\rfloor=p}}p_{l,h,q,j}.
$$

必须先对各 Query Head **单独 Softmax 归一化**，再 Page 聚合、再对四个 Query Head 等权平均；不能先平均 QK Logits。

**两个必须保存的 Scope：**

| Scope | Page 候选与分数 | 研究解释 |
|---|---|---|
| `all_visible` | 所有可见 Token 所在 Page；Page 分数为全部可见 Token 的原始概率聚合 | 整体 Decode Page 稳定性 |
| `prompt_only` | 仅有可见 Prompt Token 的 Page；Page 分数仅累计 `a(j)<T0` 的原始概率 | 对原始长上下文的持续关注 |

`prompt_only` **不重新执行 Prompt 条件 Softmax**；对某一 KV Head **已经聚合完成**的所有 Prompt Page 分数同时除以该 KV Head 的 Scope Mass 不会改变排序，但不得在 GQA 聚合前分别按四个 Query Head 的 Prompt Mass 归一化。B1 保存的是**原始 Attention 概率聚合后的 Page Mass**。当 Page 跨越 Prompt/生成边界，`prompt_only` 只纳入 Prompt Token 的贡献。

**Page Top-K：** 对每个 Scope 按 Page Mass 降序、Page ID 升序并列排序；`K_page=32`，保存：

- `page_topk_ids_all_visible`；
- `page_topk_ids_prompt_only`。

两者均为 `[C,L,Hkv,32]`、`uint32`，不足 32 个候选使用 `UINT32_MAX` 填充。**不默认保存 Page 分数**；诊断模式可追加 `page_topk_scores_*`（真实 Page Mass，非估计评分）。

**Page Scope Mass 与 Top-32 Coverage：** 对 Scope `u`，令 `\mathcal P_u` 为该 Scope 的 Page 集合、`S_{32,u}` 为 Page Top-32：

$$
M_u=\sum_{p\in\mathcal P_u} B_p,\qquad
\operatorname{Coverage}_{32,u}=\frac{\sum_{p\in S_{32,u}}B_p}{M_u}\quad (M_u>0).
$$

`all_visible` 的 `M_u≈1`；`prompt_only` 的 `M_u` 等于 KV Group 四个 Query Head 的 `prompt_mass` 等权平均。两者均为 KV Head 级数据；分别保存 `page_scope_mass`、`page_top32_coverage` 和 `page_scope_valid`，形状 `[C,L,Hkv,2]`，Scope 顺序固定 `all_visible`、`prompt_only`。

若 Scope 没有候选 Page 或其**统计路径计算出的** `M_u=0`：`page_scope_valid=false`、Coverage=`NaN`、全部 Top-K ID 填 Sentinel；不得把零质量 Page 的并列排序误认为稳定集合。候选不足 32 但 Mass 非零时 Coverage 仍有效，Top-32 RCO 是否有效另行判断。**Coverage 必须与 Scope Mass 联合解释**，不能把 Prompt 的条件 Coverage 视为占全部 Attention 的质量。

### 5.6 离线 Page 集合稳定性与 RCO

对相同 `(sample,layer,kv_head,scope)`，设 `S(s;k)` 为 Decode Step `s` 的前 k 个 Page 集合，`N_s` 与 `N_t` 为各自的候选 Page 数，`t=s+Δ`：

$$
O_k(s,t)=\frac{|S(s;k)\cap S(t;k)|}{k},\qquad B_k(t)=\frac{k}{N_t},
$$

$$
RCO_{\mathrm{raw}}(s,t;k)=\frac{O_k(s,t)-B_k(t)}{1-B_k(t)},\qquad
RCO_{\mathrm{exact\_page}}(s,t;k)=\max(0,RCO_{\mathrm{raw}}(s,t;k)).
$$

随机基线 `k/N_t` 是固定较早 Top-k、较晚在 `N_t` 个 Page 中**均匀随机**取 k 个的描述性基线，不表示 Attention 真实 Page 分布服从均匀模型，也不能自动视为显著性检验。

**有效条件：** 两步均具有完整 k 个有效 Page，`N_s≥k` 且 `N_t>k`，`page_scope_valid` 均为 true，步之间 `(sample,layer,kv_head,scope,page_size,ranking)` 一致；缺失 Step/EOS 不补零。否则该比较无效。默认 `k=32`，离线敏感性预算为 `k∈{8,16,32}`，均可由保存的有序 Top-32 恢复；时间间隔 `Δ∈{1,2,4,8,16,32,64,128,256}`。

分别输出 `RCO_exact_page_all_visible`、`RCO_exact_page_prompt_only`，**不能混用两套 Page ID 或候选 Page 数**。离线记录 `overlap_raw`、`random_baseline`、`rco_raw`、`rco_clipped`、`valid_pair_count`，并与 `page_top32_coverage`、`page_scope_mass` 一同分析。Page Coverage 很低或 Prompt Scope Mass 极小时，高 RCO 仅说明**集合排序稳定**，并不等价于保留这些 Page 具有显著 Cache 价值。

### 5.7 在线与离线的严格分工

| 在线按 Query/Step 提取并保存 | 离线重算或统计 |
|---|---|
| Entropy / Norm Entropy、Top-Mass、位置质量与距离分桶 | Head 分布、热力图、跨任务与跨长度比较 |
| Decode Token Top-64 ID（可关闭） | Token 集合重合、位置迁移 |
| Decode 双 Scope Page Top-32 ID | 多 k、多 Δ 的原始重合及 RCO |
| Decode KV Head Page Scope Mass、Coverage、有效标记 | 联合分析稳定性与注意力承载质量 |
| 完整生成 Token 轨迹和执行索引 | 与 B2 / V2 结果关联并用于回放 |

B1 不执行 `CV_prefill`、Prefill 关键 Token 频次/集合或 Prefill-to-Decode 覆盖率；它们属于 B2 新的在线累积与离线联合分析。

## 6. 只读 AttentionProbe 接口契约

### 6.1 定位与职责

B1 的 `AttentionProbe` 在**模型实际 Attention Function 调用边界**提供只读观测，不直接计算指标、不写文件。`AttentionStatsCollector` 实施第四、五节在线算法；`AttentionTraceWriter` 负责 CPU 缓冲与结构化落盘；`ExecutionMetadata` 负责 Token/Step/Head 索引；离线 `AttentionAnalyzer` 计算 RCO 等分析指标。

当前 KV-Base 的通用 `KVObserver` 是 Attention Module 的**后置 Hook**，`query` 可能为 `None`，`attention` 不保证存在，不能直接承担 B1 的统计入口。

### 6.2 实际调用边界与 Backend 接入

对于当前 Transformers 5.2.0 的 Llama，实现需要在 Q/K 投影、RoPE、KV Cache Update **之后**，Attention Output 产生**之前或同一函数作用域内**获取实际传入的 Q、K、Mask 和 Scaling。V 默认不需要写出；不自行重算投影、RoPE、Cache 更新。

必须验证**实际使用的 Backend** 能被 Probe 拦截；不同注册路径和 `eager` 回退函数不一定由相同 Registry 包装。无法接入时明确报错；不得静默漏采、自动改成 eager 或在声明精确结果时使用近似替代。

### 6.3 生命周期与元数据关联

Attention Function 不一定直接收到 `cache_position`、`sample_id`、`decode_step`。允许由 **Attention Module Forward Pre-Hook** 与 ExperimentRunner 提供轻量元数据，再在 Function 调用处与 Q/K 关联。单一生命周期管理器负责 `forward_id/phase/decode_step`，不得因前后 Hook 同时调用而重复计数。

| 字段 | 约定来源 |
|---|---|
| `sample_id`、`prompt_length` | ExperimentRunner / 固定数据集 |
| `layer_id` | Attention Module |
| `forward_id`、`phase`、`decode_step` | 统一样本级生命周期 |
| `cache_position`、`query_positions` | Module Forward 参数及合法映射 |
| `key_positions` | 已验证的完整连续 Cache 对应关系 `[0,T-1]` |
| `backend_name`、`scaling`、`dropout_p` | 实际 Attention Function 调用 |

### 6.4 `AttentionProbeContext` 逻辑字段

```text
AttentionProbeContext:
    sample_id: str
    phase: PREFILL | DECODE
    layer_id: int
    forward_id: int
    decode_step: int | null
    query: Tensor                 # [1,Hq,Q,D]，RoPE 后
    key: Tensor                   # [1,Hkv,T,D]，Cache 更新后
    attention_mask: Tensor | None # Backend 原始 Mask/偏置信息
    scaling: float
    is_causal: bool
    effective_visibility: resolved_contract
    finite_attention_bias: resolved_contract | None
    backend_name: str
    cache_position: Tensor       # [Q]
    query_positions: Tensor      # [Q]，逻辑位置
    key_positions: Tensor        # [T]，与 key 的 T 轴一一对应
    prompt_length: int
    query_dtype: dtype
    key_dtype: dtype
    dropout_p: float
```

`effective_visibility` 与 `finite_attention_bias` 是**逻辑契约**，可按 Query/Key Block 解析，**不要求预先物化 `[Q,T]` Mask**。原始 additive `attention_mask` 若已经包含有限偏置，Collector 不可对同一 Bias 重复相加。`query`/`key` 仅在回调期间有效，不允许长期持有原始 GPU Tensor。

当前 B1 仅支持 `batch_size=1`、完整连续 Cache、Prefill 完整 Forward 和 Decode `Q=1`；Prefill 可从 Q 中选择最后 `W_eff` 个 Query。不能验证 `T` 轴对应 Key 逻辑位置时直接失败。

### 6.5 只读、安全及 Patch 共存

1. Probe/Collector 不允许原地修改传入 Tensor、Mask、模型权重、Cache、函数返回值或生成设置。
2. 原 Attention Function 恰好调用一次，Probe 的计算与原函数语义独立；`eval()`、无梯度、dropout 为 0。
3. 不支持同时启用 KV Press、Head Masking、压缩或其他 Cache 干预；不兼容状态在执行前检查并拒绝。
4. KV-Base 当前 `attention_patch.py` 已在导入时包装 Registry；B1 不得再次运行全局 Patch 或永久破坏既有包装。进入 Probe 作用域时保存需替换的函数对象，退出通过 `try/finally` 恢复**进入前状态**。
5. 第一版拒绝嵌套 Probe、并发 Profiling 和跨模型共享易变全局状态；若未来支持，必须单独版本化、测试。
6. 无法确定真实因果约束、有限 Bias、Scaling、KV Key 位置或 Backend 接入时，必须失败并记录原因。

### 6.6 事件完整性及错误处理

每次 Attention 调用关联 `(sample_id,forward_id,layer_id,phase,query_position[,decode_step])`。对每个成功 Prefill/Decode Forward，预期 Layer 事件**恰好**齐全、无重复，全部 Layer Query 位置和阶段一致；Writer 必须对完成的 Step 验证完整性。

重复或缺失事件、Collector 失败、写盘失败、生成序列/Trace 长度不一致时，不得补零并标记成功。可保留诊断分片，但样本状态为 `partial` / `failed`。作用域错误退出时也必须恢复原 Attention 调用路径。

## 7. 落盘 Schema 与数据完整性

### 7.1 文件结构与原则

JSON 负责溯源、配置、生成轨迹和校验信息；NPZ 负责高频统计，按 **Sample** 隔离、Decode 每**最多 128 Step** 分片。B1 不保存完整 Attention Matrix、Q/K/V、全量 KV Cache。B2 只追加独立数据，不改写 B1 Schema。

```text
runs/<run_id>/
├── run_metadata.json
├── config_snapshot.yaml
├── schema.json
├── file_manifest.json
└── samples/
    └── <sample_id>/
        ├── sample_metadata.json
        ├── generated_ids.json
        ├── prefill_stats.npz
        ├── decode_stats_000.npz
        ├── decode_stats_001.npz
        └── debug/                      # 仅显式开启
```

### 7.2 运行级 Schema 与溯源

`schema.json` 冻结：`schema_version=headkv-v1-b1-attention-v1`、`feature_names`（有序）、`feature_dtypes`、`dimensions`、`axis_conventions`、`prefill_window`、`page_size`、`token_topk`、`page_topk`、`page_scopes`、`distance_bins`、`invalid_value_rules`、`probe_contract_version`。

双 Scope 顺序**固定**：`["all_visible","prompt_only"]`，数组 Scope 轴 `0/1` 与之对应。离线代码读取并验证 Schema，不通过隐含列顺序猜测指标。

`run_metadata.json` 至少记录：`run_id`、固定集 Manifest 身份及 Hash（原有 `data_sha256` 等按实际 manifest 字段保留）、model/tokenizer revision、全部 Generation Config、实际 Attention Backend、设备/dtype、版本/Commit、采集配置及 Hash、时间、成功/失败/部分完成状态、资源开销、落盘总量。运行状态 `running/completed/partial/failed`，中断不能标记 `completed`。

### 7.3 样本元数据与生成 Token

`sample_metadata.json` 至少有 `sample_id,dataset,task,pair_id,input_sha256,prompt_length,max_new_tokens,actual_generated_length,decode_trace_length,eos_reached,generation_stop_reason,attention_backend,prefill_query_positions,status`。

`generated_ids.json` 保存**完整生成 Token IDs** 和 SHA-256，哈希序列化规则必须在 Schema/Manifest 中明确且稳定。原始 Prompt `input_ids` 以冻结的数据集为权威来源，不在每个样本文件重复存储。标准成功流程强制校验 `decode_trace_length=max(0,actual_generated_length-1)`，中断则报告实际 trace 与样本状态。

### 7.4 `prefill_stats.npz`

| 数组 | dtype | 形状 | 备注 |
|---|---|---|---|
| `query_positions` | `uint32` | `[W_eff]` | 最后 `W_eff=min(64,T0)` 个 Prefill Query 位置 |
| `visible_token_count` | `uint32` | `[W_eff]` | 每 Query 可见 Token 数 |
| `scalar_features` | `float32` | `[W_eff,L,Hq,28]` | 仅 Query Head 级基础指标 |

不保存 B2 的 CV、关键 Token 和频次。标准因果连续 Cache 下各 Layer/Head 的可见 Token 数一致，否则该 Schema 不允许静默合并。

### 7.5 `decode_stats_<chunk>.npz`

设 `C` 是分片实际 Step 数 `1≤C≤128`，`S=2` 是 Page Scope 数，`F=28`：

| 数组 | dtype | 形状 | 必须性 |
|---|---|---|---|
| `decode_steps` | `uint32` | `[C]` | 必须 |
| `query_positions` | `uint32` | `[C]` | 必须 |
| `visible_token_count` | `uint32` | `[C]` | 必须 |
| `scalar_features` | `float32` | `[C,L,Hq,28]` | 必须 |
| `token_topk_positions` | `uint32` | `[C,L,Hq,64]` | `save_token_topk=true` 时 |
| `token_topk_weights` | `float16`（或配置指定） | `[C,L,Hq,64]` | 诊断模式 |
| `page_topk_ids_all_visible` | `uint32` | `[C,L,Hkv,32]` | **必须** |
| `page_topk_ids_prompt_only` | `uint32` | `[C,L,Hkv,32]` | **必须** |
| `page_candidate_count_all_visible` | `uint32` | `[C]` | 必须 |
| `page_candidate_count_prompt_only` | `uint32` | `[C]` | 必须 |
| `page_scope_mass` | `float32` | `[C,L,Hkv,2]` | **必须** |
| `page_top32_coverage` | `float32` | `[C,L,Hkv,2]` | **必须** |
| `page_scope_valid` | `bool` | `[C,L,Hkv,2]` | **必须** |
| `page_topk_scores_all_visible` | `float16`（或配置指定） | `[C,L,Hkv,32]` | 诊断模式 |
| `page_topk_scores_prompt_only` | `float16`（或配置指定） | `[C,L,Hkv,32]` | 诊断模式 |

Scope 质量为 0 或无候选 Page 时，Page 排名全 Sentinel、Coverage 为 `NaN`、`page_scope_valid=false`。候选 Page 不足 32 而 Scope 有效时，Coverage 可以有效但 Top-32 RCO 无效。**未启用的可选数组不创建**，不得占位为全零。

第 2.2 节的共享候选集不变量是 `page_candidate_count_*` 采用 `[C]` 而非附带
Layer/Head 轴的前提。标准运行中 `query_positions[i]=T0+decode_steps[i]`、
`visible_token_count[i]=query_positions[i]+1`；`page_candidate_count_all_visible=ceil(visible_token_count/16)`、
`page_candidate_count_prompt_only=ceil(T0/16)`，须由实际可见范围验证成立。任何
Head/Layer 专属可见性导致这些候选 Page 集不一致时，B1 v1 必须明确失败。

### 7.6 固定的 Query Head Feature Schema：`F=28`

以下是 `scalar_features` 的**唯一合法轴顺序**；其中距离分桶是 17 项，故 `2+4+4+1+17=28`：

```text
 0  entropy
 1  entropy_normalized
 2  top_mass_16
 3  top_mass_64
 4  top_mass_256
 5  top_mass_1024
 6  sink_mass_4
 7  recent_mass_64
 8  prompt_mass
 9  generated_mass
10  mean_distance
11  distance_mass_0
12  distance_mass_1
13  distance_mass_2_3
14  distance_mass_4_7
15  distance_mass_8_15
16  distance_mass_16_31
17  distance_mass_32_63
18  distance_mass_64_127
19  distance_mass_128_255
20  distance_mass_256_511
21  distance_mass_512_1023
22  distance_mass_1024_2047
23  distance_mass_2048_4095
24  distance_mass_4096_8191
25  distance_mass_8192_16383
26  distance_mass_16384_32767
27  distance_mass_ge_32768
```

**Page Scope Mass、Coverage、有效标记是 KV Head 级数组，不得并入上述 28 个 Query Head Feature。** Feature 名称和顺序改变即属 Schema 不兼容变更。

### 7.7 无效值与数据完整性

| 类型/事件 | 约定 |
|---|---|
| 无效 Token/Page ID | `UINT32_MAX`，离线集合运算必须丢弃 |
| 有效普通标量 | 必须为有限 `float32` 数值 |
| 无效 Page Coverage | `NaN`，并且 `page_scope_valid=false` |
| Scope 非零但候选不足 Top-K | Top-K 有效前缀 + Sentinel 尾部，Coverage 有效，预算不足时 RCO 无效 |
| 未启用的可选数组 | 不生成，不用全零代替 |
| 缺失 Layer/Step/失败写盘 | 样本标记 `partial/failed`，不能把零填充当有效值 |

每个成功样本的 Decode Step 从 0 连续、跨分片无重叠、跨 Layer 对齐；`generated_ids`、Trace 数和阶段索引一致。NPZ 文件先写临时文件再原子替换，Writer 在 CPU 端缓冲紧凑结果，不持有跨 Step 的 GPU Q/K。

`file_manifest.json` 为每个数据文件记录相对路径、SHA-256、逻辑数组字节数、实际文件字节数、数组名/shape/dtype、Sample 和 Step 区间；**Manifest 不自哈希**。运行中断必须列出有效完成的文件和未完成的样本。Debug 的完整 Attention/Q/K 只允许对明确选定的 Sample/Layer/Head/Step 记录，并单独说明采集范围。

## 8. B1 默认配置与资源测量

### 8.1 冻结配置

以下为默认配置示意；对应名称与第五、七节字段严格一致：

```yaml
schema_version: headkv-v1-b1-attention-v1
fixed_dataset:
  path: /absolute/path/to/frozen/prompts.jsonl  # 由运行环境填写，必须校验 manifest
collection:
  prefill_window: 64
  decode_interval: 1
  observe_all_layers: true
  observe_all_query_heads: true
  key_block_size: 1024
  statistics_dtype: float32
attention:
  top_mass_k: [16, 64, 256, 1024]
  sink_tokens: 4
  recent_tokens: 64
  distance_bins: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
token_trace:
  token_topk: 64
  save_token_topk: true
  save_token_topk_weights: false
page:
  page_size: 16
  page_topk: 32
  page_scopes: [all_visible, prompt_only]
  save_page_coverage: true
  save_page_scope_mass: true
  save_page_topk_scores: false
storage:
  format: npz
  decode_chunk_steps: 128
  save_full_attention: false
  save_full_kv_cache: false
  save_generated_ids: true
debug:
  debug_attention: false
```

`distance_bins` 是 17 桶的**左边界序列**，最后 `32768` 表示开放上界 `≥32768`；对应第五、七节的 17 个 distance Feature。双 Page Scope、Page Coverage、Page Scope Mass 和逐 Decode Step 采集是 B1 主实验强制项；不可静默关闭。`key_block_size` 可以调整而不改变指标定义，关闭可选 Token Top-K 需要更新运行配置 Hash。

模型、dtype、Backend、Greedy 生成选项、固定数据集版本与环境版本在独立的 Generation/Run 配置中记录，不由本配置片段隐式指定。

### 8.2 未压缩逻辑存储上界

100 条 Prompt 的理论生成上界为：RULER 40×128、GovReport+PG19 60×1024，共 `66,560` 生成 Token。根据第三节的 Decode Step 对应关系，全部生成至上限时：

$$
S_{\max}=40(128-1)+60(1024-1)=66{,}460.
$$

按 `L=32, Hq=32, Hkv=8, F=28, K_token=64, K_page=32, scopes=2` 计算：

| 数据 | 未压缩主要数组逻辑上界（GiB） |
|---|---:|
| Decode Query Head 28 项标量 | 7.10 |
| Decode Query Head Top-64 Token ID（可选） | 16.23 |
| Decode 两套 Page Top-32 ID（必须） | 4.06 |
| Decode Page Scope Mass、Top-32 Coverage、有效标记（必须） | 0.29 |
| Prefill 64 Query × 全部 Layer/Query Head × 28 项标量 | 0.68 |
| **合计（默认开启 Token Top-K）** | **约 28.35** |
| **合计（关闭可选 Token Top-K）** | **约 12.12** |

还不含少量索引、元数据、诊断数据和分片头部；也未预测自然 EOS、NPZ 压缩率。**这是逻辑存储上界，不是实际磁盘占用或在线显存需求。**

### 8.3 资源验证与控制

先对**两条 8K 真实样本**（建议一条短生成、一条预期长生成）分别做无 Probe Baseline 和相同 Backend 的 B1 采集。记录 Prefill 耗时、Decode 总/逐步耗时（含分位数）、GPU 峰值显存、生成长度、逻辑数组字节数、实际文件字节数、写入耗时、实际 Backend 和 Generated IDs 一致性。异步 GPU 计时须用可比的同步方法，记录硬件与外部负载。

通过后再对少量 16K/32K 样本测量 Prefill 末端 64 Query 和长 Decode 的实际开销；只有真实资源预算允许时，才运行全量 100 条。

超预算时依次排查 Tensor 引用/重复计算 → 调整 Key Block → 优化 Writer → **显式关闭可选 Token Top-K** 或诊断权重 → 缩小试运行规模/优化实现。不能静默改变 Page 排名、Page 双 Scope、核心统计或 Decode 观测频率。

资源记录采用 `headkv-resource-measurement-v1` 契约。每条 Baseline/B1 样本必须记录：

- `generation_end_to_end_seconds`：包围 `model.generate` 和样本末尾 CUDA 同步的 Host wall
  time，不含 B1 Trace 写盘；旧字段 `elapsed_seconds` 保留为同一数值的兼容别名。
- `forward_timing`：标准 Greedy 路径的第一个根模型 `forward` 为 Prefill，后续根模型
  `forward` 为 Decode；CUDA 使用同一设备流上的成对 Event，在样本边界同步后得到 Prefill
  时间及 Decode 的总时间、均值、最小/最大、p50/p95（线性插值）。它不包含 Python 生成控制、
  token 选择或写盘。CPU 仅以 `perf_counter` 作为回归测试后备，不作为 GPU 性能数据。
- B1 额外记录 `trace_write_wall_seconds`，以及 `trace_storage` 中的样本逻辑数组字节数和
  实际样本文件字节数。该字节数不含会在每次写样本后更新的运行级 Manifest；逻辑数组字节数是
  未压缩 NumPy 数组字节数。

资源记录必须同时保存其计时方法、Scope、Forward 数量及不完整 Forward 数。标准成功样本须
满足 `Prefill forward=1`、`Decode forward=max(0, generated_length-1)`；不满足时不得把该
资源记录当作可比较的 B1 标准生成测量。

## 9. B1 正确性验证与验收

### 9.1 CPU 数学与数值测试

使用随机初始化的小型 Llama GQA 模型和独立 eager 参考 Attention，覆盖下列条件：

1. Causal 可见范围、Query 自身可见、隐式 Causal/显式 Mask/有限 Bias 的语义正确；按 Key Block 全局归一化。
2. FP32 分块 Log-Sum-Exp、Entropy、归一化 Entropy、Top-K Mass、Position Mass、距离分桶/均值与完整参考 Attention 一致；测试多种 Key Block Size。FP32 初始参考容差 `atol=1e-5, rtol=1e-5`，其他 Backend 单独验证。
3. 数学不变量：概率和≈1，Entropy 范围正确，Top-K Mass 单调，距离分桶和≈1，Prompt+Generated Mass≈1，KV Head `all_visible` Page Mass≈1。
4. GQA 映射、四个 Query Head 先归一化再 Page 聚合、末页部分 Token、跨 Prompt/Generated 的 Page 边界、双 Scope 排序与 Scope Mass/Coverage 均正确。
5. Token/Page Top-K 降序及并列规则、候选不足的 Sentinel、`page_scope_valid=false`、Scope Mass=0 与 Coverage=`NaN`、`N_s<k/N_t≤k` 的无效 RCO 检测正确。
6. 离线 `overlap_raw`、`random_baseline`、`rco_raw`、`rco_clipped` 以及 k=8/16/32 与多 Δ 的计算可由已落盘 Page ID 复算。

### 9.2 Probe、生命周期与只读性

必须证明：实际 Backend Probe 被调用且每个 Layer/Forward 恰好一次；Pre-Hook 元数据与 Function Q/K 对应；Q/K 已 RoPE/Cache Update；Query/Key Position、`decode_step` 与生成 Token 关系正确；跨 Layer 同一 Forward 对齐；完整 Mask 和 Scaling 被正确重建；有/无 Probe 的 logits 在适当数值容差内一致，Greedy `generated_ids` 必须一致；原 Attention Function 只执行一次；异常和正常退出均恢复进入前接口；与 KV-Base 既有 Patch 不冲突；没有非 Debug 的完整 Q/K/Attention 持久化。

### 9.3 Schema 与完整性

验证 `F=28` Feature 顺序、数组 dtype/轴顺序、双 Page Scope、Page Coverage 和 Scope Mass、可选数组的存在条件、Sentinel/NaN、分片大小与连续 Step、各文件 SHA-256、样本与运行状态，以及完整标准运行的 `decode_trace_length=max(0,generated_length-1)`。制造中断、缺层和写入失败，确认绝不将不完整数据标为成功。

### 9.4 GPU 分阶段验收

**8K 试运行：** 两条真实输入；Baseline 与 B1 的 Greedy Generated IDs 完全一致；实际 Backend/dtype/revision/input Hash 完整；取得至少一个有效 `Δ=1` 的双 Scope Page RCO（若某样本长度不足，应另选够长的样本而不是填零）；资源报告完整；不支持的 Backend 明确失败。

**16K/32K 可行性测试：** 检查长上下文 Prefill 采集临时显存、Decode 分块耗时、Writer 写入与分片完整性。前一阶段通过不意味着后一阶段或全样本资源可行。

**B1 开发完成判据：** 数学、Probe/只读、数据完整性、至少 8K 真实模型采集和离线 RCO 计算均通过；真实资源开销已测量且正式运行方案获批准。**B1 通过不代表发现稳定 Head 分类，也不证明硬件加速效果。**

## 10. B2 扩展边界与兼容性

B2 的研究目标是在相同 `(sample_id, input_sha256, generated_ids_sha256, layer/head/position)` 坐标下，研究 Prefill 跨 Query 一致性是否能预测 Decode 注意力，进一步检验与 B1 Page 时间稳定性的关系。

B2 可新增：`CV_prefill`、跨 Query 重要 Token 频次、Prefill 关键 Token 集、Decode 对关键 Token 的全部 Attention Mass / Prompt 条件覆盖率。其二值化阈值、观察窗口、排除 initial/recent 范围、Query/KV Head 聚合、关键 Token 预算、参数敏感性及参考/评估样本划分，都需在**独立 B2 指标规范**中冻结；B1 不预先实现或锁定这些数值。

B2 可以使用 B1 保存的完整 `generated_ids` 重放相同 Token 历史；必须核对模型权重、Tokenizer、执行参数、GQA/Token 映射与数值一致性，不可只因 Prompt 相同就假定 Decode Attention 相同。

B2 的输出使用独立文件/子目录、独立 Schema Version 与运行状态，并显式引用 B1 `run_id` 和生成轨迹哈希。**不得回写 B1 的 NPZ、重定义其 28 项标量、双 Scope Page ID、数组类型和索引语义。** 两阶段联合分析按 Query Head 与 KV Head 统一映射后进行；不得为提高关联性而按结果反向调参或选择样本。KV 表示、压缩和硬件实现仍属于后续 V2 阶段。

---

## 附录 A：十节整合一致性校验记录

| 校验维度 | 统一后的唯一约定 | 结果 |
|---|---|---|
| 研究边界 | B1 = Attention 基础分布/时间演化；B2 = CV/Prefill-to-Decode | 一致 |
| 数据来源 | 100 条固定 `input_ids`；路径通过配置给出并校验 Manifest | 一致；不强制原草案示例路径 |
| 生成设置 | Greedy、Batch=1、完整连续未压缩 KV Cache、标准逐 Token Decode | 一致 |
| 时间索引 | Prefill 最后 W=64；Decode step 0 消费 `generated_ids[0]`；完整 Trace=`G_out−1` | 一致 |
| GQA | 32 Q Head、8 KV Head、`kv_head=floor(q_head/4)` | 一致 |
| Attention 公式 | 实际 Q/K/Scaling + 有效可见集合 + 单次有限 Bias；全局 Softmax | 一致 |
| 分块计算 | 两遍 Key Block、FP32 统计、全局 Log-Sum-Exp、全局 Top-K | 一致 |
| Query Head 标量 | 2 熵 + 4 Top-Mass + 4 Position-Mass + 1 Distance + 17 Histogram = **28** | 一致 |
| Token Top-K | Decode Top-64 `uint32` 位置可配置，权重仅诊断 | 一致 |
| Page Top-K | 两个必选 Scope，均为 KV Head Top-32，P=16 | 一致 |
| Page Coverage | `page_scope_mass/page_top32_coverage/page_scope_valid` 均为 `[C,L,Hkv,2]` | 一致 |
| RCO | 离线计算 k=8/16/32、Δ=1…256，双 Scope 独立随机基线及有效性 | 一致 |
| 数据版本 | `headkv-v1-b1-attention-v1`；JSON + NPZ；128 Step 分片；SHA-256 | 一致 |
| 资源预算 | 最多 66,460 Decode Trace，主要逻辑数据约 **28.35 GiB** | 已按当前 Schema 重算 |
| Probe 安全性 | 实际 Function 边界、可恢复、不得与现有 Patch 冲突；事件不漏采 | 一致 |
| B2 衔接 | 使用生成轨迹回放；独立文件/Schema；不改 B1 | 一致 |

**审查后明确的技术限制：** 规范约束了接口和正确性，但不宣称所有 Attention Backend 已获验证，也不宣称真实 GPU 资源测量已通过。在 CPU 小模型/目标 GPU Backend 测试前，不应把规范通过等同于采集系统已经实现。

## 附录 B：相对原始 v0.1 草案的主要修改

1. 第 1 节明确逻辑 Page 非物理分页、离线不恢复完整 Attention、现象验证不是论文系统复现。
2. 第 3 节固定 Decode Forward 与 `generated_ids` 的偏移关系、跨 Layer Step 对齐及标准 Causal 可见范围。
3. 第 4 节补充隐式 Causal/有限 Bias 语义、两遍分块统计和 FP32 正确性约束。
4. 第 5 节保留 Query Head 基础指标，距离桶扩展到 `≥32768`，双 Scope Page Top-K 均为必选，新增 KV Head Page Scope Mass/Coverage/Valid 和多 k RCO 输出。
5. 第 6 节明确 Pre-Hook + Attention Function 的职责拆分、Backend 实际接入验证和现有全局 Patch 的恢复规则。
6. 第 7 节固定 F=28、双 Scope/不同 Head 粒度的数组、缺失值和完整性契约。
7. 第 8/9 节更新资源计算及 CPU/GPU 分阶段验收；第 10 节保持 B2 独立规范和轨迹回放边界。
