# HeadKV V1-B2 Prefill 一致性与跨阶段指标及数据接口规范

> **文档版本：** V1-B2 Implementation Baseline v0.2
> **拟定 Schema 版本：** `headkv-v1-b2-prefill-consistency-v1`
> **研究阶段：** V1-B2（Prefill 跨 Query 一致性与 Prefill-to-Decode 关系）
> **模型与框架：** Llama-3.1-8B-Instruct；KV-Base；当前 HeadKV SDPA Attention Probe
> **状态：** 指标、接口和正式 100 条目标集合已审定；GPU 资源数字须经 6 条 Pilot 后冻结
> **上游依据：** [V1 研究计划](research_plan_v1.md)、
> [V1-B1 Attention 指标规范](v1_b1_attention_metric_interface_spec.md) 与
> [V1-B1 全量分析结果](v1_b1_full_analysis_results_v1.md)

**核心原则：** B2 研究 CateKV 所揭示的“Prefill 跨 Query 一致性可能预测 Decode
注意力”现象及其内在规律，不以复现 CateKV 的静态 Head Mask、KV 压缩系统或效率收益为
目标。主指标尽可能遵循论文公式，但必须采用明确、可检验的因果可见性、GQA 聚合、并列和
存储语义；作者公开代码中的不同口径仅作为预注册的敏感性参考，不自动成为主定义。

---

## 1. 研究问题、范围与非目标

### 1.1 核心研究问题

B2 在与 B1 相同的固定输入、模型和生成轨迹上回答以下问题：

1. 同一 KV Head 在 Prefill 末端多个 Query 上关注的重要历史 Token 是否集中且重复出现？
2. Prefill 中识别出的关键 Token 集合，在后续 Decode 中能够持续承载多少真实 Attention Mass？
3. `CV_prefill` 越高的 Head，是否具有更高的 Prefill-to-Decode 预测能力？
4. Prefill 一致性、Decode 关键集合覆盖率与 B1 的 Token/Page 时间稳定性是否相关？
5. 上述关系是否随 Layer、Head、Decode 时间、任务、输入长度和样本变化？

B2 以连续的 `CV_prefill`、Attention Mass 覆盖率及其时间曲线为主要证据。任何
“consistent/adaptive”二值标签只能作为预先定义阈值下的描述性结果，不能替代连续指标，
也不能从 B1 或 B2 的单一相关性直接推导为固定 Head 类别。

### 1.2 本阶段明确不做

B2 不包含：

- 复现 CateKV 的模型级静态 Head Mask 或 128K Variable Tracking 参考集流程；
- 实现 KV 淘汰、压缩、检索、物理 Page 管理或自定义 Attention Kernel；
- 复现论文的准确率、吞吐、延迟或显存收益；
- 保存完整 `T0×T0` Prefill Attention、全部 Decode Attention、Q/K/V 或完整 KV Cache；
- 根据正式结果反向选择阈值、预算、样本或报告口径；
- 把相关性解释为因果收益或可部署策略。

若未来研究模型级静态分类，应另建独立参考集、调参集和评估集，并形成新的运行规范；不得
使用当前 100 条评估样本同时调参和宣称泛化。

### 1.3 与 B1 的关系

B2 是 B1 的受控后续采集，不是对 B1 文件的升级：

- 沿用 B1 的 `sample_id`、`input_sha256`、逻辑位置、Layer/Head 映射和生成轨迹；
- 使用 B1 已保存的 `generated_ids` 进行强制回放；
- 输出独立目录、独立 Manifest 和独立 Schema；
- 不重写 B1 的 28 项标量、Token/Page Top-K、Page RCO 或任何原始工件；
- 联合分析通过哈希和坐标关联 B1/B2，不复制或改变 B1 指标定义。

## 2. 输入、执行方式与可追溯性

### 2.1 固定输入与运行集合

B2 主运行集合默认沿用 B1 已冻结并完成验证的 100 条输入：

| 数据集 | Prompt 长度 | 数量 | 任务 | 最大生成长度 |
|---|---:|---:|---|---:|
| GovReport | 8K / 16K | 20 / 20 | 长文摘要 | 1024 |
| PG19 | 8K / 32K | 10 / 10 | 文本续写 | 1024 |
| RULER | 8K / 32K | 20 / 20 | Needle 检索 | 128 |
| **合计** | | **100** | | |

输入直接读取冻结的 `input_ids`，不得重新 Tokenize、添加 Chat Template 或修改 Prompt。
每条样本必须核对固定数据 Manifest、`input_sha256`、实际长度和 `max_new_tokens`。

### 2.2 B1 生成轨迹强制回放

B2 不重新决定生成 Token，而是按 B1 的已验证轨迹执行 teacher-forced replay。设 Prompt
长度为 `T0`，B1 实际生成序列为：

$$
Y=(y_0,y_1,\ldots,y_{G_{out}-1}).
$$

B2 v1 要求 `G_out≥1`；零生成 Token 的样本没有可验证的 Prefill 预测目标，也没有 Decode
轨迹，应标记为不支持而不是伪造空的完成样本。

执行顺序为：

1. 对完整 Prompt 执行一次 Prefill，Prefill Logits 应预测 `y0`；
2. 对 `s=0...G_out-2`，第 `s` 个 Decode Forward 强制输入 `y_s`；
3. 该 Forward 的 Logits 应预测 `y_{s+1}`；
4. Decode Trace 长度必须为 `max(0,G_out-1)`。

每一步记录 `teacher_forced_top1_match`。在模型、Tokenizer、dtype、Backend 和执行配置完全
一致的前提下，任一步 Top-1 不匹配均说明轨迹或数值环境漂移；样本不得标记为正式完成。
强制输入既不允许继续生成新的分支，也不允许在 EOS 后扩展 B1 未记录的 Token。
Prefill 的预期/实际 Top-1 和匹配标记写入 `sample_metadata.json`。

### 2.3 上游 B1 绑定

每次 B2 运行必须记录：

- B1 完成包的路径或稳定标识、B1 `run_id` 与 Manifest SHA-256；
- 每条样本的 B1 `generated_ids_sha256` 和实际生成长度；
- 固定数据 Manifest SHA-256、模型与 Tokenizer 路径/Revision；
- HeadKV Git Commit、工作区是否 Dirty；
- PyTorch、Transformers、CUDA、GPU、dtype、实际 Attention Backend；
- 完整 B2 参数条件表及其规范化配置 Hash。

不得只凭相同 `sample_id` 或 Prompt 文本认定 B1/B2 可对齐。

B2 冻结两个不同职责的 B1 权威来源：

| 用途 | 权威工件 | SHA-256 |
|---|---|---|
| Prompt/生成轨迹回放 | `data/runs/b1_full_100_20261029_v1.manifest.json` | `039afecab0135169ccac0e96050a414b3e313d37d6108d7d648cae80bbcbd85d` |
| B1/B2 联合离线分析 | `data/analysis/b1_full_100_completed_20261103_v2/analysis_manifest.json` | `e1b0b4fc89b50b7979458b6416dd9b203e046320bdab3c503f7666854f59fd0d` |

固定数据 SHA-256 为
`30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee`。运行前必须在实际
执行环境重新计算并核对上述文件 Hash；文档记录不能代替文件存在性和内容验证。

## 3. 坐标、符号与默认参数

### 3.1 坐标体系

| 符号 | 含义 | 当前模型默认值 |
|---|---|---:|
| `L` | Transformer 层数 | 32 |
| `Hq` | Query Head 数 | 32 |
| `Hkv` | KV Head 数 | 8 |
| `G` | 每个 KV Head 对应的 Query Head 数，`Hq/Hkv` | 4 |
| `D` | Head Dimension | 128 |
| `T0` | 实际 Prompt Token 数 | 样本决定 |
| `W` | Prefill 观察 Query 数 | 64 |
| `I` | 排除的初始 Token 数 | 64 |
| `R` | 排除的 Prompt 末端 Token 数 | 64（主条件） |
| `Kmax` | 保存的最大关键 Token 预算 | 2048 |
| `B` | 关键 Token 预算集合 | `{128,512,2048}` |

Layer、Head、Query、Token 位置与 Decode Step 均从 0 开始。Query Head `h` 对应：

$$
v(h)=\left\lfloor\frac{h}{G}\right\rfloor.
$$

B2 主指标以 **KV Head** 为统计单位，因为 Llama GQA 中同组 Query Head 共享同一 K/V，且
论文要求在组内聚合观察矩阵。Query Head 概率只作为构成 KV Head 概率的中间量，不直接用
单个 Query Head 的 CV 冒充 KV Head CV。

### 3.2 主参数与预注册敏感性条件

Llama-3.1 主条件尽量采用论文公式和论文给出的阈值参数，同时选择全部观察 Query 共同可见
的历史区间作为候选集：

| `condition_id` | 用途 | `W` | `I` | `R` | Quantile `k` | 缩放 `α` | 频次单位 |
|---|---|---|---:|---:|---:|---:|---|
| `paper_token_primary` | 主结果 | 64 | 64 | 64 | 0.984 | 0.8 | Token |
| `paper_token_alpha1` | 阈值敏感性 | 64 | 64 | 64 | 0.984 | 1.0 | Token |
| `paper_token_k099` | 分位数敏感性 | 64 | 64 | 64 | 0.990 | 0.8 | Token |
| `paper_token_recent256` | 末端排除敏感性 | 64 | 64 | 256 | 0.984 | 0.8 | Token |

主结论必须以 `paper_token_primary` 为准，其余条件只检验方向和 Head 排名是否稳健。不得在
看到正式结果后增加、删除或替换条件；确需增加时必须形成新配置版本和新运行，不得并入主
条件冒充预注册分析。

作者公开代码中的 Chunk CV、低/高分位插值阈值和其他实现差异不属于 B2 v1 必选条件。
若以后比较，必须使用单独的 `metric_family` 与 Schema/配置标识，不能继续命名为
`paper_token_*`。

B2 v1 的全部 Condition 必须使用相同 `W=64`，以保持 Histogram 和观察 Query 轴一致；若
未来研究其他窗口长度，应建立新的 Schema/运行，而不是把不同 `W` 混入同一 Condition 轴。

## 4. Prefill 观察矩阵

### 4.1 观察 Query 与候选 Key

观察 Query 逻辑位置为：

$$
Q_{obs}=\{T_0-W,\ldots,T_0-1\}.
$$

当前数据全部满足 `T0>W`；若未来样本 `T0≤W+I`，必须拒绝而不是静默缩短正式定义。

对条件 `m`，候选历史 Token 集为：

$$
E_m=\{j\mid I_m\le j<T_0-R_m\}.
$$

所有正式条件要求 `R_m≥W_m`，因此 `E_m` 中每个 Token 对全部观察 Query 都因果可见，得到
严格矩形的 `W×|E_m|` 观察矩阵。这一约定避免将较早 Query 尚不可见的位置错误记为零频次，
也避免因不同 Query 的候选列数不同而使 CV 混入位置可见性偏差。

`R=64` 的主条件等价于排除整个观察窗口自身，接近论文 Algorithm 中只用观察 Query 关注
此前历史 Token 的语义；`R=256` 仅用于检验更宽 Recent 排除范围是否改变现象。

### 4.2 Query Head 概率

对 Layer `l`、Query Head `h`、观察 Query `q_i∈Qobs` 和候选 Token `j∈E_m`：

$$
z^{(m)}_{l,h,i,j}
=s_lQ_{l,h,q_i}^{\mathsf T}K_{l,v(h),j}+b_{l,h,q_i,j},
$$

$$
a^{(m)}_{l,h,i,j}
=\frac{\exp z^{(m)}_{l,h,i,j}}
{\sum_{u\in E_m}\exp z^{(m)}_{l,h,i,u}}.
$$

这里的 Softmax **仅在候选历史集 `E_m` 内重新归一化**，用于衡量 Prefill 观察窗口内部的
相对关注模式；它不等于该集合在模型完整 Attention 中承载的绝对概率质量。绝对 Attention
Mass 只在第七节的 Decode Recall 中使用完整可见集合归一化。

必须使用实际 Attention 调用边界的 RoPE 后 Q、已进入 Cache 的 K、真实 Scaling 和可验证
的 Bias/Mask 语义。当前 v1 只接受无额外有限 Bias、连续标准 Causal Cache；无法证明等价
可见性时必须失败。

### 4.3 GQA 聚合顺序

对 KV Head `v`，先分别对其 `G` 个 Query Head 完成 Softmax，再求概率均值：

$$
A^{(m)}_{l,v,i,j}
=\frac{1}{G}\sum_{h:v(h)=v}a^{(m)}_{l,h,i,j}.
$$

禁止先平均 Q、平均 Logits，或对各 Query Head 分别计算 CV 后再平均 CV；这些操作与概率
均值不等价。`A` 的轴顺序固定为 `[W,N_m]`，其中 `N_m=|E_m|`。

## 5. 论文公式版 Token CV

### 5.1 二值化阈值

将 `A` 的 `W×N_m` 个有限值展平。对每个 `(sample,l,v,m)` 独立计算：

$$
\tau^{(m)}_{l,v}=\operatorname{Quantile}_{k_m}
\left(\operatorname{vec}(A^{(m)}_{l,v})\right)\times\alpha_m.
$$

实现约定：

- 输入和 Quantile 计算使用 FP32；
- Quantile 插值方式固定为 `linear`；
- 不对 `τ` 额外 Clamp；
- 二值比较固定为 `A≥τ`；
- NaN/Inf、空候选集或非法参数直接失败；
- 参数必须写入 Condition 表和配置 Hash。

定义二值观察矩阵：

$$
B^{(m)}_{l,v,i,j}=\mathbf{1}
\left[A^{(m)}_{l,v,i,j}\ge\tau^{(m)}_{l,v}\right].
$$

### 5.2 Token 频次与 CV

每个候选 Token 跨观察 Query 的出现频次为：

$$
C^{(m)}_{l,v,j}=\sum_{i=0}^{W-1}B^{(m)}_{l,v,i,j},
\qquad C_j\in\{0,1,\ldots,W\}.
$$

定义总体均值、总体标准差和变异系数：

$$
\mu_C=\frac{1}{N_m}\sum_{j=1}^{N_m}C_j,
$$

$$
\sigma_C=\sqrt{\frac{1}{N_m}\sum_{j=1}^{N_m}(C_j-\mu_C)^2},
\qquad
CV_{prefill}=\frac{\sigma_C}{\mu_C}.
$$

标准差采用总体定义 `ddof=0`。`μC=0` 时该观测无效并必须显式标记；不得用极小常数替换
后继续当作正常 CV。主参数下理论上至少应有一个元素达到 Quantile 阈值，若仍出现零均值，
应视为实现或数值异常。

较高 `CV_prefill` 仅表示二值重要性频次集中于较少 Token；它本身不证明这些 Token 在
Decode 中仍重要，也不等价于较高 Page RCO。

### 5.3 不保存完整观察矩阵

正式运行不落盘 `A`、`B` 或完整 `C[0:N_m]`。每个 KV Head/Condition 只保存：

- `CV_prefill`、`μC`、`σC`、`τ`；
- 非零频次 Token 数与 `C=0...W` 的 65 桶 Frequency Histogram；
- 排序后的前 `Kmax` 个关键 Token 的位置和频次；
- 候选 Token 数、候选区间起止位置和有效标记。

辅助软分数 `S_j` 只用于在线确定性并列排序，不属于正式落盘字段。Debug/试运行模式可以在
独立诊断工件中保存，用于核对排序实现。

Debug 模式可对指定的一个样本、Layer、KV Head、Condition 保存小规模 `A/B/C`，但必须写入
独立诊断目录，禁止混入正式 Schema 或全量运行。

## 6. Prefill 关键 Token 集合

### 6.1 排序与并列规则

为减少整数频次并列，定义辅助软分数：

$$
S^{(m)}_{l,v,j}=\sum_{i=0}^{W-1}A^{(m)}_{l,v,i,j}.
$$

只对 `C_j>0` 的候选 Token 按以下稳定顺序排序：

1. `C_j` 降序；
2. `S_j` 降序；
3. 逻辑 Token 位置 `j` 升序。

排序的前 `min(Kmax,N^+_m)` 个位置形成 `critical_token_ranking`，其中
`N^+_m=|{j∈E_m:C_j>0}|` 是非零频次 Token 数。对预算 `b∈B`：

$$
P^{(m,b)}_{l,v}=\operatorname{prefix}
\left(critical\_token\_ranking,\min(b,N^+_m)\right).
$$

因此必须满足集合嵌套：

$$
P^{(m,128)}\subseteq P^{(m,512)}\subseteq P^{(m,2048)}.
$$

非零频次 Token 不足 `Kmax` 时使用 `UINT32_MAX` 填充位置、频次尾部填 0，并保存
`critical_token_valid_count`。不得用 `C_j=0` 的 Token 或
Initial/Recent Token 补入预算以凑满数量。

### 6.2 两类预测集合

为区分“Prefill 学到的预测能力”和“包含固定保护区的系统式保留效果”，定义：

1. **`learned_middle`：** 仅包含 `P^(m,b)`；这是 B2 的核心预测集合。
2. **`augmented_prompt`：** `P^(m,b)` 加 Initial 区间 `[0,I)` 和 Recent 区间
   `[T0-R,T0)`；只作为解释性辅助指标。

即：

$$
P^{(m,b,learned)}_{l,v}=P^{(m,b)}_{l,v},
$$

$$
P^{(m,b,augmented)}_{l,v}
=P^{(m,b)}_{l,v}\cup[0,I_m)\cup[T_0-R_m,T_0).
$$

主结论必须首先报告 `learned_middle`。若 `augmented_prompt` 很高而 `learned_middle` 很低，
不能声称 Prefill 频次成功预测了 Decode；高覆盖可能只来自固定保护区。

## 7. Decode Attention Mass 与 Prefill-to-Decode Recall

### 7.1 Decode 完整概率

对 Decode Step `s`：

$$
q_s=T_0+s,
$$

完整可见逻辑位置为 `M_s={0,...,q_s}`。对 Query Head `h`：

$$
p_{l,h,s,j}
=\operatorname{softmax}_{u\in M_s}
\left(s_lQ_{l,h,q_s}^{\mathsf T}K_{l,v(h),u}
+b_{l,h,q_s,u}\right)_j.
$$

KV Head 概率为同组 Query Head 完整概率的均值：

$$
\bar p_{l,v,s,j}=\frac{1}{G}\sum_{h:v(h)=v}p_{l,h,s,j}.
$$

与 Prefill 观察矩阵不同，Decode 必须在全部真实可见 Token 上归一化，不能只在 Prompt 或
关键集合内重新 Softmax。

### 7.2 核心 Recall 指标

对集合类型 `u∈{learned_middle,augmented_prompt}`，其全部可见 Attention Mass 为：

$$
Recall^{all}_{l,v,s,m,b,u}
=\sum_{j\in P^{(m,b,u)}_{l,v}}\bar p_{l,v,s,j}.
$$

当前 Decode 对原 Prompt 的总 Attention Mass 为：

$$
Mass^{prompt}_{l,v,s}=\sum_{j<T_0}\bar p_{l,v,s,j}.
$$

Prompt 条件 Recall 定义为：

$$
Recall^{prompt}_{l,v,s,m,b,u}
=\frac{Recall^{all}_{l,v,s,m,b,u}}
{Mass^{prompt}_{l,v,s}}.
$$

`Mass_prompt=0` 时条件 Recall 无效并写 `NaN`；不得除以极小常数制造数值。由于关键集合均
来自 Prompt，正常情况下应满足：

$$
0\le Recall^{all}\le Mass^{prompt}\le1,
\qquad
0\le Recall^{prompt}\le1.
$$

本文中的 Recall 是“Prefill 集合对 Decode Attention Mass 的覆盖率”，不是任务准确率、
检索召回率或生成质量指标。报告中必须使用全名或明确缩写，避免混淆。

### 7.3 单调性与完整性约束

同一 `(sample,l,v,s,m,u)` 下，预算增大不得使 Recall 降低；允许 FP32 累加误差容限
`1e-6`。同一预算下 `augmented_prompt` Recall 不得小于 `learned_middle` Recall。

若关键集合包含超出 Prompt 的位置、重复位置、Sentinel 被当作有效位置、Decode Step 不连续
或 Layer 不齐全，样本必须失败。

## 8. 在线算法与资源边界

### 8.1 Prefill 在线计算

正式实现只读取 Probe 回调中瞬时 Q/K：

1. 选择最后 `W=64` 个 Query；
2. 按 KV Head/GQA Group 和 Key Block 计算候选区间 Logits；
3. 对每个 Query Head 做稳定 Softmax，再按 GQA Group 平均；
4. 每次只物化一个 KV Head、一个 Condition 的 `W×N_m` FP32 观察矩阵；
5. 立即计算 Quantile、`B/C/CV`、Histogram 和 Top-`Kmax` 排名；
6. 紧凑结果移至 CPU 后释放所有 GPU 临时张量。

不得物化 `T0×T0` Attention，也不得跨 Layer 长期保留 Q/K 或多个 Layer 的观察矩阵。对 32K
输入，主路径应以单 KV Head/Condition 的 `64×N_m` 临时块为显存上界；真实峰值必须由试运行
测量，不能仅由理论尺寸宣称可行。

不同阈值但候选区间相同的 Condition 应复用同一观察矩阵；不同 `R` 的条件可顺序计算，避免
同时驻留。优化不得改变 Quantile、GQA 聚合或排序语义。

### 8.2 Decode 在线计算

Decode 使用两遍 Key Block：

1. 第一遍计算每个 Query Head 在完整可见集合上的全局 `logZ`；
2. 第二遍得到真实概率，按关键位置 Mask 累加各 Condition/Budget/集合类型的 Mass；
3. 同时累加 `Mass_prompt`；
4. 按 GQA Group 平均后输出紧凑 FP32 标量。

不得保存 Decode Token Top-K 或完整概率来离线近似 Recall；B2 Recall 必须由完整可见概率的
在线精确累加得到。Key Block 大小只控制临时资源，不得改变结果。

### 8.3 只读不变量

B2 Collector 不得：

- 原地修改 Q/K/V、Mask、Attention Output 或 Cache；
- 改变模型 Attention Backend 或调用次数；
- 让采集结果参与模型 Logits 或生成控制；
- 在回调返回后持有 GPU Q/K 引用；
- 用 B1 已保存的 Top-K 近似 B2 Recall。

## 9. 代码接口与目录组织

B2 代码继续放在 `experiments/head_analysis/` 下，不修改 KV-Base 通用核心。建议职责为：

```text
experiments/head_analysis/
  collectors/
    prefill_consistency.py   # B2 Prefill CV、关键集合、Decode Recall
    b2_attention.py          # AttentionProbe 在线 Collector 与跨阶段紧凑状态
  runners/
    b2_runner.py             # B1 轨迹加载、teacher-forced replay、资源计量
  storage/
    b2_trace_writer.py       # 独立 B2 Schema、分片、Manifest
  analysis/
    b2_consistency.py        # B2 汇总及 B1/B2 联合分析
  scripts/
    analyze_b2_run_set.py    # 命令行入口
  tests/
    test_prefill_consistency.py
    test_b2_trace_writer.py
    test_b2_runner.py
    test_b2_consistency.py
```

### 9.1 配置接口

```python
@dataclass(frozen=True)
class B2Condition:
    condition_id: str
    observation_window: int
    initial_exclusion: int
    recent_exclusion: int
    quantile_k: float
    threshold_scale_alpha: float

@dataclass(frozen=True)
class B2CollectorConfig:
    conditions: tuple[B2Condition, ...]
    critical_token_budgets: tuple[int, ...] = (128, 512, 2048)
    critical_token_max: int = 2048
    key_block_size: int = 1024
```

Condition ID 必须唯一，参数按稳定顺序序列化并写入配置 Hash。正式运行不允许在 CLI 中静默
覆盖已冻结条件；任何覆盖都产生新的 `config_hash` 和运行 ID。
配置校验还必须保证全部 Condition 的 `observation_window=64`、`recent_exclusion≥64`，预算
严格递增且最大预算等于 `Kmax`。

### 9.2 Collector 接口

```python
class PrefillConsistencyCollector:
    def collect_prefill(
        self, context: AttentionProbeContext
    ) -> B2PrefillLayerRecord: ...

    def collect_decode(
        self, context: AttentionProbeContext
    ) -> B2DecodeLayerRecord: ...
```

Collector 在 Prefill 后按 Layer 保存 CPU 关键集合，供同一样本后续 Decode 使用；样本结束后
必须显式释放状态。缺少某 Layer Prefill 状态时不得处理 Decode。

现有 `AttentionProbe` 可直接复用。其源码中的 B1 文案不改变执行语义；除非测试证明 B2 需要
新的运行边界，否则不为命名统一而重构通用 Hook。

## 10. 数据 Schema 与持久化

### 10.1 运行目录

```text
<b2_run_dir>/
  schema.json
  config_snapshot.yaml
  run_metadata.json
  resource_metrics.json
  file_manifest.json
  samples/
    <sample_id>/
      sample_metadata.json
      replay_generated_ids.json
      prefill_consistency.npz
      decode_recall_000.npz
      decode_recall_001.npz
      ...
```

JSON 记录溯源、状态和 Hash；NPZ 保存数组。Decode 每最多 128 Step 一个分片。所有文件先写
同目录临时文件，再原子替换；Manifest 保存相对路径、字节数、SHA-256、样本 ID 和 Step 范围。
`schema.json` 必须显式保存 Condition、Budget、集合类型三条轴的有序取值、数组 Shape 约定、
Sentinel/NaN 规则、Quantile 插值方式和配置 Hash；读取端不得依赖代码中的隐含默认顺序。

### 10.2 Prefill 数组

设 `M` 为 Condition 数，`Kmax=2048`：

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `cv_prefill` | `float32` | `[L,Hkv,M]` | Token 频次总体 CV |
| `frequency_mean` | `float32` | `[L,Hkv,M]` | `μC` |
| `frequency_std` | `float32` | `[L,Hkv,M]` | `σC` |
| `binary_threshold` | `float32` | `[L,Hkv,M]` | `τ` |
| `candidate_token_count` | `uint32` | `[M]` | `Nm` |
| `candidate_position_start` | `uint32` | `[M]` | `I` |
| `candidate_position_end_exclusive` | `uint32` | `[M]` | `T0-R` |
| `nonzero_frequency_count` | `uint32` | `[L,Hkv,M]` | `Cj>0` 的 Token 数 |
| `frequency_histogram` | `uint32` | `[L,Hkv,M,W+1]` | `C=0...W` 的计数 |
| `critical_token_positions` | `uint32` | `[L,Hkv,M,Kmax]` | 稳定排序后的逻辑位置 |
| `critical_token_frequencies` | `uint8` | `[L,Hkv,M,Kmax]` | 对应 `Cj` |
| `critical_token_valid_count` | `uint32` | `[L,Hkv,M]` | 有效排名长度 |
| `metric_valid` | `bool` | `[L,Hkv,M]` | 该 CV/排名是否有效 |

Histogram 每个 `(l,v,m)` 的总和必须等于 `candidate_token_count[m]`。位置尾部 Sentinel 为
`UINT32_MAX`；无效频次填 0，并由有效长度判定，不能单独依赖数值猜测。

### 10.3 Decode 数组

设分片 Step 数为 `C`、预算数为 `Nb=3`、集合类型轴
`U=[learned_middle,augmented_prompt]`：

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `decode_steps` | `uint32` | `[C]` | 样本级 Decode Step |
| `query_positions` | `uint32` | `[C]` | `T0+step` |
| `input_token_ids` | `uint32` | `[C]` | 强制输入的 `generated_ids[step]` |
| `predicted_token_ids` | `uint32` | `[C]` | 模型实际 Top-1 |
| `teacher_forced_top1_match` | `bool` | `[C]` | 是否等于 `generated_ids[step+1]` |
| `prompt_attention_mass` | `float32` | `[C,L,Hkv]` | `Mass_prompt` |
| `critical_attention_mass` | `float32` | `[C,L,Hkv,M,Nb,U]` | `Recall_all` |

预算轴固定按升序 `[128,512,2048]`，集合类型轴固定为
`[learned_middle,augmented_prompt]`；两类集合均为正式必选控制，不允许关闭或创建全零占位轴。
`Recall_prompt` 在离线阶段严格由 `critical_attention_mass / prompt_attention_mass` 派生；
`prompt_attention_mass=0` 时结果为 `NaN`。不得为方便查询重复保存这一同尺寸派生数组或完整
有效性 Mask。

### 10.4 样本与运行状态

样本状态为 `completed/partial/failed`。正式 `completed` 必须同时满足：

- Prefill 32 层记录完整且每层 8 个 KV Head/全部 Condition 有效；
- Prefill 预测的实际 Top-1 与 `generated_ids[0]` 一致；
- Decode Step 恰为 `0...G_out-2`，跨分片无缺失、重复或乱序；
- 每一步 32 层事件完整，Query 位置与输入 Token 一致；
- 全部 `teacher_forced_top1_match=true`；
- 数组 Shape、dtype、范围、单调性、Sentinel 和 Hash 校验通过。

运行状态只有在全部目标样本符合运行计划时才能标为 `completed`。失败样本必须保留错误类型、
已完成阶段、资源信息和已写工件；不得用空数组伪装成功。

### 10.5 初始逻辑存储上界

按 `L=32`、`Hkv=8`、`M=4`、`Kmax=2048`、三个预算、两类集合和 B1 理论最多
66,460 个 Decode Trace 估算：

| 主要数组 | 理论逻辑大小 |
|---|---:|
| Prefill 关键位置、频次 | 约 0.98 GiB |
| Decode `critical_attention_mass` | 约 1.52 GiB |
| Decode `prompt_attention_mass` | 约 0.06 GiB |
| Histogram、标量、索引和元数据 | 小于约 0.05 GiB |
| **合计** | **约 2.6 GiB** |

这是未考虑自然 EOS 的逻辑上界，不是实际压缩文件大小，也不包含失败诊断。正式运行和
Pilot 加上临时/失败工件应先预留至少 6 GiB；最终磁盘、时间和显存停止线必须由 6 条分层
试运行的实际结果决定，不能仅依据此估算。

## 11. B1/B2 离线联合分析

### 11.1 B2 单阶段分析

每个 Condition/Budget 分别分析：

- `CV_prefill` 的 Layer×KV Head 分布、样本内和样本间变异；
- `learned_middle` 的 `Recall_all` 与 `Recall_prompt` 时间曲线；
- Recall 的全轨迹均值、前/中/后阶段、时间斜率与曲线面积；
- 不同预算的边际收益；
- `augmented_prompt - learned_middle` 的差值；
- Condition 间 Head 排名和效应方向稳健性。

长 Decode 样本可按归一化进度 `u=s/max(1,C-1)` 比较前、中、后阶段。RULER 的短轨迹不得
插值冒充长 Decode；应单独报告实际可用 Step，并主要用于检索任务的短期跨阶段关系。

### 11.2 与 B1 的坐标映射

B2 KV Head 指标可直接与 B1 KV Head Page RCO 对齐。B1 Query Head 指标映射到 KV Head 时，
对同一 GQA Group 的四个 Query Head 做**样本内算术平均**，再开展关联分析；不得挑选其中
相关性最高的 Query Head。

主要关系包括：

| 关系 | 研究含义 |
|---|---|
| `CV_prefill × Recall` | Prefill 频次集中是否真的具有跨阶段预测能力 |
| `CV_prefill × Page RCO` | 跨 Query 一致性与 Decode Page 时间稳定性是否相关 |
| `Recall × Page RCO` | Prefill 预测能力与 Decode 集合稳定性是否同源 |
| `Recall × Token RCO` | Prefill 关键 Token 与 Decode Token 集合稳定性是否对应 |
| `CV/Recall × B1 Entropy/Top-Mass` | 一致性是否仅由稀疏、集中 Attention 解释 |

所有 RCO 必须保留 B1 原 Scope、`k`、`Δ` 和有效性定义。不得混用 `all_visible` 与
`prompt_only`，也不得把缺失长 `Δ` 的 RULER 样本填零。

### 11.3 统计单位与多重比较

正式汇总采用：

- 样本等权，而不是 Decode Step 等权；
- Layer×KV Head Profile 先在样本内汇总，再跨样本聚合；
- Spearman 作为主要排序关联，Pearson 仅作补充；
- 样本 Bootstrap；PG19 配对长度分析使用 Pair Bootstrap；
- 按 Dataset/Length/Condition/Budget 分层报告，不混成单一总体数字；
- 大规模 Layer×Head/多参数比较使用 FDR 校正，并同时报告效应量和置信区间。

主结果使用连续指标。若展示二值 Head 分组，阈值必须在分析前写入配置，且报告阈值敏感性；
不得把样本内 Top-X% 标签解释为固定模型类别。

### 11.4 证据边界

B2 可以支持或否定：

- Prefill Token 频次集中与 Decode Attention Mass 覆盖之间的统计关系；
- 这种关系的时间衰减、Head 异质性和跨任务稳定性；
- Prefill 一致性与 B1 Decode 时间稳定性的相关或独立关系。

B2 不能单独支持：

- CateKV 系统或静态 Head Mask 已被复现；
- 保留关键集合不会降低任务准确率；
- 某种 KV 压缩必然提升吞吐、延迟或显存；
- 相关 Head 行为跨模型、跨规模普遍成立；
- Attention 统计等价于 KV 表示或硬件访问规律。

## 12. 验证、试运行与完成判据

### 12.1 CPU 数学测试

使用独立 NumPy/Eager 参考实现验证：

1. 候选区间、观察 Query 和逻辑位置正确；
2. Query Head Softmax 后再 GQA 平均，且与错误的 Logit/Q 平均可区分；
3. `linear` Quantile、`×α`、`≥`、并列和 FP32 定义正确；
4. Token 频次、Histogram、总体 CV 与独立实现一致；
5. 关键 Token 的三级排序及 128/512/2048 集合嵌套正确；
6. Decode 完整 Softmax、Prompt Mass、两种 Recall 与穷举参考一致；
7. Recall 范围、预算单调性和 `augmented≥learned` 检查正确；
8. 不同 Key Block 大小不改变结果；
9. 非连续位置、非法 Mask、空候选集、缺层和重复 Step 明确失败。

### 12.2 Probe 与回放测试

小模型测试必须验证：

- 无 Probe 与只读 B2 Probe 的 Logits/生成结果一致；
- teacher-forced replay 的输入/预测 Token 与 B1 时间偏移一致；
- `G_out=1` 时 Decode Trace 为 0；
- Callback 异常后 Attention Registry、Hook 和样本状态恢复；
- Prefill 关键集合仅属于当前样本，样本间无状态泄漏。

### 12.3 Writer 与完整性测试

验证 Schema、dtype、轴顺序、Condition/Budget/集合类型顺序、Sentinel、NaN、128 Step 分片、
原子写入、文件 Hash、失败恢复及完整性判据。主动制造缺层、Hash 漂移、Top-1 不匹配、非单调
Recall 和损坏分片，确认不能标记为完成。

### 12.4 GPU 分阶段试运行

正式全量前按以下顺序验收：

1. **六条分层 Pilot：** RULER 8K/32K、GovReport 8K/16K、PG19 8K/32K 各固定 1 条；
2. **链路覆盖：** RULER 验证短 Decode，GovReport/PG19 验证长摘要/续写和全部分片；
3. **长上下文资源：** RULER 32K 与 PG19 32K 测量观察矩阵峰值显存和 Prefill 耗时；
4. **冻结资源计划：** 根据六条实测冻结时间、磁盘、显存停止线、失败恢复和批次方案；
5. 经单独审核后启动全部 100 条正式 B2，不把 Pilot 当作正式研究样本替代品。

每条试运行记录无 Probe/回放基线和 B2 的 Prefill/Decode 时间、GPU 峰值、逻辑数组字节、
实际文件字节、写入时间及 Top-1 匹配。试运行通过只证明链路和资源可行，不证明研究假设。

### 12.5 B2 开发完成判据

只有以下条件全部满足，才可称 B2 采集实现完成：

- 主公式、GQA、候选区间、关键集合与 Recall 的 CPU 精确性测试通过；
- Probe 只读、teacher-forced replay 和 B1 Hash 绑定通过；
- 独立 B2 Schema、原子 Writer、Manifest 和失败恢复通过；
- 8K、32K 与长 Decode 真实 GPU 试运行通过；
- 资源计划和正式运行集合经审核冻结。

“B2 实现完成”不等于“B2 假设成立”；现象结论必须等待正式数据和离线分析。

## 13. 实施顺序

审核冻结本规范后，按以下小阶段推进：

1. 实现纯 CPU 的 Prefill CV、关键 Token 和 Decode Recall 参考函数及单元测试；
2. 实现 `PrefillConsistencyCollector`，复用现有 `AttentionProbe`；
3. 实现 B1 轨迹加载与 teacher-forced `b2_runner`；
4. 实现独立 `b2_trace_writer`、Schema 和完整性校验；
5. 完成本地小模型链路与回归测试；
6. 执行 8K、32K 和长 Decode GPU 分阶段试运行；
7. 冻结正式资源/批次计划后采集 100 条；
8. 实现 B2 单阶段和 B1/B2 联合离线分析，形成独立结果报告。

未经审核不得跳过数学参考实现直接编写 GPU Collector，也不得在试运行前启动全量任务。

**当前实施进度（2026-11-03）：** 第 1～5 项已完成，第 6 项尚未开始。实现包括 NumPy-only
公式参考、只读 `B2AttentionCollector`、B1 Run Set/逐样本生成 ID 的双层 Hash 绑定、
teacher-forced Runner、独立 B2 Schema/原子 Writer，以及可在写入后重新核验 Hash、数组 Shape、
Step、Token 与 Mass 不变量的 Validator。Tiny Llama 已验证无 Probe 与 B2 Probe 的逐 Forward
Logits 完全一致、`G_out=1`、异常恢复和样本间状态清理。Head Analysis 全套 78 项测试、KV-Base
聚焦回归 11 项均通过，`compileall` 和 `uv pip check` 通过。以上是 CPU/小模型实现证据；8K、
32K 和长 Decode 的真实 GPU 资源与完整性仍须通过六条分层 Pilot 后才能冻结。

---

## 附录 A：论文公式、公开实现与 B2 v1 的取舍

| 项目 | 论文描述 | 作者公开实现的主要口径 | B2 v1 选择 |
|---|---|---|---|
| 观察窗口 | 最后 64 Query | 最后 64 Query | 64 |
| GQA | 组内观察矩阵均值 | Softmax 后组内均值 | Softmax 后组内均值 |
| 阈值 | `Quantile_k(A)×α` | 低/高分位插值 | 论文公式 |
| 频次单位 | Token | Chunk | Token |
| CV | Token 频次总体 CV | Chunk 频次 CV | Token 频次总体 CV |
| Initial/Recent | 暂时排除 | 正式脚本常用 64/256 | 主条件 64/64，256 为敏感性 |
| 静态分类 | 独立参考集频次 | CSV 聚合生成固定 Mask | 本阶段不做 |
| 关键集合 | 频次/软分数选取 | Chunk 化缓存策略 | Token 排名，固定预算 |
| Decode 验证 | Attention Recall | 公开代码未给出独立完整定义 | 明确定义 Mass 覆盖率 |

B2 v1 不宣称该取舍是 CateKV 唯一正确实现。它是一套以论文现象为研究起点、公式明确、
可复现且能与 HeadKV B1 结果联合分析的观测协议。

## 附录 B：实施前决策记录

本版已确认：

1. 保留 `R=64` 主条件及 `α=1.0`、`k=0.99`、`R=256` 三个预注册敏感性条件；
2. 关键 Token 预算冻结为 `{128,512,2048}`；
3. `S_j` 参与在线并列排序，但 `critical_token_soft_scores` 不进入正式全量 Schema；
4. `learned_middle` 与 `augmented_prompt` Recall 均为正式必选；
5. 正式目标集合为全部 100 条，六条分层 Pilot 只承担链路与资源验收；
6. 生成轨迹与联合分析分别绑定原始 Run Set Manifest 和 v2 Analysis Manifest；
7. 先为 Pilot/正式工件预留至少 6 GiB；正式时间、显存和批次停止线在六条 Pilot 后冻结。
