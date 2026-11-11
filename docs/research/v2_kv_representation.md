# V2：KV Head 表示结构与 Attention 行为——计划与结果

状态：方案已确认；只读采集、Prompt 全量统计、512 位置抽样、逐层落盘和固定输入回放 Runner 已完成 CPU 小模型验证，尚未启动远程真实模型实验。本文作为该阶段唯一报告，后续继续补充实际方法、结果和限制。背景及已完成工作见[研究总览](研究总览.md)。

## 1. 核心假设与三个研究层次

**核心假设：不同 KV Head 在 Attention 访问行为上表现出的稳定差异，是否对应其 K/V 数值分布、特征能量分布、有效秩及跨 Head 表示相关性上的差异？**

B1 已观察到 Head 时间稳定性差异及跨条件相对排序一致性。B2 及其探索性留出分析发现，Prefill CV 与 Decode 稳定性相关，但在当前同条件、简单线性模型下，加入当前 CV 未见超出 Head 历史稳定性基线的稳定增量。这为进一步研究 Head 表示结构提供动机，尚不能证明差异来自 K/V 的某种结构。

本轮限定为六条输入的 Pilot，依次回答三个问题：

| 层次 | 核心问题 | 主要观测 |
|---|---|---|
| 单 Head 表示分布 | 不同 Head 的表示结构是否存在差异？ | K/V 数值范围、向量范数、特征能量、协方差谱和有效秩 |
| 跨 Head 表示关系 | 不同 Head 是否具有共享的潜在表示特征？ | 同一批 Token 上的线性 CKA 及位置置换参照 |
| Attention 与 KV 表示的联系 | Attention 行为不同的 Head，其表示是否也有对应差异？ | 表示指标、CKA 与 B1 Page RCO 对齐 |

Pilot 用于确认采集正确性、成本及探索性现象，不承担独立新输入预测、因果检验或压缩收益验证。保留连续 Head 指标，不预先划分固定的 Stable/Unstable 类别。

参考来源：

- [FlexiCache](https://arxiv.org/abs/2511.00868)：Head 的 Decode 时间稳定性差异；V2 沿用 B1 的真实 Attention Page Mass/RCO，不复现其 Page 管理系统。
- [CateKV](https://arxiv.org/abs/2608.30295)：Prefill 跨 Query 特征；已有 CV 作为补充参照，不新增参数搜索。
- [KV Cache Transform Coding（KVTC）](https://arxiv.org/abs/2511.01815)：K/V 特征冗余、跨 Head 正交对齐和 RoPE 影响。本文的逐 Head 谱与 CKA 不等同于论文的整体 PCA 或 Procrustes 实验，不将相似现象写成其压缩效果复现。

## 2. 样本与执行范围

复用固定输入、模型快照、BF16、SDPA、Batch=1 以及完整无压缩 Cache。模型为 Llama-3.1-8B-Instruct：32 层、8 个物理 KV Head、每 Head 128 维；不将 GQA 的 32 个 Query Head 当作 32 个独立 KV Head。

| 条件 | 样本 | 实际 Prompt Token 数 T | B1 Decode 观测数 |
|---|---|---:|---:|
| RULER 8K | `ruler_8k_001` | 8072 | 6 |
| RULER 32K | `ruler_32k_001` | 32650 | 6 |
| GovReport 8K | `govreport_8k_001` | 7205 | 712 |
| GovReport 16K | `govreport_16k_001` | 15594 | 1023 |
| PG19 8K | `pg19_001_8k` | 8192 | 461 |
| PG19 32K | `pg19_001_32k` | 32768 | 940 |

PG19 为同一续写锚点的一对长度，不能作为两个独立来源。已有重复输出限制保留。六条输入均来自已研究的数据，Pilot 只作描述和方法验证，不做正式泛化推断。

先完整 Prefill，再回放 B1 已有生成 Token。四条长文本只回放前 256 个 Decode Forward，RULER 回放其全部 6 个 Forward；逐 Forward 核对 Top-1 与 B1 对应生成 Token，不重新选择生成分支。回放 N 次需要 B1 的前 N+1 个生成 ID：Prefill 预测 `ids[0]`，Decode 第 s 次消费 `ids[s]`、预测 `ids[s+1]`。因此长文本读取前 257 个 ID，RULER 读取前 7 个；不足时明确报错，不自动缩短协议。B1 主 RCO 来自原有完整轨迹，V2 截断回放只服务于表示采集和一致性检查。

## 3. 观测位置及只读约束

### 3.1 已核对的模型执行路径

已只读检查本地、远端 Transformers 5.2.0 的 `LlamaAttention.forward()`：

```text
Attention 模块入口：取得 layer、forward、cache_position
    ↓
k_proj(hidden_states) → reshape/transpose → K_pre
    ↓
apply_rotary_pos_emb(query, K_pre, cos, sin) → 新 Token 的 K_post
    ↓
Cache.update(K_post, V, layer_idx) → 当前完整 Cache 的 K_post/V
    ↓
Attention Function(module, Q, K_post, V, ...)
```

`v_proj` 也在 RoPE 之前执行；当前 Llama 不对 V 应用 RoPE。`AttentionProbe` 的 Attention 回调已扩展为公开实际 Q/K/V，其中 K/V 来自 Cache 更新后；早期 K 通过独立的可选投影回调公开。

### 3.2 只读观测方式

- 在实际 `k_proj` 模块上安装只读输出 Hook，从原模型这一次投影的输出抽取指定位置，得到 `K_pre`。输出原形状为 `[1,q_len,8×128]`，按模型实际 reshape 顺序恢复 Head 轴。Hook 返回 `None`，不替换输出。
- 复用 Attention 模块入口取得的 `cache_position`，将当前投影行与绝对 Token 位置关联；不能仅按当前序列长度推测位置。Prefill 与 Decode 的同层同 Forward 记录必须配对。
- 在现有 Attention Function 边界读取实际传入的 Cache 更新后 `K_post/V`。对新 Token，以绝对位置定位到 Cache 对应行；Prompt 全量统计仅在 Prefill 执行。
- 不再次调用 `k_proj`、不重算或逆变换 RoPE 来生产 `K_pre`、不改写 Attention 输入或 Cache。只把必要的统计和抽样向量转到 CPU，不跨层持有完整 GPU K/V 引用。

这些改动拟局限于 `experiments/head_analysis/` 的 runtime、Collector 和运行组装。实现时根据上述边界选择最小扩展，不提前引入通用插件系统，也不修改安装的 Transformers 源码。

`runtime/attention_probe.py` 的 `on_key_projection` 提供 `KeyProjectionContext.key_pre` 和绝对 `token_positions`；`on_attention` 提供既有 Context 的 `key` 与新增 `value`。两个回调按样本、Layer、Forward 和位置关联，投影 Hook 仅在启用早期回调时安装。Probe 不保留 K_pre。

`collectors/kv_representation.py` 在同步回调中抽样、分块计算 CPU 统计，并把逐层记录交给 `storage/kv_representation.py`。Collector 不持有完整 GPU K/V；Decode 期间仅保留 Prefill 抽样 K_post/V 的 CPU 副本，以检查历史 Prefix。Writer 复用现有 NPZ/JSON 原子写入，完整收齐所配置的全部 Layer/检查点且 Prefix 检查通过后，才标记样本完成。

`runners/kv_representation_runner.py` 复用 B2 的固定输入和 B1 轨迹读取、执行契约检查、`teacher_forced_replay` 及共用模型加载/计时。直接使用冻结 `input_ids`，不重新分词；外层安装 V2 Probe，逐层落盘，全部 Top-1 核对通过后才调用 Writer 完成。保存来源样本、实际回放 ID、检查点和必要运行信息。异常时停止后续样本、保留部分输出并记录失败；不覆盖既有目录，也不增加 Git 状态检查或 Hash 清单。

V2 的统计和 NPZ 写入发生在同步回调内，因此记录的 Forward/回放耗时包含这些开销，不沿用 B1/B2“排除写盘”的解释。GPU 路径记录峰值显存；当前只验证了 CPU 路径，真实模型资源成本仍待实测。

### 3.3 RoPE 与位置正确性

当前目标快照配置使用 `rope_type=llama3`，对应当前 Transformers 实现的幅值因子为 1。理论上单 Token 的旋转保持范数：

$$\|K_{\mathrm{pre},i}\|_2=\|K_{\mathrm{post},i}\|_2.$$

报告抽样 Token 的前后绝对范数差及相对差；近零向量以绝对差解读。允许 BF16 舍入误差，具体验收容差先由小模型 FP32/BF16 直接对照确定，再固定用于真实模型，不在遇到失败后自动放宽。此结论限定于该 RoPE 配置，不能无条件外推到带额外幅值缩放的变体。

范数一致不是充分证明：仍需检查绝对位置、Head 顺序、Hook 调用次数，以及 Attention 输入的新 Token 与更新后 Cache 对应位置的一致性。不同 Token 也可能范数接近，不能仅凭范数通过验收。

## 4. 全量统计与抽样原始向量

令完整 Prompt 为 `[0,T)`，中段为 `[64,T−64)`。所有层及 Head 使用相同位置定义。

| 表示 | 完整 Prompt | 中段候选区 | 512 个抽样位置 |
|---|---|---|---|
| K_post | 全量 Token 紧凑统计 | 全量 Token 紧凑统计 | 保存原始向量 |
| V | 全量 Token 紧凑统计 | 全量 Token 紧凑统计 | 保存原始向量 |
| K_pre | 不做全量统计 | 不做全量统计 | 保存原始向量 |

512 个位置由最前 32、最后 32 和中段均匀确定性选取的 448 个位置组成。中段第 j 个抽样位置为 `64+floor((j+0.5)×(T−128)/448)`，`j=0,…,447`；保持唯一、排序并记录绝对位置及区域标签。不得根据观察到的 RCO、CV 或表示结果选取位置。

原始抽样数组每种表示的形状为 `[L=32,Hkv=8,S=512,D=128]`，保留模型原始 BF16 数值。离线基于完全相同的 512 个位置，及其中相同的 448 个中段位置，分别计算 K_pre、K_post、V 的统计和谱。

**K_pre 与 K_post 的结构比较必须使用同一抽样范围。** 完整 Prompt K_post 与抽样 K_pre 的指标分开放置、独立标注，不能将其差异直接归因于 RoPE。抽样 K_post/V 与全量 K_post/V 的差异只用于检查抽样代表性。

### 4.1 协方差与精确 Norm 分位数

对任一范围内的 Token 矩阵 $X\in\mathbb R^{N\times128}$，保存 N、均值向量 μ、总体协方差 C，以及 Norm 统计：

$$\mu=\frac1N\sum_iX_i,\qquad C=\frac1N\sum_i(X_i-\mu)^\top(X_i-\mu).$$

总体协方差采用 `ddof=0`。优先以 FP64 分块 Chan/Welford 累积计数、均值和中心化二阶矩，最终保存 FP32 协方差。小模型用直接中心化 FP64 计算作参考，检查谱对保存精度的敏感性。禁止在 BF16 下直接累积 `XᵀX−Nμᵀμ`。

全量 K_post/V 的每个 Token 范数在 FP32 计算后转到 CPU，按 `linear` 方法计算精确 P10、P50、P90，不用分块分位数的平均值代替。建议保留这些 FP32 标量范数，形状每种表示为 `[L,Hkv,T]`；中段范围通过位置切片得到，不再保存重复的中段范数数组。抽样 K_pre 范数从保留的原始向量重算。

现实现按 Layer 落盘，单层范数为 `[Hkv,T]`；均值、协方差等按 `prompt/middle` 两个范围保存，另有特征能量和元素范围。采样原始向量为 `[Hkv,512,D]`。BF16 以 uint16 原始位写入 NPZ，元数据记录原始 dtype，读取时恢复 BF16；不是整数数值转换，也不隐式扩大为 FP32。两个 K/V 全量范围都不新增 K_pre 统计。

## 5. 三层指标与解释

### 5.1 单 Head 表示分布

每个样本、Layer、KV Head、表示和范围独立计算，保留范围及 N。

- 数值范围：元素最小值、最大值、最大绝对值；Token L2 Norm 的均值、P10/P50/P90 和最大值。它们反映数值大小，不直接等于 Token 的 Attention 重要性。
- 原始坐标特征能量：$e_j=N^{-1}\sum_iX_{ij}^2$，可由 $C_{jj}+\mu_j^2$ 得到。保存 128 维向量及其归一化分布。此指标依赖坐标系且包含均值能量，不能称为 PCA 方差。
- 中心化协方差谱：以对称化 C 的降序特征值 λ 定义 $E_k=\sum_{j\leq k}\lambda_j/\sum_j\lambda_j$，报告 `k=8/16/32/64`，以及达到 90% 方差的最小维数 r90。
- 有效秩：$p_j=\lambda_j/\sum_m\lambda_m$，$r_{eff}=\exp(-\sum_{p_j>0}p_j\log p_j)$。总方差为零时标记无效；保存完整谱以便改变解释比例阈值，无需重跑模型。

协方差的可忽略负特征值须有记录其大小的数值容差，再截为零；明显负特征值应判为计算或保存误差并检查，不能静默修补。Norm 不变不要求跨 Token 协方差谱不变，因为不同位置经历不同旋转。

### 5.2 跨 Head 表示关系

默认对全部 32 层、每层 8 个物理 KV Head 的 28 个无序组合计算 CKA；K_pre、K_post、V 分开。主范围固定为相同的 448 个中段抽样位置，避免混入首尾区域权重；全量统计不用于构造此 CKA。

对各自按特征中心化的 $X_c,Y_c\in\mathbb R^{448\times128}$：

$$CKA(X,Y)=\frac{\|X_c^\top Y_c\|_F^2}{\|X_c^\top X_c\|_F\,\|Y_c^\top Y_c\|_F}.$$

分母为零时标记无效，保留有效对数。该指标描述对齐 Token 的线性表示结构相似性。

位置置换参照固定为 **50 次**，seed=20261111。每次仅打乱一个 Head 的 Token 行，计算 `CKA(X_c,P_b Y_c)`；同时打乱两侧不会破坏对应关系，不能作为参照。同一样本的 Head 对和三种表示复用同一套置换索引，保持范围和 N 不变。保存每次结果、均值、P5/P95，另报告 `CKA−置换均值`。

全部六条共有 `6×32×28×3=16,128` 个观察值、806,400 个置换参照值。使用缓存的 Token Gram 矩阵及按层分批计算控制成本；先离线计时，不因计算较慢而事后挑选 Layer。确需改变次数或范围时，应在结果解读前明确修订方案；50 次不用于宣称正式尾部显著性。

置换参照破坏了共同输入内容与位置的对应关系，不能将“高于参照”解释为已排除了共享输入的影响。CKA 还受维度、N 和 Token 选择影响；不跨不同 N 直接作同口径比较。高 CKA 不证明可共享或合并 Cache，也不等于低秩或压缩后的模型质量可保持。

### 5.3 Attention 与 KV 表示的联系

使用已有样本的 B1 `prompt_only Page RCO、k=32、Δ=32` 作为主对照，保留未截断的 `rco_raw_mean`、有效 Step 对数及 `(sample_id,layer_id,kv_head)` 坐标。四条长文本适用，RULER 不参与该长间隔关系，可另列 `all_visible Token RCO、k=32、Δ=1` 的短轨迹描述。

主要关系限定为：

1. 完整中段 K_post、V 的 `r_eff / E16` 与 Page RCO 的关系。逐样本先在每层内减去该层 Head 均值，再对 256 个固定 Head 坐标计算 Spearman。
2. 每个 Head 与同层其余 7 个 Head 的平均 CKA 与该 Head Page RCO 的关系；观测 CKA 及其超出置换均值的部分分别保留。无效对不补零并记录实际邻居数，以同样 Layer 去中心化口径作描述。

K_pre 与 K_post 的关联差异仅用相同的 448 个中段抽样位置作敏感性比较。原始坐标能量和 r90 用于解释，不自动展开更多参数搜索。CKA 的 28 个 Head 对不能直接当成 28 个独立 Head 样本；邻居平均也不构成新的独立样本。

本轮以每条输入的结果及差异为主，不把 Layer、Head 或 Head 对当作独立实验样本。不在六条数据上训练预测器或给出正式总体显著性结论；PG19 配对不增加独立样本数。使用当前样本完整轨迹 RCO 是事后结构关联分析，不是 Prefill 预测实验。

## 6. Decode：只检查新增 Token 与历史一致性

第一轮采用“目的 A”：每个检查点只保存其对应的一个新 Token，不研究 Decode 整体分布，也不保存累计生成窗口。

| 样本范围 | 消费的生成 Token | B1 Decode Step s | Cache 绝对位置 | 每个检查点新增保存 |
|---|---|---:|---:|---|
| 四条长文本 | `generated_ids[31]` | 31 | T+31 | K_post/V 各 `[L,Hkv,128]` |
| 四条长文本 | `generated_ids[255]` | 255 | T+255 | K_post/V 各 `[L,Hkv,128]` |
| 两条 RULER | `generated_ids[5]` | 5 | T+5 | K_post/V 各 `[L,Hkv,128]` |

以上共 10 个新 Token 的全层 K/V。它们用于补充数值检查，不能形成 Decode K/V 总体分布或有效秩结论。

各检查点另读取 Prefill 保存的 512 个 K_post/V 位置，与当前 Cache 对应行作一致性比较，保存误差摘要，不重复写入整套历史向量。当前标准无压缩 Llama Cache 对既有位置应保持一致；若发现变化，先定位执行/位置/配置原因，不将其当成表示演化。Prefix 检查须记录对比数值是否完全一致及最大绝对差。

## 7. 验证与推进顺序

| 步骤 | 必须确认的内容 |
|---|---|
| 小模型采集 | 每层每 Forward 的真实 k_proj 只执行一次；Hook 不替换输出；K_pre 位置和 Head 顺序正确；新增 K_post/V 与 Cache 对应行一致 |
| 数值参考 | FP64 直接中心化参考与分块统计一致；Norm 分位数与 CPU 全量标量计算一致；已知低秩/常量数据的谱、有效秩和无效标记正确 |
| RoPE 与 CKA | 同位置前后范数误差在验证容差内；CKA 自比较为 1，正交变换与整体缩放不改变其值；置换参照可重现，但不要求每个 Head 对的观察值一定高于置换 |
| 推理与清理 | 无采集/带采集 Token、Top-1 和小模型 Logits 对照；异常后 Hook 释放；不保留跨层完整 GPU Tensor |
| 单条 8K | 先运行 GovReport 8K，实测时间、GPU 峰值及落盘大小；检查两个 Decode 新位置和历史 Prefix |
| 六条 Pilot | 再验证 16K/32K 与 RULER 短轨迹，对齐旧样本及 B1 指标，覆盖全部 32 层 |
| 离线判断 | 按三个研究层次解释结果，记录不支持假设的结果，再决定是否扩大样本 |

真实模型 Pilot 仍需依据当时 GPU 和磁盘资源安排，不把磁盘估算当成运行时间或显存承诺。

## 8. 空间预算

以下按 BF16 原始向量、FP32 落盘协方差/Norm、32 层×8 KV Head×128 维估算，均为未计压缩的逻辑字节。六条实际 Prompt 合计 104,481 Token。

| 内容 | 计算口径 | 空间 |
|---|---|---:|
| Prefill 抽样原始向量 | `6×3(K_pre/K_post/V)×32×8×512×128×2` 字节 | 576 MiB |
| 全 Prompt/中段协方差 | `6×2(K_post/V)×2(scope)×32×8×128×128×4` 字节 | 384 MiB |
| 全 Prompt Token Norm | `104481×2×32×8×4` 字节，中段复用切片 | 约 204 MiB |
| Decode 新 Token | `10×2(K_post/V)×32×8×128×2` 字节 | **1.25 MiB** |
| 离线谱、CKA 和其 50 次参照、均值、索引 | 紧凑数组，抽样协方差可从原始向量重算，不重复持久化 | 预留 10～30 MiB |

主要文件约 **1.15～1.2 GiB**，建议预留 **2～4 GiB** 作为落盘和临时空间预算。若验证后必须将全部协方差保存为 FP64，增加约 384 MiB。具体格式不得意外把原始 BF16 扩为 FP32 后仍引用本预算。

此前 Decode 约 146 MiB 的估算对应累计生成窗口；本方案只保存上述 10 个 Token，已改为 1.25 MiB。运行中的完整 Cache、模型权重、统计工作区和 CPU/GPU 峰值内存另计。All-layer CKA 增加离线计算成本，输出存储仍较小。

2026-11-11 只读检查中，项目所在共享 NFS 的 df 显示总量 163T、已用 153T、可用约 2.1T、使用率 99%；项目 `data/` 约 17G，其中 runs 约 15G。此为时间点记录，启动前需重新确认，不因本方案预算小而认为共享空间可长期保证。

## 9. 旧数据依赖、代码组织与结果入口

| 旧工件 | V2 用途及保留建议 |
|---|---|
| 固定 Prompt input_ids、来源和 Pair 元数据 | 原输入及位置对齐，必须保留 |
| B1 生成 Token、样本信息和 Run Set | 固定回放及 Top-1 对照，必须保留 |
| B1 完成数值包，或已核验的 B2 联合数值副本 | 逐样本 Head RCO、有效数与 GQA 坐标，必须保留 |
| B2 CV 和联合数值包、此次留出结果 | 研究背景和补充对照，继续保留 |
| B1/B2 原始逐 Step Trace | 本方案不重算全部 Attention，但可用于位置/指标复核；当前 B1 Reader 仍检查相关文件，Pilot 验证前继续保留 |
| 模型及 Tokenizer 快照 | 相同轨迹与表示来源，继续复用 |

已检查的 B1/B2 正式工件保存 Attention 统计、频次和关键位置，没有保存可供本轮直接使用的原始 K/V 或相应协方差。因此本轮需要模型回放，不能仅靠旧数值包生成 K/V 表示结果。不安排删除旧数据，也不重复采集已经有的 RCO 或 CV。

沿用能力目录：runtime 负责只读采集点；Collector 负责分块统计和抽样；runner 负责固定输入回放和阶段检查；storage 负责 CPU 数值落盘；analysis 负责谱、CKA 和旧 RCO 对齐。优先复用现有实现，只为真实复用需要提取公共能力。命令和最终代码入口写入[实验 README](../../experiments/head_analysis/README.md)，本报告不另设逐步骤开发文档。

拟将新数据保存到 `data/runs/v2_kv_pilot_<运行名>/`，分析输出保存到 `data/analysis/v2_kv_pilot_<分析名>/`，均使用新目录。保留样本、相位、Forward/Step、Token 位置、表示类型、范围/N、Head/维度及 dtype 的必要索引；完整字段表以实际工件为准，不将逐文件 Hash 或机器核验流水扩写进研究报告。

## 10. 结果与后续决策

已完成 CPU Tiny Llama（2 层、4 Query Head、2 KV Head、Head dim=8）的采集点验证，使用 Llama 3.1 类型 RoPE、FP32/BF16 两种 dtype：

- 一次完整 Prefill 加三个单 Token Decode，K_pre 的样本、Layer、Forward、位置与 K_post/V 对齐。
- K_pre 与独立线性投影计算、K_post 与基于实际 cos/sin 的旋转公式、V 与独立 Value 投影计算均一致；真实执行期间每层每 Forward 的 Key Projection 和 RoPE 各调用一次。
- K_post/V 与实际 Cache 数值一致；新增 Token 位置和历史 Prompt Prefix 检查通过。开启采集前后的 Logits、Greedy 生成 Token 及各次 Cache 均一致。
- 最大 K 范数相对差为 FP32 约 `9.25×10⁻⁸`、BF16 约 `1.40×10⁻³`；当前小模型测试分别使用 `1e-6` 和 `1e-2` 上限。该记录只说明本次 CPU 小模型，不代替真实模型/GPU 的误差验证。
- 两类回调抛出异常后，Probe 均清除 Hook 并恢复原 Attention 注册项，后续普通推理可继续。

随后完成 Collector/Writer 的 CPU 小模型验证：

- 512 个位置在 576、640、8K、32K 长度下均唯一并符合公式；对长度不足的 Prompt 明确报错，不悄悄改变抽样配额。
- 使用大均值、小方差输入及多种分块大小，Chan 协方差与直接中心化 FP64 参考一致；常量输入的协方差为零。Norm 和精确分位数、特征能量、元素范围均与独立计算一致。
- 两层 Tiny Llama、640 Token Prompt，在 FP32/BF16 下完成 256 个 Decode Forward，保存并读回全部 Prefill Layer 及 `s=31/255` 的单 Token 记录；原始 BF16 位保留，统计范围与位置对齐，Logits 与无采集执行一致。
- 短 Decode 单独验证 `s=5`；Prefix 一致性检查通过。人为改变历史 Key 时，错误摘要保留并报错；缺失 Layer/检查点或 Prefix 检查失败的样本不能发布为完成。

固定输入回放 Runner 也已完成 CPU 端到端验证：

- BF16、两层 Tiny Llama、640 Token Prompt：独立 Greedy 生成 260 个 ID，Runner 取前 257 个回放 256 次 Decode；逐次 Top-1 与原轨迹一致。512 位置 K_pre/K_post/V 保持原始位，检查点新 Token 的 K/V、绝对位置及历史 Prefix 均与无采集执行中的 Cache 对齐。
- RULER 分支回放 6 次、只保存 `s=5`。整批入口实际读取小型 B1 Run Set、冻结输入及生成 ID 文件，完成采集与状态登记；测试中仅替换模型加载为本地构造的小模型。
- 故意使最后一次 Top-1 不符，即使最后检查点已落盘，样本与运行也不能发布为完成；Hook 恢复后可继续正常采集。轨迹不足时拒绝采集，已有目录拒绝覆盖。
- 本轮 Runner 测试 5 项及关联模块回归 46 项均通过；CLI 帮助入口可正常调用。计时计数符合一次 Prefill 加 6/256 次 Decode。

当前仍无 V2 真实模型表示数据或科学分析结论。下一步经用户确认后安排单条 8K、六条 Pilot，实测 GPU 正确性及成本，再实现谱/CKA/RCO 离线分析；若无明显关联也如实记录。六条 Pilot 成功不等于已证明跨输入规律或压缩收益。
