# 数据集准备

本文记录 HeadKV 当前固定数据集的选择、处理和验证结果。其目标是为后续的
Prefill/Decode Head KV 分析提供可复现的输入，而不是报告模型推理性能或方法效果。
固定数据保留在远端 `data/fixed_v2/`，不纳入 Git，也不要求在本地保存模型或数据副本。

## 数据集选择

三类数据集覆盖互补的观察场景：

| 数据集 | 任务 | 主要观察目标 |
|---|---|---|
| RULER | Needle-in-a-Haystack 检索 | 控制 Needle 位置，观察检索相关的 Head 功能分布 |
| GovReport | 长文本摘要 | 观察真实长文档摘要场景中的输入差异 |
| PG19 | 文本续写 | 观察配对长度输入与较长 Decode 过程中的时间行为 |

三类任务的 Decode 长度不要求相同。跨任务比较时，应使用共同的前 64 或 128 个
Decode step，并选择相近的 Prompt 长度。PG19 的配对样本使用相同的续写锚点、
不同的上下文长度，适合作为观察代码和 Decode 时间分布的首批真实输入。

## 固定集配额

| 数据集 | 8K | 16K | 32K | 合计 |
|---|---:|---:|---:|---:|
| RULER | 20 | — | 20 | 40 |
| GovReport | 20 | 20 | — | 40 |
| PG19 | 10 | — | 10 | 20 |
| 合计 | 50 | 20 | 30 | 100 |

这里的 8K、16K、32K 是按最终模型输入 token 数控制的目标档位；实际长度记录在
每个样本的 `actual_prompt_length` 字段中。固定集的构建会逐一审计每个
“数据集 × 目标长度”组的计数。

## 原始来源与处理

### RULER

RULER 使用官方生成器的 `niah_single_2` 配置：单 key、单数值、单 query，
以 essay 作为 haystack。该配置用于更接近自然语义的单 Needle 检索场景。
历史下载的 `niah_single_1` 使用 noise haystack，与 `single_2` 是不同来源；
它保留为独立原始数据，不得覆盖或误标为 `single_2`。

为避免候选长度边界造成固定集不稳定，`niah_single_2` 先生成额外候选池，再以
目标长度 ±512 token 的严格窗口筛选。每个 8K/32K 档位从满足窗口的候选中确定性地
选择 20 条，并保留候选与选择记录作为溯源信息。

### GovReport

GovReport 的原始输入来自 Hugging Face 下载的 Parquet shard，而非
`Dataset.save_to_disk()` 目录。处理脚本直接读取这些不可变 shard，按目标长度构造
摘要 Prompt，并确保 8K 与 16K 两组不复用同一份报告。

### PG19

PG19 同样直接读取 Hugging Face 的 Parquet shard。每个来源文本保留 `url` 作为
`source_id`，保留 `short_book_title` 作为书名元数据。10 个共享续写锚点分别构成
8K 和 32K 的配对输入，因此跨长度比较能够保持相同的续写起点。

## 固定 Prompt 与记录格式

每个样本均先按 Llama 聊天模板渲染为最终 `prompt`，再以
`tokenizer.encode(prompt, add_special_tokens=False)` 得到 `input_ids`。核心字段为：

| 字段 | 含义 |
|---|---|
| `sample_id`、`dataset`、`task` | 固定样本和任务标识 |
| `target_prompt_length`、`actual_prompt_length` | 目标档位与最终 token 数 |
| `prompt`、`input_ids` | 最终渲染文本及模型实际输入 |
| `input_sha256` | `input_ids` 的哈希，用于检测输入变化 |
| `max_new_tokens` | 该样本的 Decode 上限 |
| `source_id`、`pair_id`、`reference_answer`、`metadata` | 原始来源、配对关系、任务参考答案及附加信息 |

因此，后续实验不应重新拼接原始文本或再次套用模板，而应直接使用已经冻结的
`input_ids`。这保证同一 `sample_id` 在不同实验之间具有一致的模型输入。

## 产物与当前结果

固定集位于远端 `data/fixed_v2/`：

| 文件 | 内容 |
|---|---|
| `ruler.jsonl` | 40 条：8K 20 条、32K 20 条 |
| `govreport.jsonl` | 40 条：8K 20 条、16K 20 条 |
| `pg19.jsonl` | 20 条：8K 10 条、32K 10 条 |
| `prompts.jsonl` | 三类合并后的 100 条最终输入 |
| `manifest.json` | tokenizer、来源、配额和整体哈希等溯源信息 |
| `selection_report.json`、`selected_samples.csv` | 候选筛选与最终样本选择记录 |
| `validation_report.json` | 固定集独立验证报告 |

数据处理完成后，独立验证已检查样本数、档位配额、唯一性、`input_ids` 哈希、长度、
重新分词结果和模板重渲染结果，均通过。当前 `manifest.json` 文件哈希为
`20b1fe07224de2a5fae04b3279f74e9e45b6ac088b08a76d8bf95e24b93c8f62`；
`prompts.jsonl` 的 `data_sha256` 为
`30ea0f8a85e203c8cf63a87e83805e851345c42a2a2fd9ab7f4dbdc1c63492ee`。

这些验证只证明固定 Prompt 及其输入 token 的一致性；是否能正常生成、实际 Decode
长度是否足够，以及 RULER 检索和摘要/续写质量，需由后续 Prompt 与 Decode 实验单独验证。

## Prompt 与 Decode 验证

为验证固定输入能否被真实模型直接使用，已在远端对一个分层的 12 条子集执行两项
普通推理实验。两项实验均直接将记录中的 `input_ids` 传给 `model.generate`：不重新
拼接 Prompt、不重新套用模板、不安装 Attention Observer，也不施加 Press 或其他 KV
干预。因此，它们验证的是固定输入与基础生成路径，而不是 Attention 指标或 KV 方法的
效果。

### 运行条件与选择

| 项目 | 实际设置 |
|---|---|
| 模型 | NousResearch/Meta-Llama-3.1-8B-Instruct 本地快照 |
| 模型实现 | `LlamaForCausalLM`，BF16，Greedy Decode（`do_sample=false`，`use_cache=true`） |
| Attention backend | `sdpa` |
| 输入与版本 | `data/fixed_v2/prompts.jsonl`，Git commit `e3ca6a8` |
| 选择 | RULER 8K/32K、GovReport 8K/16K、PG19 8K/32K；每个组确定性选择 2 条，共 12 条 |

初始尝试指定 `flash_attention_2`，但远端 Python 环境尚未安装 `flash_attn`，模型加载
即明确失败，未产生实验结果。为验证数据本身，正式 A/B 运行改用 `sdpa`；后续若安装
FlashAttention 并作为性能 backend 使用，须单独记录 backend 并重新确认其与采集接口的
兼容性。本节的结论不应外推为 FlashAttention 下的性能或数值结果。

实验 A 以每条最多 8 个新 token 进行最小生成探测，同时检查：`input_ids` 是否为合法
token、保存长度是否匹配、`input_sha256` 是否匹配、当前 tokenizer 重新编码是否逐 token
一致，以及当前聊天模板重渲染是否仍得到同一 `prompt`。12 条全部通过所有输入检查并
成功生成。因此，固定集已经提供可直接喂给该模型快照的确定性输入。

实验 B 使用每条记录的 `max_new_tokens`（RULER 128，GovReport/PG19 1024）执行完整
Greedy Decode，并记录实际长度、EOS、重复指标和任务诊断。报告中的 `passed=true` 仅表示
12 条输入检查与模型调用均成功；提前 EOS、重复或任务质量诊断会保留在 `anomalies`，不会
使该执行状态自动变为失败。

### 实验 B 结果

| 数据集与档位 | 样本 | 实际生成 token | 诊断结论 |
|---|---|---:|---|
| RULER 8K | `ruler_8k_001`、`ruler_8k_002` | 7、7 | 均在短答案后 EOS；参考 Needle 答案 2/2 匹配 |
| RULER 32K | `ruler_32k_001`、`ruler_32k_002` | 7、7 | 均在短答案后 EOS；参考 Needle 答案 2/2 匹配 |
| GovReport 8K | `govreport_8k_001`、`govreport_8k_002` | 713、800 | 都在自然摘要结束后 EOS；输出为内容相关的英文摘要 |
| GovReport 16K | `govreport_16k_001`、`govreport_16k_002` | 1024、1024 | 均达到上限；输出为内容相关的英文摘要 |
| PG19 8K | `pg19_001_8k`、`pg19_003_8k` | 462、691 | 前者低于 512-token 最低期望；后者达到长度但出现高重复 |
| PG19 32K | `pg19_001_32k`、`pg19_003_32k` | 941、1024 | 前者在 EOS 前达到较长 Decode；后者达到上限但出现高重复 |

RULER 的 4 条均正确检索参考答案；其“提前 EOS”是答案只含一个短数值的正常任务行为，
不能用作长 Decode 时间轨迹。GovReport 的 4 条均产生 713 token 以上的摘要，其中 16K
两条达到 1024-token 上限，是当前最稳定的长 Decode 候选。PG19 已证明能够产生 691、941
和 1024 token 的续写，但不能据此宣称当前抽样的所有 PG19 条目都适合时间稳定性研究：
`pg19_001_8k` 仅生成 462 token，`pg19_003_8k` 和 `pg19_003_32k` 的重复 4-gram 比例分别为
0.642 和 0.736，超过本次 0.6 的告警阈值。后续正式采集应从候选池按实际 Decode 长度、
EOS 和重复指标筛选 PG19 样本，而不应仅按固定 sample ID 取样。

远端运行报告保存在 `data/runs/fixed_v2_experiment_a.json` 和
`data/runs/fixed_v2_experiment_b.json`，包含输入哈希、模型与 tokenizer 路径、backend、
实际生成文本哈希、每条生成时长、EOS 和重复诊断。它们位于运行产物目录，不属于
`data/fixed_v2/` 的冻结数据集本体，也不纳入 Git。
