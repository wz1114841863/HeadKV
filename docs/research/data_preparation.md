# 数据准备与输入验证——计划与结果

状态：已完成。当前 100 条固定输入已用于 B1/B2；项目背景见[研究总览](研究总览.md)。

## 1. 研究问题与计划

为检索、摘要和续写提供可复用的最终模型输入，并确认任务模板、生成链路和 Decode
长度是否适合后续观测。先构造候选与固定集，再执行普通推理实验 A/B；这两项实验不安装
Attention Observer，不施加 KV 干预。

| 数据集 | 8K | 16K | 32K | 用途 |
|---|---:|---:|---:|---|
| RULER | 20 | — | 20 | Needle 检索、短轨迹 |
| GovReport | 20 | 20 | — | 真实长文摘要、长 Decode |
| PG19 | 10 | — | 10 | 共享续写锚点的配对长度比较 |
| 合计 | 50 | 20 | 30 | 共 100 条 |

## 2. 方法与关键口径

RULER 使用官方生成器的 `niah_single_2`：单 Key、单数值、单 Query，essay haystack。
它与原先 `niah_single_1` 的 noise haystack 不同，二者不混用。先生成额外候选池，再按
目标长度 ±512 Token 确定性筛选各档 20 条。

GovReport 和 PG19 直接读取 Hugging Face 下载的 Parquet shard，不要求转换为
`Dataset.save_to_disk()`。GovReport 两长度不复用同一报告。PG19 保留书名和 URL 来源，
十个共享续写锚点分别构造 8K/32K 输入，形成十对样本。

每条样本先按 Llama 聊天模板渲染 Prompt，再用
`tokenizer.encode(prompt, add_special_tokens=False)` 得到最终输入。核心字段如下：

| 字段 | 含义 |
|---|---|
| `sample_id/dataset/task` | 样本与任务标识 |
| `target_prompt_length/actual_prompt_length` | 目标档位与实际 Token 数 |
| `prompt/input_ids/input_sha256` | 最终文本、模型实际输入及一致性标识 |
| `max_new_tokens` | RULER 上限 128，GovReport/PG19 上限 1024 |
| `source_id/pair_id/reference_answer/metadata` | 来源、配对、参考答案与附加信息 |

后续直接读取 `input_ids`，不重新拼接文本或套用模板。8K/16K/32K 是目标长度档位，
实际长度以样本字段为准。独立检查已覆盖配额、唯一性、长度、重新分词和模板重渲染。

## 3. 主要结果

实验使用同一 Llama-3.1-8B-Instruct 快照、BF16、SDPA、Greedy Decode，每个数据与长度
条件选择两条，共 12 条。实验 A 最多生成八个 Token，12 条输入校验及模型调用全部通过。
实验 B 使用各样本完整生成上限，结果如下：

| 条件 | 实际生成 Token | 诊断 |
|---|---:|---|
| RULER 8K | 7、7 | 两条 Needle 答案均匹配，正常 EOS |
| RULER 32K | 7、7 | 两条 Needle 答案均匹配，正常 EOS |
| GovReport 8K | 713、800 | 自然结束，输出为内容相关摘要 |
| GovReport 16K | 1024、1024 | 达到上限，输出为内容相关摘要 |
| PG19 8K | 462、691 | 前者较短，后者出现较高重复 |
| PG19 32K | 941、1024 | 可形成长轨迹，后者出现较高重复 |

PG19 两条高重复样本的重复 4-gram 比例约为 0.642/0.736。该诊断随固定集保留；
B1/B2 没有通过事后替换样本改变当前结论的输入集合。

## 4. 结论与限制

固定输入可直接用于该模型，GovReport 和部分 PG19 能提供较长轨迹，但正常 EOS 意味着
不能保证每条生成达到固定长度。RULER 短答案是任务特征，只适合短时间间隔比较；
正式 B1 的 40 条检索均正确，每条只有六个 Decode 观测。

PG19 长输出不等于生成质量普遍良好；后续研究如需筛选低重复样本，应建立新集合，
不覆盖本次固定输入。实验 A/B 验证了输入和生成能力，不证明 Attention 指标正确或方法有效。
完整输入上的行为结果分别见 [B1](v1_b1_attention_analysis.md) 和 [B2](v1_b2_prefill_consistency.md)。

## 5. 数据入口

固定集保留在远端 `data/fixed_v2/`，不纳入 Git：

| 入口 | 内容 |
|---|---|
| `prompts.jsonl`、各数据集 JSONL | 100 条最终输入及任务子集 |
| `manifest.json` | Tokenizer、来源和固定集身份 |
| `selection_report.json/selected_samples.csv` | 筛选记录 |
| `validation_report.json` | 输入验证 |
| `data/runs/fixed_v2_experiment_a.json`、`data/runs/fixed_v2_experiment_b.json` | 普通推理实验报告，路径相对项目根目录 |

处理入口为 `experiments/head_analysis/scripts/prepare_dataset.py`；
使用命令和运行说明统一放在[实验 README](../../experiments/head_analysis/README.md)。
