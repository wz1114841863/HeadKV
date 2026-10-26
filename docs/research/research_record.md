# 实验记录

##　实验目标
目标一:研究 Head 的分布异质性
分析不同 Head 的 Attention 和 KV 表示在统计特征上是否存在显著差异.
例如,Attention 集中程度/历史 Token 访问位置/KV 数值分布和有效秩等.

目标二:研究 Head 特征的稳定性
判断这种差异是模型相对固有的特征,还是受输入任务/Prompt 内容和 Decode 阶段影响.
重点研究跨输入稳定性/短期与长期时间稳定性,以及 Prefill 对 Decode 的预测能力.

目标三:研究不同 Head 特征之间的关联
例如,Attention 集中程度是否影响时间稳定性?具有相似 KV 表示的 Head 是否具有相似的 Attention 行为?
重点判断不同指标之间是否存在稳定的统计关系,而非预先将 Head 划分为固定类别.

这三个目标分别对应 分布差异 → 分布稳定性 → 分布间关系.

##　实验规划
基于之前开发的KV-Base框架, 针对HeadKV进行开发.

### 模型选择
这里先选择了LLama-3.1-8B-Instruct作为验证模型.

### 数据集选择
RULER负责可控历史检索, GovReport负责长文本摘要, PG19负责连续长生成.
不要求它们的Decode长度完全一致,而是让不同任务分别承担不同的观察目标; 跨任务的直接比较则使用共同的前64或128Step,以及相近的Prompt长度.
在这三类数据中, 最应该优先构建的是PG19的10组配对样本.它们的输入上下文/续写起点和输出长度比较容易控制,很适合作为KV-Base新观测代码的首批真实测试数据.

## 具体实现
数据集	          8K	  16K	  32K	 主要用途
RULER	         20	      _	      20	检索位置/Head 功能分布
GovReport	     20	      20       _	真实长文本摘要
PG19	         10	      _	      10	配对长度/长 Decode
合计	         50	      20	  30

### 数据构建约定

固定数据集包含 100 个输入:RULER `niah_single_2` 在 8K/32K 各 20 个,
GovReport 在 8K/16K 各 20 个,PG19 的 10 个共享续写锚点配对分别构成 8K/32K
输入各 10 个.构建时必须审计每个"数据集 × 目标长度"组的精确计数,且 GovReport
跨长度组不得复用同一报告.

GovReport 和 PG19 的远程原始输入是 Hugging Face 下载的 Parquet shard,而不是
`Dataset.save_to_disk()` 目录.数据准备脚本应直接读取这些不可变 shard;不因接口
格式差异重下载或复制原始数据.PG19 需保留其 `url` 作为 source ID/
`short_book_title` 作为书名元数据.

RULER 使用官方生成器的 `niah_single_2`:单 key/单数值/单 query,essay 作为
haystack.旧的 `niah_single_1`(noise haystack)保留为独立的历史原始数据,不得覆盖
或误标为 `single_2`.最终固定提示集以新版本输出目录/manifest 和 SHA-256 冻结.
