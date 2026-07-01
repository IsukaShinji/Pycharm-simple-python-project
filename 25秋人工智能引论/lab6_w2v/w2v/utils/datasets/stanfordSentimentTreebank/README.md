# Stanford Sentiment Treebank

本目录保存 Word2Vec 实验使用的 Stanford Sentiment Treebank 数据文件。目录中已有数据集自带的 `README.txt`，这里补充项目内的快速说明。

## 文件导览

| 文件 | 说明 |
| --- | --- |
| `datasetSentences.txt` | 句子文本及句子编号。 |
| `datasetSplit.txt` | 训练、验证、测试划分信息。 |
| `dictionary.txt` | 短语到短语编号的映射。 |
| `sentiment_labels.txt` | 短语编号到情感分数的映射。 |
| `SOStr.txt`、`STree.txt` | 树结构和分词相关数据。 |
| `original_rt_snippets.txt` | 原始 Rotten Tomatoes 片段。 |
| `README.txt` | 数据集原始说明。 |

## 使用方式

该目录由 `../../treebank.py` 读取，用于生成随机中心词和上下文窗口。复现实验时保持文件名和目录结构不变。
