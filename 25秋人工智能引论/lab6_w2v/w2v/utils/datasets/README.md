# 数据集目录

本目录用于存放 Word2Vec 实验所需的数据集。目前包含 Stanford Sentiment Treebank 的本地副本。

## 子目录

| 目录 | 说明 |
| --- | --- |
| `stanfordSentimentTreebank/` | Stanford Sentiment Treebank 原始文本、划分文件、短语字典和情感标签。 |

## 注意事项

数据集文件是训练和采样的输入，不建议手动修改。若数据缺失，可回到 `../../` 运行 `get_datasets.sh` 重新准备。
