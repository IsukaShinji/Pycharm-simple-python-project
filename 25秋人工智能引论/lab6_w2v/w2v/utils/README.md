# Word2Vec 工具模块

本目录保存 Word2Vec 实验的辅助代码，主要服务于梯度检查、数据读取和基础数学工具。

## 文件导览

| 文件 | 说明 |
| --- | --- |
| `gradcheck.py` | 数值梯度检查工具，用于验证手写梯度实现是否正确。 |
| `treebank.py` | Stanford Sentiment Treebank 数据集读取与上下文采样逻辑。 |
| `utils.py` | 行归一化、softmax 等通用函数。 |
| `datasets/` | 数据集目录。 |
| `__init__.py` | Python 包标记文件。 |

## 使用场景

这些文件通常不单独运行，而是被 `../word2vec.py`、`../sgd.py` 和 `../run.py` 调用。
