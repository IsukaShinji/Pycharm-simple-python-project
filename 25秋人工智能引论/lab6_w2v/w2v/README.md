# Word2Vec 实现

本目录包含 Word2Vec 实验的主要实现代码、数据工具、环境文件和部分训练输出。代码以 NumPy 为主，实现 Skip-gram 词向量训练流程。

## 文件导览

| 文件或目录 | 说明 |
| --- | --- |
| `word2vec.py` | 核心实现：sigmoid、naive softmax loss、negative sampling loss、Skip-gram 和梯度检查入口。 |
| `sgd.py` | 随机梯度下降训练逻辑。 |
| `run.py` | 训练入口，读取数据并保存词向量结果。 |
| `utils/` | 梯度检查、通用工具和 Stanford Sentiment Treebank 数据读取工具。 |
| `get_datasets.sh` | 数据集下载脚本。 |
| `env.yml` | 实验环境配置。 |
| `saved_params_*.npy`、`saved_state_*.pickle` | 不同迭代步保存的训练参数和状态。 |
| `word_vectors.png` | 训练后词向量可视化结果。 |

## 运行提示

先准备依赖和数据，再运行训练入口：

```bash
python run.py
```

如果需要复现实验，应先检查 `utils/datasets/` 下的数据是否完整，以及当前环境是否满足 `env.yml` 中的依赖版本。
