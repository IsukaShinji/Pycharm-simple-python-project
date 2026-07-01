# CIFAR-10 Python 批次文件

本目录是 CIFAR-10 数据集的 Python 版本解压目录，由 `torchvision` 读取。

## 文件导览

| 文件 | 说明 |
| --- | --- |
| `data_batch_1` 到 `data_batch_5` | 训练集批次文件。 |
| `test_batch` | 测试集批次文件。 |
| `batches.meta` | 类别名称等元信息。 |
| `readme.html` | CIFAR-10 数据集自带说明。 |

## 使用方式

通常不需要直接读取这些文件，`../main.py` 会通过 `torchvision.datasets.CIFAR10` 自动加载。
