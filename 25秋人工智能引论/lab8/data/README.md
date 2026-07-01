# CIFAR-10 数据目录

本目录用于保存 Lab 8 图像分类实验的数据。`main.py` 通过 `torchvision.datasets.CIFAR10(root='./data', download=True)` 读取或下载数据。

## 子目录

| 目录 | 说明 |
| --- | --- |
| `cifar-10-batches-py/` | CIFAR-10 Python 版本原始批次文件。 |

## 注意事项

这是数据缓存目录，不包含训练代码。删除后重新运行 `../main.py` 会尝试再次下载数据。
