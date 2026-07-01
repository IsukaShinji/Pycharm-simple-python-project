# Lab 8 CIFAR-10 图像分类

本目录保存 CIFAR-10 图像分类实验代码、数据和训练得到的模型权重。实验使用 PyTorch 实现轻量级 VGG 和 ResNet 结构，并在 CIFAR-10 上训练和评估。

## 文件导览

| 文件或目录 | 说明 |
| --- | --- |
| `main.py` | 数据加载、模型定义、训练、测试和权重保存入口。 |
| `data/` | CIFAR-10 数据目录。 |
| `cifar_Modified_VGG.pth` | 修改版 VGG 训练权重。 |
| `cifar_Modified_ResNet.pth` | 修改版 ResNet 训练权重。 |

## 模型内容

- `BaselineNet`：课程提供的基础卷积网络结构。
- `MyVGG`：面向 CIFAR-10 缩小后的 VGG 风格网络，包含卷积、BatchNorm、ReLU、Pooling 和 Dropout。
- `MyResNet`：手写残差块和 shortcut 分支的轻量 ResNet。

## 运行方式

```bash
python main.py
```

脚本会加载或下载 CIFAR-10 数据，依次训练 Modified VGG 和 Modified ResNet，测试每类准确率，并把权重保存到当前目录。
