# 人工智能引论课程实验

本目录整理了 2025 秋季《人工智能引论》课程中的实验、作业和模型实现。内容从传统机器学习基础实验，到词向量、Transformer 机器翻译、图像分类和音频处理，覆盖了人工智能课程中“算法理解 + 代码实现 + 实验验证”的主要训练环节。

这个目录不是一个单一工程，而是一组课程实验项目集合。每个子目录对应不同主题，适合按知识点阅读，也可以作为我在 AI 基础学习阶段的阶段性记录。

## 内容概览

- 早期实验：剪枝、蒙特卡洛法、K-Means 等基础算法练习。
- 课程作业：围绕搜索、推理或模型实现的独立代码。
- Word2Vec：实现 Skip-gram、Softmax 损失、负采样和梯度检查。
- Transformer NMT：基于 Transformer 的英中机器翻译实验。
- CIFAR-10 图像分类：实现并训练修改版 VGG 和 ResNet。
- 音频实验：通过 notebook 和音频文件完成声音处理相关任务。

## 子目录说明

| 目录 | 说明 |
| --- | --- |
| `早期lab/` | 课程前期算法实验，包括剪枝、蒙特卡洛估计圆周率、K-Means 聚类。 |
| `作业/我的解答/` | 课程作业的个人实现。 |
| `lab6_w2v/` | Word2Vec 实验，实现词向量训练中的核心损失函数、梯度和 SGD 流程。 |
| `lab7/transformer-nmt-pub/` | Transformer 神经机器翻译项目，包含数据、模型、训练脚本和已有 README。 |
| `lab8/` | CIFAR-10 图像分类实验，包含修改版 VGG、ResNet 和训练得到的权重文件。 |
| `lab9/` | 音频实验，包括 notebook、PDF 说明和示例音频文件。 |

## 技术栈

- Python
- NumPy
- PyTorch
- torchvision
- scikit-learn
- Jupyter Notebook

不同实验依赖不同，建议进入具体子目录后按脚本或原有说明安装环境。`lab7/transformer-nmt-pub/requirements.txt` 和 `lab6_w2v/w2v/env.yml` 中保留了部分实验环境信息。

## 重点实现

### Word2Vec

`lab6_w2v/w2v/word2vec.py` 中实现了：

- 数值稳定版 sigmoid；
- naive softmax loss and gradient；
- negative sampling loss and gradient；
- Skip-gram 模型；
- 梯度检查与训练入口。

### Transformer NMT

`lab7/transformer-nmt-pub/` 中包含完整的 Transformer 翻译实验结构，包括数据预处理、模型定义、训练脚本和保存的模型权重。

### CIFAR-10 分类

`lab8/main.py` 中实现了两个卷积网络版本：

- 修改版 VGG：使用卷积块、BatchNorm、ReLU、Pooling 和 Dropout；
- 修改版 ResNet：手写残差块和 shortcut 分支，完成轻量级 CIFAR-10 分类模型。

## 复盘

这组实验的意义在于把课堂公式落到可运行的代码里。相比只调用库函数，手写 Word2Vec 梯度、残差块、Transformer 训练流程能更清楚地理解模型为什么这样设计，也能更快定位训练中出现的维度、数值稳定性和数据处理问题。

