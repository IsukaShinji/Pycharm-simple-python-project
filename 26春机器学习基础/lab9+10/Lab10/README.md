# Lab 10 Primal SVM 与公平性

本目录实现 Adult 数据集上的 primal SVM，并尝试通过数据增强缓解性别群体间预测差异。

## 内容

- `script_lab10.py`：实现 hinge loss、线性模型梯度、SGD primal SVM、测试函数和数据增强。
- `adult.csv`：实验数据。
- `Lab10.pdf`：课程实验说明。

## 运行

```bash
python script_lab10.py
```

输出包括原始 SVM 和数据增强后 SVM 的准确率，以及 male/female 群体预测高收入比例。
