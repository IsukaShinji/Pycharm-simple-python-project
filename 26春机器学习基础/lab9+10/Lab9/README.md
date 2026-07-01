# Lab 9 LDA 与 Logistic Regression

本目录实现 Adult 数据集上的二分类实验。

## 内容

- `script_lab9.py`：读取 `adult.csv`，处理类别特征，训练 Logistic Regression 和 LDA。
- `adult.csv`：实验数据。
- `Lab9.pdf`：课程实验说明。

## 关注点

- Logistic Regression 的 sigmoid、BCE loss 和 SGD 更新。
- LDA 的类均值、协方差和判别阈值。
- `gender_fairness_check` 中不同性别群体预测为高收入的比例差异。

## 运行

```bash
python script_lab9.py
```
