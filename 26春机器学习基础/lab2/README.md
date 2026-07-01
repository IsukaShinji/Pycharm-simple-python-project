# Lab 2 稀疏线性模型

本目录实现线性模型中的稀疏学习和鲁棒回归实验。

## 内容

- `script_lab2.py`：包含 OMP、Ridge SGD、Lasso、Elastic Net 和 IRLS。
- 数据由脚本内部模拟，可切换高斯/Laplace 先验和噪声。
- 输出主要是不同方法在测试集上的 MSE。

## 运行

```bash
python script_lab2.py
```

脚本只依赖 `numpy`，重点阅读各个训练函数的目标函数和权重更新方式。
