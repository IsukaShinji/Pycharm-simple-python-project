# Lab 3 核回归

本目录实现非参数核回归和 Kernel Ridge Regression。

## 内容

- `script_lab3.py`：实现 RBF、gate、triangle、linear 等核函数。
- 包含 Nadaraya-Watson estimator、KRR 闭式解、KRR-SGD 和 block coordinate descent。
- `result_*.png`：不同核函数和带宽配置下的拟合结果。

## 运行

```bash
python script_lab3.py
```

运行后会生成多张结果图，用于比较核函数、带宽和训练方法对曲线拟合的影响。
