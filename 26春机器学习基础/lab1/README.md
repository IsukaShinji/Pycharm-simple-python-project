# Lab 1 多项式回归

本目录实现多项式回归基础实验，比较闭式解、SGD 和归一化后的 SGD。

## 内容

- `script_lab1.py`：模拟多项式数据，实现闭式训练、SGD 训练、L2/L1/Linf 等归一化和测试误差计算。
- `Order3.png`、`Order5.png`、`Order7.png`、`Order9.png`：不同多项式阶数下的拟合结果图。

## 运行

```bash
python script_lab1.py
```

脚本只依赖 `numpy` 和 `matplotlib`，运行后会在当前目录生成对应阶数的拟合图。
