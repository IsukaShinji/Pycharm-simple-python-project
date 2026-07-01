# Lab 5 压缩感知

本目录实现稀疏数据生成、随机投影和数据恢复实验。

## 内容

- `script_lab5.py`：实现 sparse data generation、normal/Bernoulli random projection、基于稀疏系数的 recovery 和协方差可视化。
- `data_cov.png`：原始数据协方差。
- `est_data_cov_*.png`：恢复数据协方差。
- `cs_data_cov_*.png`：压缩观测协方差。

## 运行

```bash
python script_lab5.py
```

实验用于观察投影维度和随机投影分布对压缩感知恢复质量的影响。
