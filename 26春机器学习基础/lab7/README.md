# Lab 7 EM 点集配准

本目录实现基于 EM 的二维点集仿射配准实验。

## 内容

- `script_lab7.py`：读取 `fish.mat`，估计点集对应关系、仿射矩阵、平移向量和方差。
- `fish.mat`：实验数据。
- `result.png`：目标点、源点和配准后源点的可视化。
- `correspondence.png`：估计对应关系图。

## 运行

```bash
python script_lab7.py
```

脚本依赖 `numpy`、`matplotlib` 和 `scipy.io.loadmat`。
