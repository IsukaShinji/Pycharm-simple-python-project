# Lab 6 流形学习与非线性降维

本目录实现三类降维方法：Kernel PCA、Locally Linear Embedding 和 Laplacian Eigenmaps。

## 内容

- `script_lab6.py`：模拟 3D manifold 数据，实现距离矩阵、核矩阵、KNN 图、LLE 和 Laplacian Eigenmaps。
- `data_3d.png`、`data_2d.png`：原始数据可视化。
- `KPCA_*.png`：不同核和带宽的 Kernel PCA 结果。
- `LLE_*.png`：不同邻居数的 LLE 结果。
- `LE_*.png`：不同图连接和归一化设置下的 Laplacian Eigenmaps 结果。

## 运行

```bash
python script_lab6.py
```

重点比较邻居数、核带宽、是否使用 normalized Laplacian 对低维嵌入结构的影响。
