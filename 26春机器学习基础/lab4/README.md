# Lab 4 PCA、RPCA 与矩阵分解

本目录围绕 PCA 及其变体展开实验。

## 内容

- `script_lab4.py`：实现 PCA、PCA whitening、hard-thresholding 形式的 Robust PCA、协方差投毒样本生成和 NMF。
- `result_gauss.png`、`result_outlier.png`：PCA/RPCA 在不同噪声下的比较。
- `whitening_gauss.png`、`whitening_outlier.png`：白化前后对比。
- `poisoning_pca.png`：加入两个约束 outlier 后对 PCA 方向的影响。

## 运行

```bash
python script_lab4.py
```

重点关注协方差矩阵、特征分解、白化缩放和低秩/稀疏分解的关系。
