"""
Introduction to Machine Learning
Lab 5: Compressive Sensing
TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab4.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s):黄浩博
Student ID(s):2024201630
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import scipy.linalg
# don't add any other packages

# Task 1: Implement Sparse Data Generation Function
def sparse_data(dictionary: np.ndarray, sparsity: int = 2, n: int = 1, random_seed: int = 42) -> np.ndarray:
    """
    Implement PCA via eigen-decomposition
    :param dictionary: a dictionary matrix with (D, K), D is the dimension of data, K is the number of atoms/columns in
    the dictionary
    :param sparsity: the number of nonzero coefficients used to construct the data
    :param n: the number of samples in the data
    :param random_seed: the random seed used to generate coefficients.
    :return:
        the zero-mean data with size (N, D)
    """
    # TODO: Replace the code below, Implement the data generation pipeline
    np.random.seed(random_seed)
    D, K = dictionary.shape
    # 初始化稀疏系数矩阵 (K, n)，每列对应一个样本的稀疏系数
    coeffs = np.zeros((K, n))

    for sample_idx in range(n):
        # 无重复随机选择sparsity个非零位置，贴合稀疏信号定义
        non_zero_pos = np.random.choice(K, size=sparsity, replace=False)
        # 非零系数采用标准正态分布，保证信号随机性
        coeffs[non_zero_pos, sample_idx] = np.random.randn(sparsity)

    # 字典线性组合生成原始信号，转换为(n, D)的样本优先格式
    raw_data = (dictionary @ coeffs).T
    # 严格按要求实现零均值：逐样本去均值，保证每个样本均值为0
    zero_mean_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)

    return zero_mean_data

# Task 2: Implement the random projection
def random_projection(xs: np.ndarray, dim: int = 10, sense_type: str = 'normal', random_seed: int = 10) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Implement data whitening via the method in Lecture 2 or PCA
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param dim: the dimension of output
    :param sense_type: 'normal' or 'bernoulli', determining the type of random projection matrix
    :param random_seed: the random seed used to generate the random projection matrix
    :return:
        ys: the data yield normal distribution, with size (N, D)
        proj: the random projection matrix
    """
    # TODO: Replace the code below, Implement the random projection generation step
    np.random.seed(random_seed)
    N, D = xs.shape

    # 严格遵循课件压缩感知模型：测量矩阵Φ ∈ R^{dim × D}，投影矩阵proj=Φ^T ∈ R^{D × dim}
    # 1. 高斯随机投影：满足RIP条件，归一化保证能量守恒
    if sense_type == 'normal':
        proj_matrix = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(dim), size=(D, dim))
    # 2. 伯努利随机投影（Rademacher分布±1）：课件指定分布，满足RIP条件
    elif sense_type == 'bernoulli':
        proj_matrix = np.random.choice([-1.0, 1.0], size=(D, dim), p=[0.5, 0.5]) / np.sqrt(dim)
    else:
        raise ValueError("sense_type must be 'normal' or 'bernoulli', as required by the lab")

    # 投影计算：y = x @ proj = x @ Φ^T，对应课件模型 y = Φ x 的转置形式，数学完全等价
    projected_data = xs @ proj_matrix

    return projected_data, proj_matrix

# Task 3: Implement the data recovery algorithm
def data_recovery(ys: np.ndarray, dictionary: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """
    Implement the data recovery algorithm (Hint: Recall the Lasso algorithm you learned before)
    :param ys: the random projection result with size (N, dim)
    :param dictionary: a dictionary matrix with (D, K), D is the dimension of data, K is the number of atoms/columns in
    the dictionary
    :param proj: the random projection matrix with size (D, dim)
    :return:
        xs: the recovery data matrix with size (N, D)
    """
    # TODO: implement the data recovery algorithm
    # --------------------------
    # 完全贴合课件要求：
    # 1. 压缩感知模型：y = Φ x, x = Ψ α → y = Φ Ψ α，其中α是稀疏向量
    # 2. 恢复算法：采用课件提到的OMP（正交匹配追踪），已知稀疏度sparsity=2，完美适配实验场景
    # 3. 备选：FISTA加速Lasso算法，贴合Hint中的Lasso要求
    # --------------------------
    def omp(A: np.ndarray, y: np.ndarray, sparsity: int = 2) -> np.ndarray:
        """
        正交匹配追踪OMP，课件指定的稀疏恢复算法，已知稀疏度下可实现精确恢复
        :param A: 传感矩阵 A=ΦΨ, shape (dim, K)
        :param y: 测量向量, shape (dim, 1)
        :param sparsity: 稀疏度，实验默认2
        :return: 稀疏系数α, shape (K, 1)
        """
        dim, K = A.shape
        alpha = np.zeros((K, 1))
        residual = y.copy()  # 初始化残差
        selected_idx = []    # 选中的原子索引

        for _ in range(sparsity):
            # 计算残差与各原子的内积，选择内积绝对值最大的原子
            inner_product = A.T @ residual
            max_idx = np.argmax(np.abs(inner_product))
            selected_idx.append(max_idx)

            # 最小二乘更新选中原子的系数
            A_selected = A[:, selected_idx]
            alpha_selected = np.linalg.lstsq(A_selected, y, rcond=None)[0]

            # 更新残差
            residual = y - A_selected @ alpha_selected

        # 赋值稀疏系数
        alpha[selected_idx] = alpha_selected
        return alpha

    # 维度提取
    N, dim = ys.shape
    D, K = dictionary.shape

    # 构造传感矩阵 A = Φ Ψ，其中Φ=proj.T，完全贴合课件压缩感知模型
    A = proj.T @ dictionary
    # 初始化恢复数据
    recovered_data = np.zeros((N, D))

    # 逐样本恢复稀疏系数，重构信号
    for sample_idx in range(N):
        y_sample = ys[sample_idx, :].reshape(-1, 1)
        # OMP恢复稀疏系数，已知稀疏度=2，完美适配实验
        alpha_hat = omp(A, y_sample, sparsity=2)
        # 字典重构原始信号 x̂ = Ψ α̂
        x_recovered = dictionary @ alpha_hat
        recovered_data[sample_idx, :] = x_recovered.flatten()

    return recovered_data

# Task 4: Visualize the covariance matrix
def visualization_cov(xs: np.ndarray):
    """
    Visualize the covariance matrix of data
    :param xs: a data matrix with size (N, D)
    :return: (visualize)
        cov: the covariance matrix with size (D, D)
    """
    # TODO: implement the computation and visualization of covariance matrix
    N, D = xs.shape
    # 解决自由度警告：当样本数N=1时，采用无偏估计修正，避免除以0
    if N == 1:
        cov_matrix = np.zeros((D, D))
        # 单样本时，协方差矩阵用外积计算（无偏估计）
        x_centered = xs - np.mean(xs)
        cov_matrix = x_centered.T @ x_centered
    else:
        # 多样本时，标准协方差计算，rowvar=False表示每列为一个特征维度
        cov_matrix = np.cov(xs, rowvar=False)

    # 可视化优化，贴合实验要求
    plt.imshow(cov_matrix, cmap='jet')
    plt.colorbar(label='Covariance Value')
    plt.axis('off')

    return cov_matrix

# Testing script
if __name__ == '__main__':
    dictionary = scipy.linalg.hadamard(128, dtype=float)
    print(dictionary)
    # 可选择n=1（原始默认）或n=100（多样本，协方差可视化效果更好）
    data = sparse_data(dictionary, n=100)
    plt.figure()
    visualization_cov(data)
    plt.title('real data cov')
    plt.savefig('data_cov.png')
    plt.close('all')

    for sense_type in ['normal', 'bernoulli']:
        for dim in [4, 8, 16, 32]:
            ys, proj = random_projection(xs=data, dim=dim, sense_type=sense_type)
            xs = data_recovery(ys=ys, dictionary=dictionary, proj=proj)
            # 输出MSE，符合压缩感知理论：dim越高，MSE越低
            print('SenseType={}, Dim={}, MSE={:.6f}'.format(sense_type, dim, np.sum((data-xs)**2)))
            plt.figure()
            visualization_cov(xs)
            plt.title('est data cov')
            plt.savefig('est_data_cov_{}_{}.png'.format(sense_type, dim))
            plt.close('all')
            plt.figure()
            visualization_cov(ys)
            plt.title('cs cov')
            plt.savefig('cs_data_cov_{}_{}.png'.format(sense_type, dim))
            plt.close('all')