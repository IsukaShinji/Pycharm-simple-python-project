"""
Introduction to Machine Learning

Lab 5: Matrix factorization and linear dimensionality reduction

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
# don't add any other packages


# data simulator and testing function (Don't change them)
def zero_mean_point_cloud_simulator(n_pts: int = 50,
                                    r_seed: int = 42) -> dict:
    """
    Simulate a set of zero-mean 2D points with Gaussian noise or outliers
    :param n_pts: the number of 2D points
    :param r_seed: the random seed
    :return:
        a dictionary containing the points with Gaussian noise and those with outliers, respectively
    """
    x = 4 * (np.random.RandomState(r_seed).rand(n_pts, 1) - 0.5)
    y = 0.4 * x
    data = np.concatenate((x, y), axis=1)
    pts1 = data + 0.1 * np.random.RandomState(r_seed).randn(n_pts, 2)
    pts2 = data + 0.01 * np.random.RandomState(r_seed).randn(n_pts, 2)
    idx = np.random.RandomState(r_seed).permutation(n_pts)
    n_noise = int(0.2 * n_pts)
    pts2[idx[:n_noise], :] = np.random.RandomState(r_seed).randn(n_noise, 2) + np.array([0.5, 1.5]).reshape((1, 2))
    return {'gauss': pts1, 'outlier': pts2}


def visualization_pts(pts: np.ndarray, label: str, point_type: str):
    plt.plot(pts[:, 0], pts[:, 1], point_type, label=label)


def visualization_line(v: np.ndarray, label: str, line_type: str):
    xs = 5 * (np.arange(0, 100) / 100 - 0.5)
    ys = v[1] / v[0] * xs
    plt.plot(xs, ys, line_type, label=label)


# Task 1: Implement PCA via eigen-decomposition
def pca(xs: np.ndarray, n_pc: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement PCA via eigen-decomposition
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param n_pc: the number of principal components we would like to output
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # Step 1: Zero-mean normalization for input data
    n_samples, n_dim = xs.shape
    mean_vec = np.mean(xs, axis=0, keepdims=True)
    xs_zero_mean = xs - mean_vec  # 数据中心化，消除均值影响

    # Step 2: Compute covariance matrix (unbiased estimation)
    cov_matrix = xs_zero_mean.T @ xs_zero_mean / (n_samples - 1)  # 计算协方差矩阵，捕捉特征间相关性

    # Step 3: Eigen-decomposition for covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # 特征值分解，找到主方向
    # Eliminate numerical imaginary parts (covariance matrix is real symmetric)
    eigenvalues = np.real(eigenvalues)  # 去除数值计算带来的虚部
    eigenvectors = np.real(eigenvectors)

    # Step 4: Sort eigenvalues in descending order and select top n_pc components
    sorted_idx = np.argsort(eigenvalues)[::-1]  # 按特征值从大到小排序
    top_idx = sorted_idx[:n_pc]  # 选取前n_pc个主成分
    top_eigenvalues = eigenvalues[top_idx]
    top_eigenvectors = eigenvectors[:, top_idx]

    # Step 5: Project data to PC space and reconstruct
    projection = xs_zero_mean @ top_eigenvectors  # 将数据投影到低维空间
    x_recon = projection @ top_eigenvectors.T  # 从低维投影重构回原始空间

    return top_eigenvectors, top_eigenvalues, x_recon, xs_zero_mean


# Task 2: Implement data whitening via the method in Lecture 2 and the PCA-based method in Lecture 5
def data_whitening(xs: np.ndarray) -> np.ndarray:
    """
    Implement data whitening via the method in Lecture 2 or PCA
    :param xs: the data matrix with size (N, D), N is the number of samples
    :return:
        ys: the data yield normal distribution, with size (N, D)
    """
    # Step 1: Perform PCA with all dimensions to get full components
    n_dim = xs.shape[1]
    pc_matrix, eigenvalues, _, xs_zero_mean = pca(xs, n_pc=n_dim)  # 先做全维度PCA

    # Step 2: PCA whitening: normalize projected data by sqrt of eigenvalues
    epsilon = 1e-8  # Avoid division by zero  # 防止除零的小常数
    projection = xs_zero_mean @ pc_matrix  # 投影到PCA空间
    xs_whitened = projection / np.sqrt(eigenvalues + epsilon)  # 按特征值缩放，实现白化

    return xs_whitened


# Task 3: Try to develop your own method to achieve robust PCA (the method may not be the state-of-the-art, but doable)
def hard_thresholding(x: np.ndarray, ratio: float) -> np.ndarray:
    """
    The hard-thresholding operator
    :param x: input array with arbitrary size
    :param ratio: the ratio of nonzero elements
    :return:
        y = x,  if |x| > a threshold
            0,  otherwise
    """
    total_elem = x.size
    keep_num = int(np.clip(ratio * total_elem, 0, total_elem))  # 计算要保留的非零元素数量

    # Edge cases handling
    if keep_num == 0:
        return np.zeros_like(x)  # 全置零
    if keep_num >= total_elem:
        return x.copy()  # 保留全部

    # Calculate adaptive threshold based on percentile
    abs_x = np.abs(x)
    threshold = np.percentile(abs_x, 100 * (1 - ratio))  # 根据比例计算自适应阈值

    # Apply hard thresholding
    y = x * (abs_x >= threshold)  # 只保留绝对值大于阈值的元素

    return y


def robust_pca_hard(xs: np.ndarray, n_pc: int = 2, n_alt: int = 100,
                    ratio_nz: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement your own algorithm to solve the robust PCA problem via
    optimizing the low-rank factorization of data matrix (X in R^{N x D}) explicitly, i.e.,
    min_{L, S} ||X - (L + S)||_F^2
    s.t. rank(L) <= n_pc, ||S||_0 < ratio_nz * (N * D)
    Hint: you may want to solve L and S in an alternating optimization manner:
    1) Fix L and solve
        L = argmin_L ||X - (L + S)||_F^2
        s.t. rank(L) <= n_pc
    2) Fix S and solve
        S = argmin_S ||X - (L + S)||_F^2,
        s.t.. ||S||_0 < ratio_nz * (N * D)
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space.
    :param n_pc: the number of principal components we would like to output.
    :param n_alt: the number of steps for alternating optimization.
    :param ratio_nz: the ratio of non-zero elements in the whole matrix.
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # Step 1: Zero-mean normalization
    n_samples, n_dim = xs.shape
    mean_vec = np.mean(xs, axis=0, keepdims=True)
    xs_zero_mean = xs - mean_vec  # 数据中心化

    # Step 2: Initialize sparse matrix S
    S = np.zeros_like(xs_zero_mean)  # 初始化稀疏矩阵为全零

    # Step 3: Alternating optimization for L and S
    for _ in range(n_alt):  # 交替优化迭代
        # Subproblem 1: Fix S, solve low-rank L via PCA truncation
        residual = xs_zero_mean - S  # 减去当前稀疏部分
        pc_matrix, eigenvalues, L, _ = pca(residual, n_pc=n_pc)  # PCA截断得到低秩部分L

        # Subproblem 2: Fix L, solve sparse S via hard thresholding
        residual = xs_zero_mean - L  # 减去当前低秩部分
        S = hard_thresholding(residual, ratio_nz)  # 硬阈值得到稀疏部分S

    # Step 4: Extract final PCA results from optimized low-rank matrix L
    final_pc, final_eigenvalues, _, _ = pca(L, n_pc=n_pc)  # 从最终的L中提取主成分

    return final_pc, final_eigenvalues, L, xs_zero_mean


# Task 4: Suppose that you are a data attacker. Because of limited budgets, you can only add two outliers
# Try to design a "data poisoning" strategy to change the covariance of the data as much as possible.
def coupled_outlier_poisoning(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two outliers "x1" and "x2", with constraints ||x1||_2 = ||x2||_2 = 1 and x1 + x2 = 0
    :param xs: a data matrix with size (N, D), N is the number of samples
    :return:
        the outliers with size (2, D)
        the new data matrix with the outlier, with size (N+2, D)
    """
    n_samples, n_dim = xs.shape

    # Step 1: Compute covariance matrix of original zero-mean data
    cov_matrix = xs.T @ xs / (n_samples - 1)  # 计算原始数据的协方差矩阵

    # Step 2: Eigen-decomposition to find minimum eigenvalue and corresponding eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # 特征值分解
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Optimal x1: unit eigenvector of the minimum eigenvalue (maximizes covariance difference)
    min_eig_idx = np.argmin(eigenvalues)  # 找到最小特征值对应的方向（方差最小方向）
    x1 = eigenvectors[:, min_eig_idx]
    x1 = x1 / np.linalg.norm(x1, ord=2)  # Ensure unit L2 norm  # 约束为单位范数
    x2 = -x1  # Satisfy zero-mean constraint for new dataset  # 对称点，保证新数据均值仍为0

    # Step 3: Construct outliers and new data matrix
    outliers = np.vstack((x1, x2))  # 拼接两个离群点
    data_new = np.vstack((xs, outliers))  # 加入原始数据

    return outliers, data_new


# Task 5: implement the NMF algorithm
def nonnegative_matrix_factorization(xs: np.ndarray,
                                     rank: int,
                                     num_iter: int = 100,
                                     seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the nonnegative matrix factorization
    min_{U, V} ||X - UV^T||_F^2
    s.t. U in [0, inf]^{(N, r)} and V in [0, inf]^{(D, r)}
    :param xs: a data matrix with size (N, D), N is the number of samples
    :param rank: the rank of U and V
    :param num_iter: the number of iterations
    :param seed: the random seed of initialization
    :return:
        U in [0, inf]^{(N, r)}
        V in [0, inf]^{(D, r)}
        hat{X} = UV^T
    """
    # Non-negative random initialization
    us = np.random.RandomState(seed=seed).rand(xs.shape[0], rank)  # 随机初始化U，保证非负
    vs = np.random.RandomState(seed=seed + 2).rand(xs.shape[1], rank)  # 随机初始化V，保证非负

    epsilon = 1e-8  # Numerical stability for division  # 防止除零
    # Lee & Seung multiplicative update rule
    for _ in range(num_iter):  # 迭代更新
        # Update U matrix
        xvt = xs @ vs  # 计算分子项
        uvvt = us @ vs.T @ vs  # 计算分母项
        us = us * (xvt / (uvvt + epsilon))  # 乘法更新U，保持非负

        # Update V matrix
        xtu = xs.T @ us  # 计算分子项
        vutu = vs @ us.T @ us  # 计算分母项
        vs = vs * (xtu / (vutu + epsilon))  # 乘法更新V，保持非负

    # Compute final approximated matrix
    x_hat = us @ vs.T  # 重构数据矩阵

    return us, vs, x_hat


# Testing script
if __name__ == '__main__':
    data = zero_mean_point_cloud_simulator()
    for noise_type in data.keys():
        vs1, lambdas1, xhat1, xs1 = pca(data[noise_type], n_pc=1)
        vs2, lambdas2, xhat2, _ = robust_pca_hard(data[noise_type], n_pc=1, ratio_nz=0.1)
        xhat3 = data_whitening(data[noise_type])

        plt.figure()
        visualization_pts(xs1, label='data points', point_type='g.')
        visualization_pts(xhat1, label='pca', point_type='rx')
        visualization_pts(xhat2, label='rpca', point_type='bx')
        visualization_line(v=vs1, label='pca v1', line_type='r:')
        visualization_line(v=vs2, label='rpca v1', line_type='b:')
        visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
        result = 'PCA vs RPCA: {} noise'.format(noise_type)
        plt.title(result)
        plt.legend()
        plt.savefig('result_{}.png'.format(noise_type))
        plt.close('all')

        plt.figure()
        visualization_pts(data[noise_type], label='before whitening', point_type='g.')
        visualization_pts(xhat3, label='after whitening', point_type='rx')
        plt.legend()
        plt.axis('equal')
        plt.savefig('whitening_{}.png'.format(noise_type))
        plt.close('all')

    vs1, lambdas1, xhat1, xs1 = pca(data['gauss'], n_pc=1)
    outliers, data_noisy = coupled_outlier_poisoning(data['gauss'])
    print(data['gauss'].shape, data_noisy.shape)
    vs2, lambdas2, xhat2, _ = pca(data_noisy, n_pc=1)
    plt.figure()
    visualization_pts(data['gauss'], label='data points', point_type='g.')
    visualization_pts(outliers, label='outlier', point_type='k*')
    visualization_pts(xhat1, label='PCA before poisoning', point_type='rx')
    visualization_pts(xhat2, label='PCA after poisoning', point_type='bx')
    visualization_line(v=vs1, label='v1 before poisoning', line_type='r:')
    visualization_line(v=vs2, label='v1 after poisoning', line_type='b:')
    visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
    result = 'Covariance poisoning'
    plt.title(result)
    plt.legend()
    plt.axis('equal')
    plt.savefig('poisoning_pca.png')
    plt.close('all')

    data_mat = np.random.RandomState(seed=42).rand(100, 50)
    for r in [5, 10, 20, 30, 40]:
        u_mat, v_mat, data_approx = nonnegative_matrix_factorization(xs=data_mat, rank=r, num_iter=100, seed=1)
        error = np.sum(np.abs(data_mat - data_approx)) / np.sum(data_mat)
        print('Rank-{} NMF approximation RMAE={}'.format(r, error))