"""
Introduction to Machine Learning

Lab 7: Gaussian mixture model: its application to point cloud alignment

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab7.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): 黄浩博
Student ID(s): 2024201630
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
# don't add any other packages


def squared_distance_matrix(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Construct a N x M distance matrix from a data matrix with size (N, D)
    Each element d_{ij} = ||x_i - y_j ||_2^2

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :return:
        a distance matrix with size (N, M)
    """
    # 矩阵化展开距离公式 ||x-y||^2
    return np.sum(xs**2, axis=1, keepdims=True) + np.sum(ys**2, axis=1) - 2 * xs @ ys.T


def estimate_variance(xs: np.ndarray, ys: np.ndarray, affine: np.ndarray,
                      translation: np.ndarray, responsibility: np.ndarray) -> float:
    """
    Estimate the variance of GMM.
    For simplification, we assume all the Gaussian distributions share the same variance,
    and each feature dimension is independent, so the variance can be represented as a scalar.

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param affine: an affine matrix with size (D, D)
    :param translation: a translation vector with size (1, D)
    :param responsibility: the responsibility matrix with size (N, M)
    :return:
        the variance of each Gaussian distribution, a float
    """
    # 计算均方误差作为方差
    dist_sq = squared_distance_matrix(xs, ys @ affine + translation)
    return np.sum(responsibility * dist_sq) / (xs.shape[0] * xs.shape[1])


def e_step(xs: np.ndarray, ys: np.ndarray, affine: np.ndarray, translation: np.ndarray, variance: float) -> np.ndarray:
    """
    The e-step of the em algorithm, estimating the responsibility P=[p(y_m | x_n)] based on current model

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param affine: an affine matrix with size (D, D)
    :param translation: a translation vector with size (1, D)
    :param variance: a float controlling the variance of each Gaussian component
    :return:
        the responsibility matrix P=[p(y_m | x_n)] with size (N, M),
        which row is the conditional probability of clusters given the n-th sample x_n
    """
    # 计算隐变量后验概率
    dist_sq = squared_distance_matrix(xs, ys @ affine + translation)
    p = np.exp(-dist_sq / (2 * variance))
    return p / np.sum(p, axis=1, keepdims=True)


def m_step(xs: np.ndarray, ys: np.ndarray,
           responsibility: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    the m-step of the em algorithm:

    min_{affine, translation, variance} 1/(2*variance) * sum_{m,n} p(y_m | x_n) ||x_n - affine y_m - translation||_2^2

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param responsibility: the responsibility matrix P=[p(y_m | x_n)] with size (N, M)
    :return:
        an affine matrix with size (D, D)
        a translation vector with size (D, 1)
        the variance of GMM, a float
        the registered point cloud ys_new, with size (M, D)
    """
    N = xs.shape[0]
    w = np.sum(responsibility, axis=0)

    # 求解两组点云的质心
    X_bar = np.mean(xs, axis=0, keepdims=True)
    Y_bar = (w @ ys / N).reshape(1, -1)

    # 去中心化及协方差矩阵计算
    ys_c, xs_c = ys - Y_bar, xs - X_bar
    S_yy = ys_c.T @ (w[:, None] * ys_c)
    S_yx = ys_c.T @ responsibility.T @ xs_c

    # 最小二乘更新参数
    affine = np.linalg.pinv(S_yy) @ S_yx
    translation = X_bar - Y_bar @ affine
    variance = estimate_variance(xs, ys, affine, translation, responsibility)
    ys_new = ys @ affine + translation

    return affine, translation, variance, ys_new


def em_for_alignment(xs: np.ndarray, ys: np.ndarray, num_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    The em algorithm for aligning two point clouds based on affine transformation
    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param num_iter: the number of EM iterations
    :return:
        ys_new: the aligned points: ys_new = ys @ affine + translation
        responsibility: the responsibility matrix P=[p(y_m | x_n)] with size (N, M),
        whose elements indicating the correspondence between the points
    """
    ys_new = np.zeros_like(ys)
    # initialize model parameters:
    responsibility = np.ones((xs.shape[0], ys.shape[0])) / ys.shape[0]
    dim = xs.shape[1]
    affine = np.eye(dim)
    translation = np.zeros((1, dim))

    variance = estimate_variance(xs, ys, affine, translation, responsibility)

    for i in range(num_iter):
        responsibility = e_step(xs, ys, affine, translation, variance)
        affine, translation, variance, ys_new = m_step(xs, ys, responsibility)

        # 实时打印进度条
        progress = (i + 1) / num_iter
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '█' * filled + '-' * (bar_len - filled)
        print(f'\rEM Iteration [{bar}] {i + 1}/{num_iter}', end='', flush=True)

    print()
    return ys_new, responsibility


if __name__ == '__main__':
    # 全局参数配置
    MAT_FILE = 'fish.mat'
    NUM_ITERATIONS = 100
    OUTPUT_RESULT = 'result.png'
    OUTPUT_CORR = 'correspondence.png'

    print(f"Loading data from {MAT_FILE} ...")
    fish = loadmat(MAT_FILE)
    xs = fish['X']
    ys = fish['Y']
    print(f"Target points: {xs.shape[0]}, Source points: {ys.shape[0]}")

    ys_new, prob = em_for_alignment(xs, ys, num_iter=NUM_ITERATIONS)

    print(f"Saving alignment result to {OUTPUT_RESULT} ...")
    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1], label='target')
    plt.scatter(ys[:, 0], ys[:, 1], label='source')
    plt.scatter(ys_new[:, 0], ys_new[:, 1], label='aligned source')
    plt.legend()
    plt.savefig(OUTPUT_RESULT)
    plt.close()

    print(f"Saving correspondence map to {OUTPUT_CORR} ...")
    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.scatter(ys[:, 0], ys[:, 1])
    idx = np.argmax(prob, axis=1)
    for n in range(xs.shape[0]):
        plt.plot([xs[n, 0], ys[idx[n], 0]], [xs[n, 1], ys[idx[n], 1]], 'k-')
    plt.savefig(OUTPUT_CORR)
    plt.close()

    print("All tasks completed.")