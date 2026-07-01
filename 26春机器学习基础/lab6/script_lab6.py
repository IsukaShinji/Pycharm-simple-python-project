"""
Introduction to Machine Learning

Lab 6: Nonlinear dimensionality reduction

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab6.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s):黄浩博
Student ID(s):2024201630
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
# don't add any other packages


# data simulator and testing function (Don't change them)
def simulate_3d_manifold(n_pts: int = 500, noise_level: float = 0.01, r_seed: int = 42) -> dict:
    """
    Simulate a set of 3D points lying on a manifold, the manifold is a 2D geometry embedded in the 3D space.
    :param n_pts: the number of 3D points
    :param r_seed: the random seed
    :param noise_level: the standard deviation of Gaussian noise
    :return:
        a dictionary containing the 3D points with Gaussian noise and their 2D latent codes.
    """

    t1 = 5 * np.pi / 3 * np.random.RandomState(r_seed).rand(n_pts, 1)
    t2 = 5 * np.pi / 3 * np.random.RandomState(1).rand(n_pts, 1)
    latent_code = np.concatenate((t1, t2), axis=1)
    x1 = 3 + np.cos(t1) * np.cos(t2)
    x2 = 3 + np.cos(t1) * np.sin(t2)
    x3 = np.sin(t1)
    data = np.concatenate((x1, x2, x3), axis=1) + noise_level * np.random.RandomState(r_seed).randn(n_pts, 3)
    return {'3d': data, '2d': latent_code}


def visualization_3d_pts(pts3d: np.ndarray, prefix: str = 'data'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    plt.savefig('{}_3d.png'.format(prefix))
    plt.close()


def visualization_2d_pts(pts2d: np.ndarray, prefix: str = 'data'):
    plt.figure(figsize=(12, 12))
    plt.scatter(pts2d[:, 0], pts2d[:, 1])
    plt.savefig('{}_2d.png'.format(prefix))
    plt.close()


# Task 1: Implement Kernel PCA
def distance_matrix(xs: np.ndarray, distance_type: str = 'L2') -> np.ndarray:
    """
    Construct a N x N distance matrix from a data matrix with size (N, D)
    :param xs: a data matrix with size (N, D)
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        a distance matrix with size (N, N)
    """
    N = xs.shape[0]
    dist = np.zeros((N, N))
    if distance_type == 'L2':
        for i in range(N):
            diff = xs[i] - xs
            dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    elif distance_type == 'L1':
        for i in range(N):
            diff = xs[i] - xs
            dist[i] = np.sum(np.abs(diff), axis=1)
    else:
        raise ValueError("distance_type must be 'L2' or 'L1'")
    return dist


def kernel(x: np.ndarray, k_type: str = 'rbf', bandwidth: float = 1) -> np.ndarray:
    """
    Implement typical kernel functions
    1) RBF kernel: k(x, y) = exp(-||x - y||_2^2 / bandwidth)
    2) Linear kernel: k(x, y) = <x, y>

    Hint: Recall your Lab work 4

    :param x: a set of samples with size (N, D), where N is the number of samples, D is the dimension of features
    :param k_type: the type of kernels, including 'rbf', 'linear'
    :param bandwidth: the hyperparameter controlling the width of rbf kernels
    :return:
        return a matrix with size (M, N)
    """
    N = x.shape[0]
    if k_type == 'linear':
        K = x @ x.T
    elif k_type == 'rbf':
        # Compute squared Euclidean distance matrix
        # Use the identity: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
        sq_norms = np.sum(x ** 2, axis=1).reshape(-1, 1)
        dist_sq = sq_norms + sq_norms.T - 2 * (x @ x.T)
        # Avoid negative zeros due to numerical errors
        dist_sq = np.maximum(dist_sq, 0)
        K = np.exp(-dist_sq / bandwidth)
    else:
        raise ValueError("k_type must be 'linear' or 'rbf'")
    return K


def kernel_pca(xs: np.ndarray, d: int, k_type: str = 'rbf', bandwidth: float = 1) -> np.ndarray:
    """
    Implement kernel PCA
    :param xs: the data matrix with shape (N, D)
    :param d: the number of dimensions after dimensionality reduction
    :param k_type: the type of kernels, including 'rbf', 'linear'
    :param bandwidth: the hyperparameter controlling the width of rbf kernels
    :return:
    """
    N = xs.shape[0]
    # Compute kernel matrix
    K = kernel(xs, k_type, bandwidth)
    # Centering the kernel matrix
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    # Eigen decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top d components, ignore negative eigenvalues
    pos_mask = eigvals > 0
    eigvals = eigvals[pos_mask]
    eigvecs = eigvecs[:, pos_mask]
    d = min(d, len(eigvals))
    eigvals = eigvals[:d]
    eigvecs = eigvecs[:, :d]
    # Projection: scaled eigenvectors
    # In kernel PCA, the transformed data is eigvecs * sqrt(eigvals)
    # Each column corresponds to a principal component
    transformed = eigvecs * np.sqrt(eigvals)
    return transformed


# Task 2: Construct a K-NN graph from data points
def construct_knn_graph(xs: np.ndarray, k: int = 5, distance_type: str = 'L2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a K-NN graph from the data points and output the adjacency matrix and the index matrix
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param k: the number of principal components we would like to output
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        an adjacency matrix with size (N, N)
        an index matrix with size (N, k), the n-th row contains the indices of the neighbors of the n-th sample.
    """
    N = xs.shape[0]
    # Compute distance matrix
    D = distance_matrix(xs, distance_type)
    # Set diagonal to inf to avoid self-neighbor
    np.fill_diagonal(D, np.inf)
    # For each point, find k smallest distances
    indices = np.zeros((N, k), dtype=int)
    for i in range(N):
        # Get the indices of the k nearest neighbors (excluding self)
        indices[i] = np.argpartition(D[i], k)[:k]
    # Build symmetric adjacency matrix (unweighted, 0/1)
    adj = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in indices[i]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj, indices


# Task 2: Implement the Locally Linear Embedding algorithm
def locally_linear_embedding(xs: np.ndarray, k: int = 5, dim: int = 2, distance_type: str = 'L2') -> np.ndarray:
    """
    Implement the locally linear embedding algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph
    :param dim: the dimension of latent code, where dim < D
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    N, D = xs.shape
    # Step 1: construct K-NN graph and get neighbor indices (N x k)
    _, indices = construct_knn_graph(xs, k, distance_type)
    # Step 2: compute reconstruction weights W (N x N sparse)
    W = np.zeros((N, N))
    for i in range(N):
        # Neighbors of i
        neigh_idx = indices[i]
        Xi = xs[neigh_idx]  # shape (k, D)
        xi = xs[i]          # shape (D,)
        # Center the neighbors around xi
        Z = Xi - xi         # shape (k, D)
        # Compute local covariance matrix C = Z Z^T (k x k)
        C = Z @ Z.T
        # Regularization to handle singular matrices
        trace = np.trace(C)
        if trace > 0:
            C += 1e-3 * trace * np.eye(k)
        # Solve for weights: C * w = 1_k, then normalize to sum to 1
        ones = np.ones(k)
        try:
            w = np.linalg.solve(C, ones)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            w = np.linalg.pinv(C) @ ones
        w = w / np.sum(w)
        W[i, neigh_idx] = w
    # Step 3: build cost matrix M = (I - W)^T (I - W)
    IW = np.eye(N) - W
    M = IW.T @ IW
    # Eigen decomposition of M
    eigvals, eigvecs = np.linalg.eigh(M)
    # The embedding is given by the eigenvectors corresponding to the smallest eigenvalues,
    # excluding the zero eigenvalue (the constant vector)
    # Usually we take the eigenvectors with the smallest dim+1 eigenvalues, then discard the smallest
    # (the one with eigenvalue ~0)
    idx_sorted = np.argsort(eigvals)
    # Use dim smallest non-zero eigenvalues
    embed = eigvecs[:, idx_sorted[1:dim+1]]
    return embed


# Task 3: Implement the Laplacian eigenmap algorithm
def laplacian_eigenmaps(xs: np.ndarray, k: int = None, dim: int = 2,
                        normalize: bool = True, bandwidth: float = 4) -> np.ndarray:
    """
    Implement the Laplacian Eigenmap algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph, if k is None, we obtain a fully-connected graph
    :param dim: the dimension of latent code, where dim < D
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :param normalize: use normalized Laplacian or not
    :param bandwidth: the bandwidth of kernel for computing the similarity matrix
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    N = xs.shape[0]
    # Step 1: Build similarity matrix W
    if k is None:
        # Fully connected graph: use RBF kernel on all pairs
        W = kernel(xs, k_type='rbf', bandwidth=bandwidth)
    else:
        # Sparse graph: use K-NN with RBF weights
        D_mat = distance_matrix(xs, distance_type='L2')
        W = np.zeros((N, N))
        for i in range(N):
            # Find k nearest neighbors (excluding self)
            dist_i = D_mat[i]
            dist_i[i] = np.inf
            neighbors = np.argpartition(dist_i, k)[:k]
            # Compute RBF similarities
            for j in neighbors:
                sim = np.exp(- (dist_i[j] ** 2) / bandwidth)
                W[i, j] = sim
                W[j, i] = sim
    # Step 2: Degree matrix D (diagonal)
    deg = np.sum(W, axis=1)
    # Avoid zero degrees (isolated points)
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    # Step 3: Laplacian
    L = np.diag(deg) - W
    if normalize:
        # Symmetric normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        # Compute eigenvectors of L_sym (smallest eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Sort ascending
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        # The first eigenvector (eigval=0) is constant, skip it
        embed = eigvecs[:, 1:dim+1]
    else:
        # Unnormalized Laplacian: solve generalized eigenproblem L v = λ D v
        # Convert to symmetric problem: L_sym = D^{-1/2} L D^{-1/2} with eigenvectors v_sym,
        # then v = D^{-1/2} v_sym
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        eigvals, eigvecs_sym = np.linalg.eigh(L_sym)
        idx = np.argsort(eigvals)
        eigvecs_sym = eigvecs_sym[:, idx]
        # Recover generalized eigenvectors v = D^{-1/2} v_sym
        # Skip the first (constant) eigenvector
        embed = D_inv_sqrt @ eigvecs_sym[:, 1:dim+1]
    return embed


# Testing script
if __name__ == '__main__':
    data = simulate_3d_manifold()
    visualization_3d_pts(data['3d'], prefix='data')
    visualization_2d_pts(data['2d'], prefix='data')
    for h in [0.01, 0.1, 1, 10, 100]:
        z0 = kernel_pca(xs=data['3d'], d=2, k_type='rbf', bandwidth=h)
        visualization_2d_pts(z0, prefix='KPCA_rbf_{}'.format(int(np.log10(h))))
    z1 = kernel_pca(xs=data['3d'], d=2, k_type='linear')
    visualization_2d_pts(z1, prefix='KPCA_linear')

    for k in [3, 5, 10, 25, 50, 100, 200]:
        z1 = locally_linear_embedding(xs=data['3d'], k=k)
        visualization_2d_pts(z1, prefix='LLE_{}'.format(k))

    for k in [3, 5, 10, 25, 50, 100, 200, None]:
        for normalize in [True, False]:
            z2 = laplacian_eigenmaps(xs=data['3d'], k=k, normalize=normalize)
            if k is None:
                prefix = 'LE_full_{}'.format(normalize)
            else:
                prefix = 'LE_{}_{}'.format(k, normalize)
            visualization_2d_pts(z2, prefix=prefix)