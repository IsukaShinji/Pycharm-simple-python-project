import numpy as np
import matplotlib.pyplot as plt


def generate_data(k, n_per_cluster, mu, sigma):
    """
    生成 k 个簇的二维高斯分布数据
    :param k: 簇的数量
    :param n_per_cluster: 每个簇的样本数量
    :param mu: 每个簇的均值（二维数组）
    :param sigma: 每个簇的标准差（二维数组）
    :return: 生成的数据点
    """
    data = []
    for i in range(k):
        cluster = np.random.randn(n_per_cluster, 2) * sigma[i] + mu[i]
        data.extend(cluster)
    return np.array(data)


def kmeans(data, k, max_iterations=100):
    """
    K-means 聚类算法（严格遵循伪代码逻辑）
    :param data: 输入数据点 (n_samples, n_features)
    :param k: 聚类的类别数
    :param max_iterations: 最大迭代次数
    :return: 聚类标签 (n_samples,), 聚类中心 (k, n_features)
    """
    # Step 2: 随机初始化聚类中心
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Step 4: 为每个数据点找到最近的聚类中心
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)

        # Step 5: 更新聚类中心为每个簇的均值
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 检查是否收敛（聚类中心不再变化）
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def visualize_clusters(data, labels, centroids):
    """
    可视化聚类结果
    :param data: 输入数据点
    :param labels: 聚类标签
    :param centroids: 聚类中心
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(8, 6))

    # 绘制每个聚类的点
    for label in unique_labels:
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")

    # 绘制聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

    plt.title("K-means Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # 参数设置
    k = 3  # 聚类的数量
    n_per_cluster = 100  # 每个簇的样本数量
    mu = np.array([[0, 0], [5, 5], [10, 0]])  # 每个簇的均值
    sigma = np.array([[1, 1], [1, 1], [1, 1]])  # 每个簇的标准差

    # 生成数据
    data = generate_data(k, n_per_cluster, mu, sigma)

    # 运行 K-means 算法
    labels, centroids = kmeans(data, k)

    # 可视化结果
    visualize_clusters(data, labels, centroids)