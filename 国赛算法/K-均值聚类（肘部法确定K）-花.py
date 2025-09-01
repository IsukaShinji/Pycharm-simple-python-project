from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载数据（取前两个特征，方便画图）
iris = load_iris()
X = iris.data[:, :2]  # 萼片长度、萼片宽度

# 计算不同 K 值的 SSE
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # inertia_ 属性就是 SSE

# 画肘部曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o', linestyle='--')
plt.xlabel('K 值（群数）')
plt.ylabel('SSE（误差平方和）')
plt.title('Iris 数据集的肘部曲线')
plt.grid(True)
plt.show()