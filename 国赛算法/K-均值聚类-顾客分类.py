# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # 用于3D可视化
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples

def generate_customer_data():
    """模拟生成顾客消费数据(RFM模型)
    返回:包含CustomerID、Recency、Frequency、Monetary的DataFrame
    """
    # 高价值顾客(20人):最近消费(≤30天)、高频(≥12次)、高金额(≥10000元)
    high_value = pd.DataFrame({
        'Recency': np.random.randint(1, 31, size=20),  # 最近1-30天消费
        'Frequency': np.random.randint(12, 20, size=20),  # 年消费12-19次
        'Monetary': np.random.randint(10000, 20001, size=20)  # 年消费10000-20000元
    })
    # 中等价值顾客(50人):中等消费频率(4-11次)、中等金额(3000-9999元)、最近1-6个月消费(31-180天)
    medium_value = pd.DataFrame({
        'Recency': np.random.randint(31, 181, size=50),  # 最近31-180天消费
        'Frequency': np.random.randint(4, 12, size=50),  # 年消费4-11次
        'Monetary': np.random.randint(3000, 10000, size=50)  # 年消费3000-9999元
    })
    # 低价值顾客(30人):低频(≤3次)、低金额(≤2999元)、很久未消费(≥181天)
    low_value = pd.DataFrame({
        'Recency': np.random.randint(181, 366, size=30),  # 最近181-365天消费(近1年未消费)
        'Frequency': np.random.randint(1, 4, size=30),  # 年消费1-3次
        'Monetary': np.random.randint(1000, 3000, size=30)  # 年消费1000-2999元
    })
    # 合并三类数据,打乱顺序,添加顾客ID
    data = pd.concat([high_value, medium_value, low_value], ignore_index=True)
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)  # 随机打乱
    data['CustomerID'] = range(1, len(data) + 1)  # 添加顾客ID
    # 调整列顺序(CustomerID在前,方便查看)
    data = data[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    return data


def preprocess_data(data):
    """数据预处理:提取RFM特征并归一化
    参数:data - 原始顾客数据(包含CustomerID、Recency、Frequency、Monetary)
    返回:scaled_features - 归一化后的特征矩阵(numpy数组)
          scaler - 归一化器(用于后续反归一化质心)
    """
    # 提取RFM特征(排除CustomerID)
    features = data[['Recency', 'Frequency', 'Monetary']]
    # 使用Min-Max归一化(缩放到0-1区间)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def plot_elbow_curve(scaled_features):
    """绘制肘部曲线,选择最佳K值
    参数:scaled_features - 归一化后的特征矩阵
    """
    sse = []  # 存储不同K值的SSE
    k_range = range(1, 11)  # 尝试K=1到10
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)  # 使用K-Means++初始化
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)  # inertia_属性是SSE
    # 绘制肘部曲线
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse, marker='o', linestyle='--', color='#1f77b4')
    plt.xlabel('K值(群数)', fontsize=12)
    plt.ylabel('SSE(误差平方和)', fontsize=12)
    plt.title('肘部曲线(选择最佳K值)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def perform_kmeans(scaled_features, k=3):
    """执行K-Means聚类
    参数:scaled_features - 归一化后的特征矩阵
          k - 群数(默认3)
    返回:labels - 每个样本的群标签(numpy数组)
          centroids - 群质心(归一化后的坐标,numpy数组)
          kmeans_model - K-Means模型对象
    """
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    labels = kmeans.fit_predict(scaled_features)  # 拟合模型并预测标签
    centroids = kmeans.cluster_centers_  # 获取群质心(归一化后的坐标)
    return labels, centroids, kmeans


def plot_clustering_results(data, scaled_features, labels, centroids, scaler):
    """可视化聚类结果(3D散点图+2D pairwise散点图)
    参数:data - 原始顾客数据
          scaled_features - 归一化后的特征矩阵
          labels - 群标签
          centroids - 归一化后的质心
          scaler - 归一化器(用于反归一化质心)
    """
    # 反归一化质心(将归一化的坐标转换为原始范围)
    original_centroids = scaler.inverse_transform(centroids)
    original_centroids = pd.DataFrame(original_centroids, columns=['Recency', 'Frequency', 'Monetary'])

    # 1. 3D散点图(Recency、Frequency、Monetary)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制数据点(按群标签着色)
    scatter = ax.scatter(
        data['Recency'], data['Frequency'], data['Monetary'],
        c=labels, cmap='viridis', s=50, alpha=0.8
    )
    # 绘制质心(红色星号,放大显示)
    centroid_scatter = ax.scatter(
        original_centroids['Recency'], original_centroids['Frequency'], original_centroids['Monetary'],
        c='red', marker='*', s=200, label='质心'
    )
    # 设置坐标轴标签和标题
    ax.set_xlabel('最近一次消费天数(Recency)', fontsize=10)
    ax.set_ylabel('年消费次数(Frequency)', fontsize=10)
    ax.set_zlabel('年消费总额(Monetary/元)', fontsize=10)
    ax.set_title('顾客消费数据K-Means聚类结果(3D视图)', fontsize=14)
    # 添加图例(群标签+质心)
    legend1 = ax.legend(*scatter.legend_elements(), title='群标签', loc='upper left')
    ax.add_artist(legend1)
    ax.legend(handles=[centroid_scatter], loc='upper right')

    # 2. 2D Pairwise散点图(两两特征组合)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列
    feature_pairs = [('Recency', 'Frequency'), ('Recency', 'Monetary'), ('Frequency', 'Monetary')]
    for i, (x_col, y_col) in enumerate(feature_pairs):
        ax = axes[i]
        # 绘制数据点
        ax.scatter(data[x_col], data[y_col], c=labels, cmap='viridis', s=50, alpha=0.8)
        # 绘制质心(原始范围)
        ax.scatter(original_centroids[x_col], original_centroids[y_col], c='red', marker='*', s=200, label='质心')
        # 设置标签和标题
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f'{x_col} vs {y_col}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()  # 添加子图图例
    plt.tight_layout()  # 调整子图间距
    plt.show()


def evaluate_clustering(scaled_features, labels):
    """评估聚类效果(轮廓系数)
    参数:scaled_features - 归一化后的特征矩阵
          labels - 群标签
    """
    # 计算平均轮廓系数
    silhouette_avg = silhouette_score(scaled_features, labels)
    print(f'平均轮廓系数:{silhouette_avg:.4f}')
    # 计算每个样本的轮廓系数
    sample_silhouette_values = silhouette_samples(scaled_features, labels)

    # 绘制轮廓系数直方图
    fig, ax = plt.subplots(figsize=(8, 5))
    n_clusters = len(np.unique(labels))  # 群数
    y_lower = 10  # 底部留白
    for i in range(n_clusters):
        # 获取第i个群的轮廓系数
        ith_cluster_silhouette = sample_silhouette_values[labels == i]
        ith_cluster_silhouette.sort()  # 排序
        size_cluster = ith_cluster_silhouette.shape[0]  # 群大小
        y_upper = y_lower + size_cluster  # 该群的y轴上限

        # 绘制直方图(填充颜色)
        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0,
            ith_cluster_silhouette,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        # 添加群标签(在直方图中间)
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i), fontsize=10)
        y_lower = y_upper + 10  # 下一个群的y轴起始位置

    # 设置坐标轴标签和标题
    ax.set_xlabel('轮廓系数值', fontsize=12)
    ax.set_ylabel('群标签', fontsize=12)
    ax.set_title(f'各群轮廓系数分布(平均:{silhouette_avg:.4f})', fontsize=14)
    ax.set_xlim([-0.1, 1])  # x轴范围(轮廓系数取值范围)
    ax.set_yticks([])  # 隐藏y轴刻度
    plt.show()


if __name__ == '__main__':
    # 1. 生成模拟数据
    print('=== 1. 生成模拟顾客数据 ===')
    data = generate_customer_data()
    print('数据生成完成(100条),前5条数据:')
    print(data.head(), '\n')

    # 2. 数据预处理(归一化)
    print('=== 2. 数据预处理(归一化) ===')
    scaled_features, scaler = preprocess_data(data)
    print(f'归一化后的特征形状:{scaled_features.shape}(100个样本,3个特征)\n')

    # 3. 肘部法选择最佳K值(运行一次后可注释,直接用K=3)
    print('=== 3. 绘制肘部曲线(选择K值) ===')
    plot_elbow_curve(scaled_features)
    print('提示:根据肘部曲线,最佳K值为3(案例预期)\n')

    # 4. 执行K-Means聚类(K=3)
    print('=== 4. 执行K-Means聚类(K=3) ===')
    labels, centroids, kmeans_model = perform_kmeans(scaled_features, k=3)
    data['Cluster'] = labels  # 将群标签添加到原始数据
    print('聚类完成,群标签已添加到数据中,前5条数据:')
    print(data.head(), '\n')

    # 5. 反归一化质心(查看原始范围)
    print('=== 5. 反归一化质心(原始范围) ===')
    original_centroids = scaler.inverse_transform(centroids)
    original_centroids = pd.DataFrame(original_centroids, columns=['Recency', 'Frequency', 'Monetary'])
    original_centroids['Cluster'] = range(3)  # 添加群标签
    print(original_centroids[['Cluster', 'Recency', 'Frequency', 'Monetary']], '\n')

    # 6. 可视化聚类结果
    print('=== 6. 可视化聚类结果 ===')
    plot_clustering_results(data, scaled_features, labels, centroids, scaler)
    print('可视化完成(关闭图表继续运行)\n')

    # 7. 评估聚类效果(轮廓系数)
    print('=== 7. 评估聚类效果(轮廓系数) ===')
    evaluate_clustering(scaled_features, labels)
    print('评估完成\n')

    # 8. 业务结果解读(策略建议)
    print('=== 8. 业务结果解读(策略建议) ===')
    for cluster in range(3):
        # 获取该群的质心(原始范围)
        centroid = original_centroids[original_centroids['Cluster'] == cluster].iloc[0]
        # 统计群大小
        cluster_size = len(data[data['Cluster'] == cluster])
        print(f'\n群{cluster}:')
        print(f' - 群大小:{cluster_size}人')
        print(
            f' - 质心特征:最近消费{centroid["Recency"]:.0f}天,年消费{centroid["Frequency"]:.0f}次,年消费总额{centroid["Monetary"]:.0f}元')

        # 根据质心特征判断群类型,给出策略建议
        if centroid['Recency'] <= 30 and centroid['Frequency'] >= 12 and centroid['Monetary'] >= 10000:
            print(' - 群类型:高价值顾客(核心利润来源)')
            print(' - 策略建议:提供专属优惠券、VIP服务、优先体验新商品,维护忠诚度')
        elif 31 <= centroid['Recency'] <= 180 and 4 <= centroid['Frequency'] <= 11 and 3000 <= centroid[
            'Monetary'] <= 9999:
            print(' - 群类型:中等价值顾客(主要客群)')
            print(' - 策略建议:通过推荐系统提升消费频率,或推出满减活动提高客单价')
        else:
            print(' - 群类型:低价值/流失顾客(需唤醒)')
            print(' - 策略建议:发送唤醒邮件/短信,提供限时折扣,吸引回头消费')
