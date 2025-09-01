# 导入必要的库(小白需提前安装:pip install numpy pandas scipy matplotlib sklearn)
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # 用于评估聚类效果


def plot_dendrogram(features, sample_labels, method='average'):
    """
    绘制层次聚类树状图(核心函数,展示样本间的层级关系)
    参数说明:
    sample_labels: 样本标签(格式:样本ID+真实亚型,用于树状图显示)
    features: 特征数据(基因表达量,DataFrame 格式)
    method: 簇间距离度量方法(数模首选'average',即平均Linkage)
    返回:
    linkage_matrix: 层次聚类链接矩阵(用于后续计算簇数量和结果)
    """
    # 1. 计算层次聚类链接矩阵(凝聚式,自底向上合并)
    # metric='euclidean':样本间距离用欧氏距离(最常用)
    linkage_matrix = hierarchy.linkage(features.values, method=method, metric='euclidean')

    # 2. 绘制树状图
    plt.figure(figsize=(12, 6))  # 设置图幅大小
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=sample_labels,  # 样本标签(显示在x 轴)
        leaf_rotation=45,  # 标签旋转45 度,避免重叠
        leaf_font_size=10,  # 标签字体大小
        color_threshold=0,  # 初始颜色阈值(0 表示所有分支同色,后续可调整)
        above_threshold_color='gray'  # 超过阈值的分支颜色(灰色)
    )

    # 添加图标题和坐标轴标签
    plt.title('肿瘤患者样本层次聚类树状图(平均Linkage)', fontsize=14)
    plt.xlabel('样本(样本ID+真实肿瘤亚型)', fontsize=12)
    plt.ylabel('合并距离(值越大,样本越不相似)', fontsize=12)

    # 显示网格线(y 轴,辅助判断合并距离)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局(避免标签溢出)
    plt.tight_layout()
    return linkage_matrix


def determine_optimal_clusters(linkage_matrix, features, max_k=10):
    """
    用肘方法确定最优簇数量(避免主观判断,数模常用)
    参数说明:
    linkage_matrix: 层次聚类链接矩阵(来自plot_dendrogram 函数)
    features: 特征数据(基因表达量,DataFrame 格式)
    max_k: 最大尝试的簇数量(不超过样本量-1,默认10)
    返回:
    optimal_k: 最优簇数量(肘点对应的k 值)
    """
    # 1. 计算不同k 值的SSE(误差平方和:每个点到簇中心的距离平方和)
    n_samples = linkage_matrix.shape[0] + 1  # 样本量=链接矩阵行数+1
    max_k = min(max_k, n_samples - 1)  # 确保max_k 不超过样本量-1
    sse = []  # 存储每个k 对应的SSE

    for k in range(1, max_k + 1):
        # 获取k 个簇的标签(criterion='maxclust'表示指定簇数量)
        cluster_labels = hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')

        # 计算每个簇的中心(均值)
        cluster_centers = [features.values[cluster_labels == i].mean(axis=0) for i in range(1, k + 1)]

        # 计算SSE(累加每个点到簇中心的距离平方)
        sse_k = 0
        for i in range(1, k + 1):
            cluster_points = features.values[cluster_labels == i]
            center = cluster_centers[i - 1]
            sse_k += ((cluster_points - center) ** 2).sum().sum()  # 按行求和再总和
        sse.append(sse_k)

    # 2. 绘制肘方法曲线(找SSE 下降最快的点)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='--', color='#1f77b4')
    plt.title('肘方法确定最优簇数量', fontsize=14)
    plt.xlabel('簇数量k', fontsize=12)
    plt.ylabel('SSE(误差平方和,值越小簇越紧凑)', fontsize=12)
    plt.xticks(range(1, max_k + 1))  # x 轴显示所有k 值
    plt.grid(alpha=0.7)  # 显示网格线
    plt.tight_layout()

    # 3. 寻找肘点(SSE 下降速率突然变慢的点)
    # 计算相邻k 值的SSE 差值(绝对值越大,下降越快)
    sse_diff = np.diff(sse)
    optimal_k = np.argmax(np.abs(sse_diff)) + 2  # 修正肘点计算逻辑
    return optimal_k


if __name__ == '__main__':
    # 1. 模拟生成案例数据(与用户提供的案例完全一致)
    data = pd.DataFrame({
        '基因X 表达量': [11, 12, 10, 11, 3, 4, 2, 3],  # A 类高表达,B 类低表达
        '样本ID': [1, 2, 3, 4, 5, 6, 7, 8],
        '基因Y 表达量': [9, 10, 3, 4, 9, 10, 3, 4],  # A1/B1 高表达,A2/B2 低表达
        '基因Z 表达量': [3, 2, 9, 8, 2, 3, 10, 9],  # A1/B1 低表达,A2/B2 高表达
        '真实肿瘤亚型': ['A1', 'A1', 'A2', 'A2', 'B1', 'B1', 'B2', 'B2']  # 真实分类(用于验证)
    })
    print("=== 模拟生成的肿瘤患者基因表达数据===")
    print('\n' + '-' * 60 + '\n')
    print(data)

    # 2. 数据预处理(提取特征,准备聚类)
    features = data[['基因X 表达量', '基因Y 表达量', '基因Z 表达量']]  # 提取基因表达量作为特征
    sample_labels = [f'样本{id}({subtype})' for id, subtype in zip(data['样本ID'], data['真实肿瘤亚型'])]  # 样本标签(含真实亚型)

    # 3. 绘制层次聚类树状图(直观展示样本关系)
    print("=== 正在绘制层次聚类树状图===")
    linkage_matrix = plot_dendrogram(features, sample_labels)  # 调用函数绘制树状图
    # 添加颜色阈值线(y=8,将簇分成A､B 两个大簇,与真实情况一致)
    plt.axhline(y=8, color='red', linestyle='--', linewidth=1.5, label='颜色阈值(y=8)')
    plt.legend()  # 显示图例
    plt.show()  # 显示树状图
    print('\n' + '-' * 60 + '\n')

    # 4. 用肘方法确定最优簇数量(避免主观判断)
    print("=== 正在用肘方法确定最优簇数量===")
    optimal_k = determine_optimal_clusters(linkage_matrix, features, max_k=8)  # 传入特征数据
    print(f"肘方法推荐的最优簇数量:{optimal_k}")
    plt.show()  # 显示肘方法曲线
    print('\n' + '-' * 60 + '\n')

    # 5. 获取聚类结果(根据最优簇数量)
    print(f"=== 获取{optimal_k}个簇的聚类结果===")
    cluster_labels = hierarchy.fcluster(linkage_matrix, optimal_k, criterion='maxclust')  # 获取簇标签
    data['聚类结果'] = cluster_labels  # 将结果添加到原数据中
    # 打印聚类结果(按聚类结果排序,方便查看)
    print("聚类结果(按聚类结果排序):")
    print(data.sort_values(by='聚类结果'))
    print('\n' + '-' * 60 + '\n')

    # 6. 验证聚类结果(与真实肿瘤亚型对比)
    print("=== 聚类结果与真实肿瘤亚型对比===")
    # 计算轮廓系数(评估聚类效果,值越大越好,范围[-1,1])
    silhouette_avg = silhouette_score(features.values, cluster_labels)
    print(f"轮廓系数(聚类效果评估):{silhouette_avg:.2f}")  # 若接近1,说明聚类效果好

    # 打印每个簇的真实肿瘤亚型分布(验证是否与真实情况一致)
    print("\n 每个簇的真实肿瘤亚型分布:")
    print(data.groupby('聚类结果')['真实肿瘤亚型'].value_counts())