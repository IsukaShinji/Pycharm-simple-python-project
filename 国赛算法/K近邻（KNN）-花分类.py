import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. 模拟数据(根据用户提供的表格)
# 训练集:特征(花萼长、花萼宽、花瓣长、花瓣宽)+ 标签(0=山鸢尾,1=变色鸢尾,2=维吉尼亚鸢尾)
X_train_raw = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 山鸢尾
    [4.9, 3.0, 1.4, 0.2],  # 山鸢尾
    [4.7, 3.2, 1.3, 0.2],  # 山鸢尾
    [6.0, 2.2, 4.0, 1.0],  # 变色鸢尾
    [5.5, 2.3, 4.0, 1.3],  # 变色鸢尾
    [5.7, 2.8, 4.5, 1.3],  # 变色鸢尾
    [6.7, 3.0, 5.2, 2.3],  # 维吉尼亚鸢尾
    [6.3, 2.9, 5.6, 1.8],  # 维吉尼亚鸢尾
    [6.2, 3.4, 5.4, 2.3],  # 维吉尼亚鸢尾
    [6.5, 3.2, 5.1, 2.0]  # 维吉尼亚鸢尾
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 训练集标签

# 测试集:待预测的两个样本(特征与训练集一致)
X_test_raw = np.array([
    [5.0, 3.3, 1.5, 0.3],  # 样本11(特征接近山鸢尾)
    [6.4, 3.1, 5.5, 2.2]  # 样本12(特征接近维吉尼亚鸢尾)
])


# 2. 特征缩放(Min-Max 归一化)
def scale_features(X_train, X_test):
    """
    对训练集和测试集进行Min-Max归一化(用训练集的最值缩放,避免数据泄露)
    参数:
        X_train: 训练集特征(原始值,形状为(样本数, 特征数))
        X_test: 测试集特征(原始值,形状为(样本数, 特征数))
    返回:
        X_train_scaled: 归一化后的训练集特征(0-1区间)
        X_test_scaled: 归一化后的测试集特征(0-1区间)
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合缩放器(计算最值)
    X_test_scaled = scaler.transform(X_test)  # 用训练集的最值缩放测试集
    return X_train_scaled, X_test_scaled


# 执行特征缩放
X_train, X_test = scale_features(X_train_raw, X_test_raw)


# 3. 实现KNN分类器(含加权投票)
class KNNClassifier:
    def __init__(self, k=5, weighted=True):
        """
        KNN分类器初始化
        参数:
            k: 邻居数量(必须为正整数,默认5)
            weighted: 是否使用加权投票(默认True,距离倒数加权;False为平等投票)
        """
        if k <= 0:
            raise ValueError("K值必须为正整数")
        self.k = k
        self.weighted = weighted
        self.X_train = None  # 保存训练集特征(已归一化)
        self.y_train = None  # 保存训练集标签(整数编码)

    def fit(self, X_train, y_train):
        """
        拟合模型(KNN无参数学习,仅保存训练数据)
        参数:
            X_train: 训练集特征(已归一化,形状为(样本数, 特征数))
            y_train: 训练集标签(整数编码,形状为(样本数,))
        """
        self.X_train = X_train
        self.y_train = y_train

    def _compute_euclidean_distance(self, x):
        """
        计算单个测试样本与所有训练样本的欧氏距离
        参数:
            x: 单个测试样本(已归一化,形状为(特征数,))
        返回:
            distances: 与所有训练样本的欧氏距离(形状为(训练集样本数,))
        """
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # 广播计算

    def _weighted_vote(self, distances, neighbor_indices):
        """
        加权投票(距离倒数,距离越小权重越大)
        参数:
            distances: 测试样本与K个邻居的距离(形状为(K,))
            neighbor_indices: K个邻居在训练集中的索引(形状为(K,))
        返回:
            predicted_class: 预测类别(整数编码)
        """
        neighbor_labels = self.y_train[neighbor_indices]  # 获取K个邻居的标签
        weights = 1.0 / (distances + 1e-8)  # 计算权重(加1e-8避免除以0)
        class_weights = {}
        for label, weight in zip(neighbor_labels, weights):
            class_weights[label] = class_weights.get(label, 0.0) + weight  # 累加权重
        return max(class_weights, key=class_weights.get)  # 选择加权和最大的类别

    def predict(self, X_test):
        """
        预测测试集结果
        参数:
            X_test: 测试集特征(已归一化,形状为(样本数, 特征数))
        返回:
            predictions: 测试集预测结果(整数编码,形状为(样本数,))
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("模型未拟合,请先调用fit方法")
        if self.k > len(self.X_train):
            raise ValueError(f"K值({self.k})超过训练集大小({len(self.X_train)})")

        predictions = []
        for x in X_test:
            distances = self._compute_euclidean_distance(x)  # 计算距离
            top_k_indices = np.argsort(distances)[:self.k]  # 取前K个邻居索引
            top_k_distances = distances[top_k_indices]  # 取前K个邻居距离

            if self.weighted:
                pred = self._weighted_vote(top_k_distances, top_k_indices)
            else:
                # 平等投票(取出现次数最多的标签)
                pred = np.bincount(self.y_train[top_k_indices]).argmax()
            predictions.append(pred)
        return np.array(predictions)


# 4. 交叉验证选择最优K值
def select_best_k(X_train, y_train, k_candidates=[3, 5, 7, 9], cv=5):
    """
    用交叉验证选择最优K值
    参数:
        X_train: 训练集特征(已归一化)
        y_train: 训练集标签
        k_candidates: 待尝试的K值列表(默认奇数)
        cv: 交叉验证折数(默认5)
    返回:
        best_k: 最优K值(平均准确率最高)
        k_accuracy: 每个K值的平均准确率
    """
    # 检查候选K值有效性
    valid_k = [k for k in k_candidates if k > 0 and k <= len(X_train)]
    if not valid_k:
        raise ValueError("候选K值均无效(需为正整数且不超过训练集大小)")
    if set(k_candidates) - set(valid_k):
        print(f"警告:候选K值中{set(k_candidates) - set(valid_k)}超过训练集大小,已跳过")

    # 打乱数据并划分成cv折
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    fold_size = len(X_train) // cv
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(cv)]
    if len(X_train) % cv != 0:
        folds[-1] = np.concatenate([folds[-1], indices[cv * fold_size:]])

    k_accuracy = {}
    for k in valid_k:
        fold_acc = []
        for fold_idx in range(cv):
            # 划分训练集和验证集
            val_indices = folds[fold_idx]
            train_indices = np.concatenate([folds[i] for i in range(cv) if i != fold_idx])
            X_train_fold = X_train[train_indices]
            y_train_fold = y_train[train_indices]
            X_val_fold = X_train[val_indices]
            y_val_fold = y_train[val_indices]

            # 检查K值是否超过当前训练集大小
            if k > len(X_train_fold):
                print(f"警告:在折{fold_idx + 1}中,K={k}超过训练集大小,该折准确率设为0")
                fold_acc.append(0.0)
                continue

            # 训练并评估
            knn = KNNClassifier(k=k, weighted=True)
            knn.fit(X_train_fold, y_train_fold)
            y_val_pred = knn.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_val_pred)
            fold_acc.append(acc)

        # 计算平均准确率
        avg_acc = np.mean(fold_acc)
        k_accuracy[k] = avg_acc
        print(f"K={k}时,{cv}折交叉验证平均准确率:{avg_acc:.4f}")

    # 选择最优K值(准确率最高,若相同则取较小K值)
    best_k = max(k_accuracy, key=lambda x: (k_accuracy[x], -x))
    print(f"\n最优K值:{best_k}(平均准确率:{k_accuracy[best_k]:.4f})")
    return best_k, k_accuracy

# 5. 主程序:训练模型并预测测试集
if __name__ == "__main__":
    # 步骤1:交叉验证选择最优K值
    print("=== 交叉验证选择最优K值 ===")
    k_candidates = [3, 5, 7, 9]  # 候选K值(奇数)
    best_k, _ = select_best_k(X_train, y_train, k_candidates)

    # 步骤2:用最优K值训练模型并预测测试集
    print("\n=== 测试集预测结果 ===")
    best_knn = KNNClassifier(k=best_k, weighted=True)
    best_knn.fit(X_train, y_train)
    y_test_pred = best_knn.predict(X_test)

    # 步骤3:输出预测结果(映射为品种名称)
    class_map = {0: "山鸢尾", 1: "变色鸢尾", 2: "维吉尼亚鸢尾"}
    for i in range(len(X_test)):
        sample_id = 11 + i  # 测试集样本ID(11、12)
        predicted_class = class_map[y_test_pred[i]]
        print(f"样本ID {sample_id}:预测品种为｢{predicted_class}｣")
