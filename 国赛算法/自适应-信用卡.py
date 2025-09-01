import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class AdaBoost:
    """
    AdaBoost 算法实现(二分类)
    参数:
        n_estimators: 弱分类器数量(默认5)
    属性:
        estimators: 保存弱分类器及反转标志((clf, reverse))
        alpha: 保存弱分类器的权重
    """
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.estimators = []  # 每个元素是(clf, reverse),reverse 表示是否反转预测
        self.alpha = []  # 弱分类器的权重

    def fit(self, X, y):
        """
        训练AdaBoost 模型
        参数:
            X: 特征矩阵(n_samples, n_features)
            y: 标签(n_samples,),取值为+1(正类,如欺诈)或-1(负类,如正常)
        """
        n_samples = X.shape[0]
        # 1. 初始化样本权重:每个样本权重相等,总和为1
        w = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # 2. 训练弱分类器(决策树桩,max_depth=1)
            clf = DecisionTreeClassifier(max_depth=1, random_state=42)
            clf.fit(X, y, sample_weight=w)  # 传入样本权重
            y_pred = clf.predict(X)

            # 3. 计算加权错误率(误分类样本的权重之和)
            error = w.dot(y_pred != y)  # 等价于sum(w[i] for i in range(n_samples) if y_pred[i] != y[i])

            # 确保弱分类器性能优于随机猜测(错误率<0.5),否则反转预测
            reverse = False
            if error >= 0.5:
                # 反转标签重新训练(将y 替换为-y)
                clf.fit(X, -y, sample_weight=w)
                y_pred = -clf.predict(X)  # 反转预测结果以匹配原标签
                error = w.dot(y_pred != y)  # 重新计算错误率(此时应<0.5)
                reverse = True  # 标记该分类器需要反转预测

            # 4. 计算弱分类器的权重(投票权):避免error=0 或1 导致无穷大
            error = np.clip(error, 1e-10, 1 - 1e-10)  # 限制error 范围
            alpha = 0.5 * np.log((1 - error) / error)  # 公式:α_t = 0.5 * ln((1-ε_t)/ε_t)
            self.alpha.append(alpha)
            self.estimators.append((clf, reverse))  # 保存分类器及反转标志

            # 5. 更新样本权重(误分类样本权重升高,正确分类样本权重降低)
            # 公式:w_{t+1,i} = w_{t,i} * exp(-α_t * y_i * y_pred_i) / Z_t(Z_t 为归一化常数)
            w *= np.exp(-alpha * y * y_pred)
            w /= w.sum()  # 归一化,确保权重总和为1

    def predict(self, X):
        """
        预测样本类别
        参数:
            X: 特征矩阵(n_samples, n_features)
        返回:
            y_pred: 预测标签(n_samples,),取值为+1 或-1
        """
        final_pred = np.zeros(X.shape[0])
        for (clf, reverse), alpha in zip(self.estimators, self.alpha):
            pred = clf.predict(X)
            if reverse:
                pred = -pred  # 反转预测以匹配原标签(训练时反转过标签)
            final_pred += alpha * pred  # 加权求和(弱分类器投票)
        return np.sign(final_pred)  # 取符号得到最终预测(+1=正类,-1=负类)

if __name__ == "__main__":
    # 1. 模拟信用卡欺诈检测数据(8 个正常交易,2 个欺诈交易)
    data = {
        '交易金额(元)': [800, 1200, 300, 1500, 200, 900, 1100, 400, 500, 200],
        '交易时间(小时)': [10, 15, 9, 16, 11, 13, 14, 12, 14, 20],
        '交易地点(是否异地)': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1=异地,0=本地
        '账户历史(月)': [24, 18, 36, 12, 60, 48, 30, 54, 36, 1],
        'y': [-1, -1, -1, -1, -1, -1, -1, -1, +1, +1]  # +1=欺诈(正类),-1=正常(负类)
    }
    df = pd.DataFrame(data)
    y = df['y'].values  # 标签(10 行1 列)
    X = df.drop('y', axis=1).values  # 特征矩阵(10 行4 列)

    # 2. 训练AdaBoost 模型(用3 个弱分类器)
    adaboost = AdaBoost(n_estimators=3)
    adaboost.fit(X, y)

    # 3. 预测训练集(评估模型效果)
    y_pred = adaboost.predict(X)

    # 4. 输出评估结果
    print("=== 模型效果评估===")
    print(f"训练集准确率:{accuracy_score(y, y_pred):.2f}")
    print("混淆矩阵(行=真实标签,列=预测标签;行/列顺序:-1(正常)､+1(欺诈)):")
    print(confusion_matrix(y, y_pred))  # 混淆矩阵解读:[0,0]=正常样本正确数,[1,1]=欺诈样本正确数

    # 5. 测试难样本(欺诈样本9 和10,索引从0 开始)
    print("\n=== 难样本预测===")

    # 样本9(索引8):交易金额500 元､时间14 点､异地､账户历史36 个月(真实标签+1)
    sample9 = np.array([[500, 14, 1, 36]])
    pred9 = adaboost.predict(sample9)[0]
    print(f"样本9 预测结果:{pred9}(正确标签:+1)")

    # 样本10(索引9):交易金额200 元､时间20 点､本地､账户历史1 个月(真实标签+1)
    sample10 = np.array([[200, 20, 0, 1]])
    pred10 = adaboost.predict(sample10)[0]
    print(f"样本10 预测结果:{pred10}(正确标签:+1)")












