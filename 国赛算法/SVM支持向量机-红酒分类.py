import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def generate_wine_data():
    """模拟葡萄酒数据(含2个噪声点)"""
    # 特征:[酒精含量(%vol), 柠檬酸含量(g/L)]
    X = np.array([
        [8.5, 0.2],  # 红葡萄酒(正常)
        [9.0, 0.3],  # 红葡萄酒(正常)
        [9.5, 0.4],  # 红葡萄酒(支持向量)
        [10.0, 0.5],  # 红葡萄酒(正常)
        [13.0, 0.7],  # 红葡萄酒(噪声点)
        [11.0, 0.6],  # 白葡萄酒(正常)
        [11.5, 0.7],  # 白葡萄酒(正常)
        [12.0, 0.8],  # 白葡萄酒(支持向量)
        [12.5, 0.9],  # 白葡萄酒(正常)
        [10.5, 0.5]  # 白葡萄酒(噪声点)
    ])
    # 标签:+1=红葡萄酒,-1=白葡萄酒
    y = np.array([+1, +1, +1, +1, +1, -1, -1, -1, -1, -1])
    return X, y

def preprocess_data(X):
    """标准化特征(均值0,方差1)"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def plot_decision_boundary(model, X, y, scaler, title):
    """绘制SVM决策边界和支持向量"""
    # 生成网格点
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    # 标准化网格点并预测
    xx_scaled = scaler.transform(np.c_[xx1.ravel(), xx2.ravel()])
    Z = model.predict(xx_scaled)
    Z = Z.reshape(xx1.shape)
    # 绘制决策边界和数据点
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[y == +1, 0], X[y == +1, 1], c='red', label='红葡萄酒(+1)')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='白葡萄酒(-1)')
    # 标记支持向量
    support_vectors = scaler.inverse_transform(model.support_vectors_)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                s=150, edgecolor='black', facecolor='none', label='支持向量')
    # 设置图表信息
    plt.title(title)
    plt.xlabel('酒精含量(%vol)')
    plt.ylabel('柠檬酸含量(g/L)')
    plt.legend()
    plt.show()

def train_linear_svm(X_scaled, y, C=1.0):
    """训练线性核SVM模型"""
    model = SVC(kernel='linear', C=C)
    model.fit(X_scaled, y)
    return model

def train_rbf_svm(X_scaled, y, param_grid=None):
    """训练RBF核SVM模型(网格搜索调参)"""
    if param_grid is None:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_scaled, y)
    print(f"RBF核最佳参数:{grid.best_params_}")
    print(f"RBF核最佳交叉验证准确率:{grid.best_score_:.2f}")
    return grid.best_estimator_

if __name__ == "__main__":
    # 1. 生成数据
    X, y = generate_wine_data()
    print("原始数据:")
    print(pd.DataFrame(np.c_[X, y], columns=['酒精含量', '柠檬酸含量', '标签']))

    # 2. 预处理数据
    X_scaled, scaler = preprocess_data(X)

    # 3. 训练线性核SVM
    print("\n=== 线性核SVM ===")
    linear_model = train_linear_svm(X_scaled, y, C=1.0)
    print(f"线性核支持向量数量(红葡萄酒/白葡萄酒):{linear_model.n_support_}")

    # 4. 训练RBF核SVM(调参)
    print("\n=== RBF核SVM(网格搜索调参)===")
    rbf_model = train_rbf_svm(X_scaled, y)
    print(f"RBF核支持向量数量(红葡萄酒/白葡萄酒):{rbf_model.n_support_}")

    # 5. 绘制决策边界
    plot_decision_boundary(linear_model, X, y, scaler, title='线性核SVM分类边界(C=1.0)')
    plot_decision_boundary(rbf_model, X, y, scaler, title='RBF核SVM分类边界(最佳参数)')