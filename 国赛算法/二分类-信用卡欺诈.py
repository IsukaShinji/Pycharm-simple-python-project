# 导入必要库(小白需记住这些常用库)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_curve, auc,
                             precision_recall_curve, confusion_matrix)


# 设置随机种子(保证结果可重复)
np.random.seed(42)

def generate_credit_card_data(n_samples=1000, fraud_ratio=0.2):
    """
    生成信用卡交易模拟数据(小白重点理解特征分布差异)
    参数:
    n_samples: 总样本量
    fraud_ratio: 欺诈样本比例(类不平衡)
    返回:
    df: 包含特征和标签的DataFrame
    """
    # 计算正常/欺诈样本量
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # ---------------------- 1. 生成正常交易数据(0 类)
    normal_data = {
        '金额': np.random.normal(loc=1000, scale=200, size=n_normal),  # 正常金额:均值1000,标准差200
        '时间': np.random.randint(low=10, high=21, size=n_normal),  # 正常时间:10-20 点(白天)
        '是否异地': np.random.binomial(n=1, p=0.1, size=n_normal),  # 正常异地:10%概率
        '5 分钟内交易次数': np.random.poisson(lam=1, size=n_normal)  # 正常次数:均值1 次
    }
    normal_df = pd.DataFrame(normal_data)
    normal_df['标签'] = 0  # 0=正常

    # ---------------------- 2. 生成欺诈交易数据(1 类)
    fraud_data = {
        '金额': np.random.normal(loc=5000, scale=1000, size=n_fraud),  # 欺诈金额:均值5000,标准差1000(更大)
        '时间': np.random.randint(low=0, high=6, size=n_fraud),  # 欺诈时间:0-5 点(凌晨)
        '是否异地': np.random.binomial(n=1, p=0.8, size=n_fraud),  # 欺诈异地:80%概率(更高)
        '5 分钟内交易次数': np.random.poisson(lam=3, size=n_fraud)  # 欺诈次数:均值3 次(更频繁)
    }
    fraud_df = pd.DataFrame(fraud_data)
    fraud_df['标签'] = 1  # 1=欺诈

    # ---------------------- 3. 合并数据并打乱顺序----------------------
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序
    return df


def plot_roc_pr_curve(y_true, y_prob):
    """绘制ROC 曲线和PR 曲线(评估模型性能的关键图)"""
    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC 曲线
    ax1.plot(fpr, tpr, color='blue', label=f'ROC 曲线(AUC={roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='随机猜测')
    ax1.set_xlabel('假阳性率(FPR)')
    ax1.set_ylabel('真阳性率(TPR)')
    ax1.set_title('ROC 曲线(评估整体区分能力)')
    ax1.legend()

    # PR 曲线
    ax2.plot(recall, precision, color='green', label='PR 曲线')
    ax2.set_xlabel('召回率(Recall)')
    ax2.set_ylabel('精确率(Precision)')
    ax2.set_title('PR 曲线(评估类不平衡下的性能)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, scaler, feature1='金额', feature2='时间'):
    """
    绘制逻辑回归的决策边界(小白重点理解:线性模型如何划分类别)
    参数:
    model: 训练好的逻辑回归模型
    X: 特征数据(DataFrame)
    y: 标签数据
    scaler: 训练集的标准化器(用于网格数据标准化)
    feature1: 横轴特征
    feature2: 纵轴特征
    """
    # 选取两个特征,固定其他特征为均值(或模式)
    fixed_features = {  # 其他特征:是否异地(取0,本地)､5 分钟内交易次数(取1,正常次数)
        '是否异地': 0,
        '5 分钟内交易次数': 1
    }

    # 生成网格点(覆盖两个特征的取值范围)
    x1 = np.linspace(X[feature1].min(), X[feature1].max(), 100)
    x2 = np.linspace(X[feature2].min(), X[feature2].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)

    # 构造网格点的特征数据(包含固定特征,保持与原数据列顺序一致)
    grid_data = pd.DataFrame({
        feature1: X1.ravel(),
        feature2: X2.ravel(),
        **fixed_features
    })

    # 调整列顺序与原数据一致(避免标准化时特征顺序错误)
    grid_data = grid_data[X.columns]

    # 标准化网格点特征(使用训练集的标准化器)
    grid_data_scaled = scaler.transform(grid_data)

    # 预测网格点的概率(p=0.5 是决策边界)
    y_prob_grid = model.predict_proba(grid_data_scaled)[:, 1]
    y_prob_grid = y_prob_grid.reshape(X1.shape)

    # 绘制决策边界(p=0.5 的等高线)
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X1, X2, y_prob_grid, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.3)
    plt.colorbar(contour, label='欺诈概率')

    # 绘制样本点(正常=蓝色,欺诈=红色)
    sns.scatterplot(x=X[feature1], y=X[feature2], hue=y, palette={0: 'blue', 1: 'red'},
                    alpha=0.7, edgecolor='black')

    # 添加标签和标题
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'逻辑回归决策边界({feature1} vs {feature2})')
    plt.legend(title='标签', labels=['正常', '欺诈'])
    plt.show()

# ---------------------- 主程序:逻辑回归完整流程----------------------
if __name__ == '__main__':
    # 1. 生成模拟数据(小白可调整n_samples 和fraud_ratio)
    df = generate_credit_card_data(n_samples=1000, fraud_ratio=0.2)
    print("数据形状:", df.shape)
    print("标签分布:\n", df['标签'].value_counts(normalize=True))  # 查看类不平衡情况(正常80%,欺诈20%)

    # 2. 划分特征(X)和标签(y)
    X = df.drop('标签', axis=1)
    y = df['标签']

    # 3. 拆分训练集和测试集(70%训练,30%测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print("训练集形状:", X_train.shape)
    print("测试集形状:", X_test.shape)

    # 4. 特征标准化(避免尺度差异影响模型,必须fit 在训练集上)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练逻辑回归模型(处理类不平衡:class_weight='balanced')
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    # 输出模型参数(小白可解释特征重要性)
    print("\n模型权重(w):", model.coef_[0])
    print("模型偏置(b):", model.intercept_[0])
    print("特征重要性(绝对值越大,对欺诈的贡献越大):")
    for feature, weight in zip(X.columns, model.coef_[0]):
        print(f"  {feature}: {weight:.2f}")

    # 6. 预测测试集(概率和类别)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # 欺诈概率
    y_pred = model.predict(X_test_scaled)  # 默认阈值0.5 的预测类别

    # 7. 评估模型性能(默认阈值0.5)
    print("\n---------------------- 默认阈值(0.5)评估 ----------------------")
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 绘制ROC/PR 曲线(评估整体性能)
    plot_roc_pr_curve(y_test, y_prob)

    # 8. 调整阈值(比如提高到0.7,追求高精确率,避免误判正常交易)
    threshold = 0.7
    y_pred_high_precision = (y_prob >= threshold).astype(int)
    print(f"\n---------------------- 阈值={threshold}评估----------------------")
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_high_precision))
    print(classification_report(y_test, y_pred_high_precision))

    # 9. 绘制决策边界(选取“金额”和“时间”两个关键特征,传入标准化器)
    plot_decision_boundary(model, X_test, y_test, scaler, feature1='金额', feature2='时间')