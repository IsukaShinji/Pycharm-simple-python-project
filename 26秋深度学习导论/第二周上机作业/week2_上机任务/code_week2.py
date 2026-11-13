import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===== 1. 损失函数 =====
def rmse(y_true, y_pred):
    """
    计算均方根误差(Root Mean Square Error)
    
    参数:
        y_true: 真实值数组，形状为(n_samples,)
        y_pred: 预测值数组，形状为(n_samples,)
    
    返回:
        float: 均方根误差值
    """
    # TODO: 实现RMSE计算公式
    return np.sqrt(np.mean((y_true - y_pred) ** 2))  # 计算误差平方的均值并开根号

def softmax(z):
    """
    计算softmax概率，将任意实数向量转换为概率分布
    
    参数:
        z: 输入矩阵，形状为(n_samples, n_classes)
    
    返回:
        softmax概率矩阵，形状与z相同
    """
    # TODO: 实现softmax函数
    z_max = np.max(z, axis=1, keepdims=True)  # 提取每行最大值防止指数运算溢出
    exp_z = np.exp(z - z_max)  # 减去最大值后计算指数
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 按行求和实现概率归一化


def cross_entropy_loss(y_true, y_prob):
    """
    计算交叉熵损失
    
    参数:
        y_true: 真实标签数组，形状为(n_samples,)，包含整数类别标签
        y_prob: 预测概率矩阵，形状为(n_samples, n_classes)
    
    返回:
        float: 平均交叉熵损失值
    """
    # TODO: 实现交叉熵损失计算
    n_samples = y_true.shape[0]
    y_prob = np.clip(y_prob, 1e-15, 1.0)  # 截断极小值防止log(0)计算崩溃
    correct_logprobs = -np.log(y_prob[np.arange(n_samples), y_true])  # 提取真实类别对应的负对数概率
    return np.mean(correct_logprobs)  # 求所有样本损失均值

# ===== 2. 加载数据 =====
def load_dataset(path='data.csv'):
    """
    从CSV文件加载数据集
    
    参数:
        path: 数据文件路径，默认为'data.csv'
    
    返回:
        X_features: 特征矩阵，形状为(n_samples, n_features)
        y_price: 房价连续值目标，形状为(n_samples,)
        y_class: 分类标签，形状为(n_samples,)
    """
    dataset = np.genfromtxt(path, delimiter=',', skip_header=1)
    X_features = dataset[:, :-2]
    y_price = dataset[:, -2]
    y_class = dataset[:, -1].astype(int)
    
    print(f"特征维度: {X_features.shape}, 房价目标: {y_price.shape}, 分类标签: {y_class.shape}")
    return X_features, y_price, y_class

# ===== 3. 计算梯度 =====
def compute_gradients_regression(X, y, w, b, reg_lambda=0.0):
    """
    计算线性回归的梯度(含L2正则化)
    
    参数:
        X: 特征矩阵，形状为(n_samples, n_features)
        y: 目标值，形状为(n_samples,)
        w: 权重向量，形状为(n_features,)
        b: 偏置项，标量
        reg_lambda: L2正则化系数，默认为0.0
    
    返回:
        dw: 权重梯度，形状与w相同
        db: 偏置梯度，标量
    """
    # TODO: 实现线性回归梯度计算
    n_samples = X.shape[0]
    error = X.dot(w) + b - y

    dw = (1 / n_samples) * X.T.dot(error) + reg_lambda * w
    db = (1 / n_samples) * np.sum(error)

    return dw, db

def compute_gradients_softmax(X, y, W, b, reg_lambda=0.0):
    """
    计算Softmax分类器的梯度(含L2正则化)
    
    参数:
        X: 特征矩阵，形状为(n_samples, n_features)
        y: 真实标签，形状为(n_samples,)
        W: 权重矩阵，形状为(n_features, n_classes)
        b: 偏置向量，形状为(n_classes,)
        reg_lambda: L2正则化系数，默认为0.0
    
    返回:
        dW: 权重梯度矩阵，形状与W相同
        db: 偏置梯度向量，形状与b相同
        loss: 当前损失值
    """
    # TODO: 实现Softmax梯度计算
    n_samples = X.shape[0]
    logits = X.dot(W) + b
    probs = softmax(logits)

    dz = probs.copy()
    dz[np.arange(n_samples), y] -= 1  # Softmax+CrossEntropy组合求导极简形式: P - Y

    dW = (1 / n_samples) * X.T.dot(dz) + reg_lambda * W
    db = (1 / n_samples) * np.sum(dz, axis=0)

    loss = cross_entropy_loss(y, probs) + 0.5 * reg_lambda * np.sum(W ** 2)

    return dW, db, loss

# ===== 4. 梯度下降训练线性回归 =====
def train_linear_regression(X_train, y_train, learning_rate=0.01, num_iterations=1000, reg_lambda=0.0):
    """
    使用梯度下降训练线性回归模型
    
    参数:
        X_train: 训练特征矩阵，形状为(n_samples, n_features)
        y_train: 训练目标值，形状为(n_samples,)
        learning_rate: 学习率，默认为0.01
        num_iterations: 迭代次数，默认为1000
        reg_lambda: L2正则化系数，默认为0.0
    
    返回:
        w: 训练后的权重向量，形状为(n_features,)
        b: 训练后的偏置项，标量
    """
    # 初始化参数
    w = np.zeros(X_train.shape[1])
    b = 0
    
    for i in range(num_iterations):
        # 计算梯度
        dw, db = compute_gradients_regression(X_train, y_train, w, b, reg_lambda)
        
        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 每1000次迭代打印损失
        if (i+1) % 1000 == 0:
            y_pred = X_train.dot(w) + b
            print(f"Iter {i+1}, RMSE: {rmse(y_train, y_pred):.4f}")
    
    return w, b

# ===== 5. 梯度下降训练Softmax分类 =====
def train_softmax(X_train, y_train, num_classes, learning_rate=0.01, num_iterations=1000, reg_lambda=0.0):
    """
    使用梯度下降训练Softmax分类器
    
    参数:
        X_train: 训练特征矩阵，形状为(n_samples, n_features)
        y_train: 训练标签，形状为(n_samples,)
        num_classes: 类别数量
        learning_rate: 学习率，默认为0.01
        num_iterations: 迭代次数，默认为1000
        reg_lambda: L2正则化系数，默认为0.0
    
    返回:
        W: 训练后的权重矩阵，形状为(n_features, num_classes)
        b: 训练后的偏置向量，形状为(num_classes,)
    """
    # 初始化参数
    W = np.zeros((X_train.shape[1], num_classes))
    b = np.zeros(num_classes)
    
    for i in range(num_iterations):
        # 计算梯度和损失
        dW, db, loss = compute_gradients_softmax(X_train, y_train, W, b, reg_lambda)
        
        # 更新参数
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # 每1000次迭代打印损失
        if (i+1) % 1000 == 0:
            print(f"Iter {i+1}, CrossEntropy Loss: {loss:.4f}")
    
    return W, b


def run_experiments(X_train, y_price_train, y_class_train, num_classes):
    """
    测试超参数对损失与模型效果的影响，并绘制中英双语曲线图
    """
    # 设置微软雅黑字体(好像只有这个能正常显示）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 回归任务学习率调小以防止梯度爆炸，分类任务保持原样
    lr_reg = [0.0001, 0.0005, 0.001, 0.002]
    lr_clf = [0.001, 0.005, 0.01, 0.05]
    num_iterations = 2000

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # === 1. 线性回归：RMSE下降曲线 ===
    for lr in lr_reg:
        w = np.zeros(X_train.shape[1])
        b = 0
        loss_hist = []
        for i in range(num_iterations):
            dw, db = compute_gradients_regression(X_train, y_price_train, w, b)
            w -= lr * dw
            b -= lr * db
            if i % 100 == 0:
                y_pred = X_train.dot(w) + b
                loss_hist.append(rmse(y_price_train, y_pred))
        axes[0].plot(range(0, num_iterations, 100), loss_hist, label=f'LR={lr}')

    axes[0].set_title('回归任务: RMSE下降曲线\n(Regression: RMSE vs Iterations)')
    axes[0].set_xlabel('迭代次数 (Iterations)')
    axes[0].set_ylabel('均方根误差 (RMSE)')
    axes[0].legend()

    # === 2. 分类任务：交叉熵损失下降 ===
    # === 3. 分类任务：准确率上升曲线 ===
    for lr in lr_clf:
        W = np.zeros((X_train.shape[1], num_classes))
        b = np.zeros(num_classes)
        loss_hist = []
        acc_hist = []
        for i in range(num_iterations):
            dW, db, loss = compute_gradients_softmax(X_train, y_class_train, W, b)
            W -= lr * dW
            b -= lr * db

            # 每100步记录一次Loss和Accuracy
            if i % 100 == 0:
                loss_hist.append(loss)
                logits = X_train.dot(W) + b
                probs = softmax(logits)
                preds = np.argmax(probs, axis=1)
                acc = np.mean(preds == y_class_train)
                acc_hist.append(acc)

        axes[1].plot(range(0, num_iterations, 100), loss_hist, label=f'LR={lr}')
        axes[2].plot(range(0, num_iterations, 100), acc_hist, label=f'LR={lr}')

    axes[1].set_title('分类任务: 交叉熵损失下降\n(Classification: Loss vs Iterations)')
    axes[1].set_xlabel('迭代次数 (Iterations)')
    axes[1].set_ylabel('交叉熵损失 (Cross Entropy Loss)')
    axes[1].legend()

    axes[2].set_title('分类任务: 准确率上升曲线\n(Classification: Accuracy vs Iterations)')
    axes[2].set_xlabel('迭代次数 (Iterations)')
    axes[2].set_ylabel('准确率 (Accuracy)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('week2_experiments.png', dpi=150)
    plt.show()

# ===== 6. 主程序 =====
def main():
    """
    主函数，执行完整的训练和评估流程
    """
    # 加载数据
    X, y_price, y_class = load_dataset('data_processed.csv')
    
    # 划分训练集和测试集
    X_train, X_test, y_price_train, y_price_test, y_class_train, y_class_test = train_test_split(
        X, y_price, y_class, test_size=0.25, random_state=42
    )

    # 获取类别数量
    num_classes = len(np.unique(y_class))

    # --- 新增：运行超参数对比实验并画图 ---
    print("\n=== 运行超参数实验并绘制曲线 ===")
    run_experiments(X_train, y_price_train, y_class_train, num_classes)


    print("\n=== 训练线性回归模型（房价预测） ===")
    # 训练线性回归模型
    w, b = train_linear_regression(
        X_train, y_price_train, 
        learning_rate=0.001, 
        num_iterations=10000, 
        reg_lambda=0.001
    )
    
    # 在测试集上评估回归模型
    y_pred_test = X_test.dot(w) + b
    print(f"测试集平均 RMSE: {rmse(y_price_test, y_pred_test):.4f}")

    print("\n=== 训练Softmax分类器（房价等级分类） ===")

    
    # 训练Softmax分类器
    W, b_class = train_softmax(
        X_train, y_class_train, 
        num_classes=num_classes, 
        learning_rate=0.01, 
        num_iterations=10000, 
        reg_lambda=0.01
    )

    # 在测试集上评估分类模型
    logits = X_test.dot(W) + b_class
    probs = softmax(logits)
    y_pred_class = np.argmax(probs, axis=1)
    acc = np.mean(y_pred_class == y_class_test)
    print(f"测试集分类准确率: {acc:.4f}")

if __name__ == '__main__':
    main()
