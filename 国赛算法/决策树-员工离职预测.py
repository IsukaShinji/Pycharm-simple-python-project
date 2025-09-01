# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # 用于categorical特征转数值
from sklearn.model_selection import train_test_split  # 划分训练集/测试集
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 决策树模型及可视化
from sklearn.metrics import accuracy_score  # 计算准确率
import matplotlib.pyplot as plt  # 绘图库

def prepare_data():
    """
    生成并预处理模拟数据(修正后的数据:使“是否加班”与“是否离职”不完全相关)
    返回:特征矩阵X,目标向量y,特征名称feature_names,类别名称class_names
    """
    # 1. 模拟员工离职数据(修正:调整2个样本的离职状态,使特征与目标不完全相关)
    data = pd.DataFrame({
        '工作年限(年)': [1, 3, 2, 5, 1, 4, 2, 3, 1, 5],
        '月薪(元)': [5000, 8000, 6000, 10000, 5500, 9000, 6500, 7000, 4500, 11000],
        '是否加班': ['是', '否', '是', '否', '是', '否', '否', '是', '是', '否'],
        '团队氛围评分(1-5)': [3, 4, 2, 5, 3, 4, 3, 2, 1, 5],
        '是否离职': ['是', '否', '是', '是', '是', '否', '否', '否', '是', '否']  # 修正:索引3(不加班)离职,索引7(加班)不离职
    })

    # 2. 处理categorical特征(将文字转为数值,决策树需要数值输入)
    le_overtime = LabelEncoder()  # 处理“是否加班”
    data['是否加班'] = le_overtime.fit_transform(data['是否加班'])  # 否→0,是→1
    le_leave = LabelEncoder()  # 处理“是否离职”(目标变量)
    data['是否离职'] = le_leave.fit_transform(data['是否离职'])  # 否→0,是→1

    # 3. 划分特征(X)和目标(y)
    feature_names = ['工作年限(年)', '月薪(元)', '是否加班', '团队氛围评分(1-5)']
    X = data[feature_names]
    y = data['是否离职']

    # 4. 返回预处理后的数据
    return X, y, feature_names, le_leave.classes_  # classes_是类别名称(['否', '是'])

def train_and_evaluate(X, y, feature_names, max_depth=3):
    """
    训练决策树模型并评估性能
    返回:训练好的模型clf,测试集准确率accuracy
    参数:X(特征)、y(目标)、feature_names(特征名称)、max_depth(树最大深度,防止过拟合)
    """
    # 1. 划分训练集(80%)和测试集(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # random_state固定随机种子,结果可重复
    )

    # 2. 初始化决策树分类器(CART算法,使用基尼指数)
    clf = DecisionTreeClassifier(
        criterion='gini',  # 分裂准则:基尼指数(衡量混乱程度)
        max_depth=max_depth,  # 预剪枝:限制树的最大深度,防止过拟合
        random_state=42
    )

    # 3. 训练模型
    clf.fit(X_train, y_train)

    # 4. 用测试集预测并评估准确率
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. 输出结果
    print(f"测试集准确率:{accuracy:.2f}")  # 打印准确率(保留两位小数)
    return clf, accuracy

def plot_feature_importance(clf, feature_names):
    """
    可视化特征重要性(决策树认为哪些特征对预测最有用)
    参数:clf(训练好的模型)、feature_names(特征名称)
    """
    # 1. 获取特征重要性(数值越大,特征越重要)
    importances = clf.feature_importances_

    # 2. 绘制柱状图
    plt.figure(figsize=(10, 6))  # 设置图的大小
    plt.bar(
        feature_names,  # x轴:特征名称
        importances,  # y轴:重要性得分
        color='skyblue'  # 柱子颜色
    )
    plt.xlabel('特征名称', fontsize=12)  # x轴标签
    plt.ylabel('重要性得分', fontsize=12)  # y轴标签
    plt.title('员工离职预测特征重要性', fontsize=14)  # 图标题
    plt.xticks(rotation=45)  # 旋转x轴标签,避免重叠
    plt.tight_layout()  # 自动调整布局,防止标签截断
    plt.show()  # 显示图

def plot_tree_structure(clf, feature_names, class_names):
    """
    可视化决策树结构(直观看到决策过程)
    参数:clf(训练好的模型)、feature_names(特征名称)、class_names(类别名称,如['否', '是'])
    """
    # 1. 设置图的大小
    plt.figure(figsize=(20, 12))

    # 2. 绘制决策树
    plot_tree(
        clf,  # 训练好的模型
        feature_names=feature_names,  # 特征名称(显示在节点上)
        class_names=class_names,  # 类别名称(显示在叶节点上,如['否', '是'])
        filled=True,  # 用颜色填充节点(颜色越深,节点越纯)
        rounded=True,  # 节点边框用圆角
        fontsize=10,  # 字体大小
        proportion=True  # 显示样本比例(如“3/5”表示该节点有3个正例,5个总样本)
    )

    # 3. 设置图标题
    plt.title('员工离职预测决策树结构', fontsize=14)

    # 4. 显示图
    plt.show()

# 主程序(入口函数)
if __name__ == "__main__":
    # 1. 准备数据
    X, y, feature_names, class_names = prepare_data()
    print("数据预处理完成!")

    # 2. 训练模型并评估
    print("\n=== 模型训练与评估===")
    clf, accuracy = train_and_evaluate(X, y, feature_names, max_depth=3)

    # 3. 特征重要性分析
    print("\n=== 特征重要性评估===")
    plot_feature_importance(clf, feature_names)

    # 4. 决策树结构可视化
    print("\n=== 决策树结构可视化===")
    plot_tree_structure(clf, feature_names, class_names)