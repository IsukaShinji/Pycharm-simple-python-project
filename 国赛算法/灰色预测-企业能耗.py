import numpy as np
import math
from matplotlib import pyplot as plt


def ago(x0):
    """累加生成(AGO)函数:将原始序列累加，增强趋势性"""
    return np.cumsum(x0)


def adjacent_mean(x1):
    """计算紧邻均值序列:用于将离散数据与连续微分方程联系起来"""
    return (x1[1:] + x1[:-1]) / 2


def least_squares(x0, z1):
    """最小二乘法求解GM(1,1)模型参数a(发展系数)和b(灰作用量)"""
    # 构造Y矩阵(原始序列从第二个元素开始，形状为(n-1,1))
    Y = x0[1:].reshape(-1, 1)
    # 构造B矩阵(第一列为-z1，第二列为全1，形状为(n-1,2))
    B = np.hstack([-z1.reshape(-1, 1), np.ones_like(z1).reshape(-1, 1)])
    # 计算(B^T * B)的逆矩阵
    try:
        B_T_B = B.T @ B
        inv_B_T_B = np.linalg.inv(B_T_B)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵不可逆，请检查数据是否符合要求(单调、正数)")
    # 求解参数θ = [a, b]^T
    theta = inv_B_T_B @ B.T @ Y
    return theta[0, 0], theta[1, 0]


def gm11_model(a, b, x1_0):
    """构建GM(1,1)累加序列预测模型"""

    def model(t):
        if t < 1:
            raise ValueError("t必须≥1(1对应第一个数据点)")
        return (x1_0 - b / a) * math.exp(-a * (t - 1)) + b / a

    return model


def predict_original(gm_model, n):
    """从累加序列预测原始序列(逆累加生成,IAGO)"""
    # 预测累加序列(t=1到n)
    x1_hat = np.array([gm_model(t) for t in range(1, n + 1)])
    # 逆累加:x0_hat[0] = x1_hat[0], x0_hat[k] = x1_hat[k] - x1_hat[k-1](k≥1)
    x0_hat = np.concatenate([[x1_hat[0]], np.diff(x1_hat)])
    return x0_hat


def residual_correction(x0, x0_hat):
    """残差修正:用残差序列建立GM(1,1)模型，修正预测值"""
    # 计算残差(从第二个点开始，第一个点残差为0)
    e0 = x0 - x0_hat
    e0 = e0[1:]  # 残差序列长度为len(x0)-1
    # 残差全0，无需修正
    if np.all(e0 == 0):
        print("残差序列全为0，无需修正")
        return x0_hat, None
    # 对残差做AGO和紧邻均值
    e1 = ago(e0)
    z_e1 = adjacent_mean(e1)
    # 求解残差模型参数
    try:
        a_e, b_e = least_squares(e0, z_e1)
    except ValueError as e:
        print(f"残差模型无法建立:{e}，无需修正")
        return x0_hat, None
    # 构建残差累加预测模型
    residual_model = gm11_model(a_e, b_e, e1[0])
    # 预测残差原始值(对应原始序列t=2到t=len(x0))
    e1_hat = np.array([residual_model(t) for t in range(1, len(e0) + 1)])
    e0_hat = np.concatenate([[e1_hat[0]], np.diff(e1_hat)])
    # 修正原始预测值(t≥2)
    x0_hat_corrected = x0_hat.copy()
    x0_hat_corrected[1:] += e0_hat
    return x0_hat_corrected, residual_model


def model_test(x0, x0_hat):
    """模型精度检验:计算相对误差、后验差比C、小误差概率P"""
    # 残差序列(从第二个点开始)
    e0 = x0 - x0_hat
    e0 = e0[1:]
    n = len(e0)
    # 相对误差(%)
    relative_errors = [abs(e / x0[i + 1]) * 100 for i, e in enumerate(e0)]
    # 后验差比C:残差标准差/原始数据标准差
    if n == 0:
        C = 0
        P = 0
    else:
        mu_e = np.mean(e0)
        sigma_e = np.std(e0, ddof=1)  # 样本标准差
        mu_x = np.mean(x0)
        sigma_x = np.std(x0, ddof=1)
        C = sigma_e / sigma_x if sigma_x != 0 else 0
        # 小误差概率P:|e_k - μ_e| < 0.6745σ_x的比例
        threshold = 0.6745 * sigma_x
        P = sum(abs(e - mu_e) < threshold for e in e0) / n
    # 输出检验结果
    print("\n===模型精度检验===")
    print(f"相对误差(%):{[round(re, 2) for re in relative_errors]}")
    print(f"后验差比C:{round(C, 2)}(C<0.35为优秀)")
    print(f"小误差概率P:{round(P, 2)}(P>0.95为优秀)")
    # 判断模型等级
    if C < 0.35 and P > 0.95:
        print("模型等级:优秀")
    elif 0.35 <= C < 0.5 and 0.8 <= P <= 0.95:
        print("模型等级:良好")
    elif 0.5 <= C < 0.65 and 0.7 <= P < 0.8:
        print("模型等级:可用")
    else:
        print("模型等级:不可用")
    return relative_errors, C, P


if __name__ == "__main__":
    # 1. 输入原始数据(某企业2024年1-4月能源消耗，吨标准煤)
    x0 = np.array([120, 150, 190, 240])
    months = np.arange(1, len(x0) + 1)  # 1-4月
    print(f"原始数据(1-4月):{x0}")

    # 2. 累加生成(AGO):增强趋势性
    x1 = ago(x0)
    print(f"累加序列:{x1.round(2)}")

    # 3. 计算紧邻均值:连接离散与连续
    z1 = adjacent_mean(x1)
    print(f"紧邻均值序列:{z1.round(2)}")

    # 4. 求解GM(1,1)参数a和b
    a, b = least_squares(x0, z1)
    print(f"\n模型参数:发展系数a={a.round(4)},灰作用量b={b.round(4)}")

    # 5. 构建累加预测模型
    gm_model = gm11_model(a, b, x1[0])
    print(
        f"累加序列模型:x1(t) = ({x1[0].round(2)} - {b.round(2)}/{a.round(2)}) * exp(-{a.round(2)}*(t-1)) + {b.round(2)}/{a.round(2)}")

    # 6. 预测原始序列(1-4月拟合值)
    x0_hat = predict_original(gm_model, len(x0))
    print(f"\n原始序列拟合值(1-4月):{x0_hat.round(2)}")

    # 7. 残差修正:提高预测精度
    x0_hat_corrected, residual_model = residual_correction(x0, x0_hat)
    if residual_model is not None:
        print(f"修正后拟合值(1-4月):{x0_hat_corrected.round(2)}")

    # 8. 模型精度检验
    model_test(x0, x0_hat_corrected if residual_model is not None else x0_hat)

    # 9. 预测5月能源消耗(未来1步)
    n_predict = 1  # 预测1个月(5月)
    total_months = len(x0) + n_predict  # 1-5月
    # 预测累加序列(1-5月)
    x1_hat_all = np.array([gm_model(t) for t in range(1, total_months + 1)])
    # 逆累加得到原始预测(1-5月)
    x0_hat_all = predict_original(gm_model, total_months)
    # 5月预测值(原模型)
    x0_hat_may = x0_hat_all[-n_predict:]
    print(f"\n5月能源消耗预测(原模型):{x0_hat_may.round(2)}吨标准煤")

    # 10. 残差修正未来预测(如果残差模型存在)
    if residual_model is not None:
        # 残差序列对应原始序列t=2-4，未来t=5对应残差模型t=4
        e1_hat_may = residual_model(4)  # 残差累加预测(t=4)
        e1_prev = ago(x0[1:] - x0_hat[1:])[-1]  # 残差累加前值(t=3)
        e0_hat_may = e1_hat_may - e1_prev  # 残差原始预测(t=5)
        # 修正5月预测值
        x0_hat_may_corrected = x0_hat_may + e0_hat_may
        print(f"5月能源消耗预测(残差修正后):{x0_hat_may_corrected.round(2)}吨标准煤")

    # 11. 可视化结果(帮助理解趋势)
    plt.figure(figsize=(10, 6))
    plt.plot(months, x0, 'o-', label='原始数据', markersize=8)
    plt.plot(months, x0_hat_corrected if residual_model is not None else x0_hat, 's-', label='拟合值', markersize=8)
    plt.plot([5], x0_hat_may_corrected if residual_model is not None else x0_hat_may, '^-', label='预测值',
             markersize=10, color='red')
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('能源消耗(吨标准煤)', fontsize=12)
    plt.title('GM(1,1)模型能源消耗预测', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(1, 6))  # 显示1-5月
    plt.show()