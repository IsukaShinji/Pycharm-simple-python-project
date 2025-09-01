import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. 模拟训练数据(带噪声的线性数据)
# 真实模型:y = 2x + 1（真实权重w=2，偏置b=1）
# ------------------------------
x_data = np.array([1, 2, 3, 4, 5])  # 输入x
# 生成带±0.1噪声的实际值，模拟真实测量数据
y_true = 2 * x_data + 1 + np.random.uniform(-0.1, 0.1, size=x_data.shape)
print(f"实际值y_true: {y_true.round(2)}")
print("模拟训练数据:")
print(f"输入x: {x_data}")


# ------------------------------
# 2. 定义适应度函数(均方误差MSE，越小表示模型越优)
# ------------------------------
def calculate_mse(w, b, x, y):
    """
    计算线性模型y = w*x + b 的均方误差(MSE)
    参数:
        w: 权重(float)
        b: 偏置(float)
        x: 输入数据(numpy数组)
        y: 实际值(numpy数组)
    返回:
        mse: 均方误差(float)
    """
    y_pred = w * x + b  # 模型预测值
    mse = np.mean((y - y_pred) ** 2)  # 均方误差计算公式
    return mse


# ------------------------------
# 3. 初始化PSO参数（小白经典参数，平衡性能与计算量）
# ------------------------------
n_particles = 30  # 粒子数量（候选解的数量）
w_inertia = 0.8  # 惯性权重（控制探索新区域与细化当前区域的平衡）
c1 = 2  # 认知学习因子（重视粒子自身经验）
c2 = 2  # 社会学习因子（重视群体经验）
max_iter = 100  # 最大迭代次数（确保算法收敛）

# 位置范围（w和b的可行域，根据问题实际范围调整）
w_min, w_max = 0, 4  # 权重w的范围（真实值为2）
b_min, b_max = 0, 2  # 偏置b的范围（真实值为1）

# 速度范围（限制粒子移动速度，避免"飞过头"或飞出可行域）
v_w_min, v_w_max = -1, 1  # 权重w的速度范围
v_b_min, v_b_max = -0.5, 0.5  # 偏置b的速度范围

# ------------------------------
# 4. 初始化粒子群（每个粒子代表一组(w,b)候选解）
# ------------------------------
particles = []
for _ in range(n_particles):
    # 随机初始化位置（w, b）：在可行域内取值
    w = random.uniform(w_min, w_max)
    b = random.uniform(b_min, b_max)

    # 随机初始化速度（v_w, v_b）：在速度范围内取值
    v_w = random.uniform(v_w_min, v_w_max)
    v_b = random.uniform(v_b_min, v_b_max)

    # 初始个体最优(pbest)：初始时为当前位置（无历史数据）
    pbest_w = w
    pbest_b = b
    # 计算初始pbest的适应度（MSE）
    pbest_fitness = calculate_mse(pbest_w, pbest_b, x_data, y_true)

    # 将粒子信息存入字典（便于管理参数）
    particles.append({
        'w': w,  # 当前权重值
        'b': b,  # 当前偏置值
        'v_w': v_w,  # 当前权重的速度
        'v_b': v_b,  # 当前偏置的速度
        'pbest_w': pbest_w,  # 个体最优权重
        'pbest_b': pbest_b,  # 个体最优偏置
        'pbest_fitness': pbest_fitness  # 个体最优适应度（MSE）
    })

# ------------------------------
# 5. 初始化全局最优(gbest)：所有粒子pbest中的最优解（MSE最小）
# ------------------------------
# 先以第一个粒子的pbest为初始gbest
gbest_w = particles[0]['pbest_w']
gbest_b = particles[0]['pbest_b']
gbest_fitness = particles[0]['pbest_fitness']

# 遍历所有粒子，找到真正的初始gbest（MSE最小）
for p in particles:
    if p['pbest_fitness'] < gbest_fitness:
        gbest_fitness = p['pbest_fitness']
        gbest_w = p['pbest_w']
        gbest_b = p['pbest_b']

# ------------------------------
# 6. 迭代优化（PSO核心过程：更新速度→位置→pbest→gbest）
# ------------------------------
fitness_history = []  # 记录每代gbest的MSE，用于后续可视化
for iter in range(max_iter):
    # 遍历每个粒子，更新速度、位置、个体最优
    for p in particles:
        # 生成0~1的随机数r1、r2，增加算法随机性（避免局部最优）
        r1 = random.random()
        r2 = random.random()

        # 1. 更新速度（核心公式：惯性项 + 认知项 + 社会项）
        # 权重w的速度更新
        p['v_w'] = (w_inertia * p['v_w'] +
                    c1 * r1 * (p['pbest_w'] - p['w']) +  # 认知项：向自身pbest靠拢
                    c2 * r2 * (gbest_w - p['w']))  # 社会项：向群体gbest靠拢
        # 偏置b的速度更新（逻辑同上）
        p['v_b'] = (w_inertia * p['v_b'] +
                    c1 * r1 * (p['pbest_b'] - p['b']) +
                    c2 * r2 * (gbest_b - p['b']))

        # 限制速度在设定范围内（避免速度过大导致粒子"飞过头"）
        p['v_w'] = max(min(p['v_w'], v_w_max), v_w_min)
        p['v_b'] = max(min(p['v_b'], v_b_max), v_b_min)

        # 2. 更新位置（新位置 = 当前位置 + 新速度）
        p['w'] += p['v_w']
        p['b'] += p['v_b']

        # 限制位置在可行域内（避免粒子飞出有效参数范围）
        p['w'] = max(min(p['w'], w_max), w_min)
        p['b'] = max(min(p['b'], b_max), b_min)

        # 3. 更新个体最优(pbest)：若当前位置MSE更小，则更新
        current_fitness = calculate_mse(p['w'], p['b'], x_data, y_true)
        if current_fitness < p['pbest_fitness']:
            p['pbest_w'] = p['w']
            p['pbest_b'] = p['b']
            p['pbest_fitness'] = current_fitness

    # 4. 更新全局最优(gbest)：遍历所有粒子pbest，找MSE最小的
    for p in particles:
        if p['pbest_fitness'] < gbest_fitness:
            gbest_fitness = p['pbest_fitness']
            gbest_w = p['pbest_w']
            gbest_b = p['pbest_b']

    # 记录当前代的gbest MSE（用于可视化收敛过程）
    fitness_history.append(gbest_fitness)

    # 每10代打印一次进度，便于观察优化过程
    if (iter + 1) % 10 == 0:
        print(f"迭代次数:{iter + 1:2d}, 当前gbest MSE:{gbest_fitness:.4f}")

# ------------------------------
# 7. 输出优化结果
# ------------------------------
print("\n" + "-" * 50)
print("PSO 优化完成!")
print(f"最优权重w: {gbest_w:.4f}（真实值:2）")
print(f"最优偏置b: {gbest_b:.4f}（真实值:1）")
print(f"最小均方误差(MSE): {gbest_fitness:.4f}")
print("-" * 50 + "\n")

# ------------------------------
# 8. 可视化结果（直观观察收敛过程与模型拟合效果）
# ------------------------------
# 图1：迭代次数与MSE的关系（验证算法是否收敛）
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), fitness_history, color='blue', linewidth=2, label='gbest MSE')
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('均方误差(MSE)', fontsize=12)
plt.title('PSO 优化线性模型: 迭代次数与MSE关系', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 图2：模型拟合效果（验证优化后的模型是否贴近实际数据）
plt.figure(figsize=(10, 6))
# 绘制实际数据点（红色散点）
plt.scatter(x_data, y_true, color='red', s=100, label='实际数据(带噪声)')
# 生成平滑的x轴数据（用于绘制拟合曲线）
x_plot = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
# 用最优参数计算预测值
y_plot = gbest_w * x_plot + gbest_b
# 绘制拟合曲线（蓝色实线）
plt.plot(x_plot, y_plot, color='blue', linewidth=2,
         label=f'拟合模型: y={gbest_w:.2f}x + {gbest_b:.2f}')
plt.xlabel('输入x', fontsize=12)
plt.ylabel('输出y', fontsize=12)
plt.title('PSO 优化线性模型: 拟合效果', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()