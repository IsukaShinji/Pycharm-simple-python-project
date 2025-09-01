import numpy as np
import random

def generate_stock_data(n_days, true_transition, initial_state=0):
    """
    模拟股票每日走势数据（生成状态序列）
    参数：
        n_days (int)：模拟天数（至少1天）
        true_transition (np.ndarray)：真实状态转移矩阵（3×3，行=当前状态，列=下一状态）
        initial_state (int)：初始状态（0=涨，1=跌，2=平）
    返回：
        list：长度为n_days的状态序列（元素为0/1/2）
    """
    if n_days < 1:
        raise ValueError("天数必须至少为1")
    if true_transition.shape != (3, 3):
        raise ValueError("转移矩阵必须是3×3格式")

    states = [initial_state]  # 初始化状态序列
    for _ in range(n_days - 1):
        current_state = states[-1]
        # 按当前状态的转移概率随机选择下一状态
        next_state = random.choices(
            population=[0, 1, 2],
            weights=true_transition[current_state],
            k=1
        )[0]
        states.append(next_state)
    return states


def compute_transition_matrix(states):
    """
    根据状态序列计算估计的状态转移矩阵
    参数：
        states (list)：股票走势状态序列（元素为0/1/2）
    返回：
        np.ndarray：3×3的转移矩阵（行=当前状态，列=下一状态）
    """
    n = len(states)
    if n < 2:
        raise ValueError("状态序列长度至少为2（需至少1次转移）")

    # 初始化计数矩阵（统计“当前状态→下一状态”的次数）
    counts = np.zeros((3, 3), dtype=int)
    for t in range(1, n):
        prev_state = states[t - 1]
        curr_state = states[t]
        counts[prev_state][curr_state] += 1

    # 计数转换为概率（每行除以该行总转移次数）
    transition_matrix = np.zeros((3, 3))
    for i in range(3):
        total = counts[i].sum()
        if total > 0:
            transition_matrix[i] = counts[i] / total
        else:
            # 若状态i未出现，默认均匀分布（实际场景中极少发生）
            transition_matrix[i] = np.array([1 / 3, 1 / 3, 1 / 3])
    return transition_matrix


def compute_stationary_distribution(transition_matrix, max_iter=1000, tol=1e-6):
    """
    用迭代法计算马尔可夫链的平稳分布（长期趋势）
    参数：
        transition_matrix (np.ndarray)：3×3的状态转移矩阵
        max_iter (int)：最大迭代次数（防止无限循环）
        tol (float)：收敛阈值（两次迭代差值小于tol则停止）
    返回：
        np.ndarray：平稳分布（行向量，对应“涨、跌、平”的长期概率）
    """
    if transition_matrix.shape != (3, 3):
        raise ValueError("转移矩阵必须是3×3格式")

    # 初始化平稳分布为均匀分布
    pi = np.array([1 / 3, 1 / 3, 1 / 3])
    for _ in range(max_iter):
        old_pi = pi.copy()
        pi = pi @ transition_matrix  # 迭代更新分布（行向量×转移矩阵）
        # 检查收敛性（L2范数小于阈值则停止）
        if np.linalg.norm(pi - old_pi) < tol:
            break
    return pi


if __name__ == "__main__":
    # ==================== 1. 设定模拟参数 ====================
    # 真实状态转移矩阵（行=当前状态：0=涨，1=跌，2=平；列=下一状态）
    true_P = np.array([
        [0.5, 0.3, 0.2],  # 涨→涨、跌、平的概率
        [0.4, 0.4, 0.2],  # 跌→涨、跌、平的概率
        [1 / 3, 1 / 3, 1 / 3]  # 平→涨、跌、平的概率（精确值1/3）
    ])
    n_days = 30  # 模拟30天的股票数据
    initial_state = 0  # 第1天初始状态为“涨”（0=涨）
    state_labels = {0: "涨", 1: "跌", 2: "平"}  # 状态文字映射

    # ==================== 2. 模拟股票走势数据 ====================
    print(f"=== 模拟{n_days}天的股票走势数据 ===")
    states = generate_stock_data(n_days, true_P, initial_state)
    for day in range(n_days):
        print(f"第{day + 1}天：{state_labels[states[day]]}")

    # ==================== 3. 计算估计的状态转移矩阵 ====================
    print("\n=== 估计的状态转移矩阵 ===")
    estimated_P = compute_transition_matrix(states)
    print("行：当前状态（涨、跌、平）；列：下一状态（涨、跌、平）")
    print(np.round(estimated_P, 3))  # 保留3位小数打印

    # ==================== 4. 计算平稳分布（长期趋势） ====================
    print("\n=== 平稳分布（长期趋势） ===")
    stationary_pi = compute_stationary_distribution(estimated_P)
    print(f"涨的长期概率：{stationary_pi[0]:.4f}")
    print(f"跌的长期概率：{stationary_pi[1]:.4f}")
    print(f"平的长期概率：{stationary_pi[2]:.4f}")

    # ==================== 5. 预测未来状态（当前状态=涨） ====================
    print("\n=== 未来状态预测（当前状态为“涨”） ===")
    current_state = 0  # 当前状态为“涨”
    # 明天的状态分布（转移矩阵的当前行）
    tomorrow_dist = estimated_P[current_state]
    print(f"明天：涨{tomorrow_dist[0]:.4f}、跌{tomorrow_dist[1]:.4f}、平{tomorrow_dist[2]:.4f}")
    # 后天的状态分布（明天分布×转移矩阵）
    day_after_tomorrow_dist = tomorrow_dist @ estimated_P
    print(
        f"后天：涨{day_after_tomorrow_dist[0]:.4f}、跌{day_after_tomorrow_dist[1]:.4f}、平{day_after_tomorrow_dist[2]:.4f}")
    # 大后天的状态分布（后天分布×转移矩阵）
    third_day_dist = day_after_tomorrow_dist @ estimated_P
    print(f"大后天：涨{third_day_dist[0]:.4f}、跌{third_day_dist[1]:.4f}、平{third_day_dist[2]:.4f}")