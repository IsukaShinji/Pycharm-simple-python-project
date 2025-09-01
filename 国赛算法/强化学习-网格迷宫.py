import numpy as np
import random

# 1. 迷宫参数定义（可根据需求修改）
# ------------------------------
GRID_SIZE = 3  # 迷宫尺寸：3x3网格
ACTIONS = [0, 1, 2, 3]  # 动作空间：0=上，1=下，2=左，3=右
START_STATE = (0, 0)  # 起点：(行, 列)
END_STATE = (2, 2)  # 终点：(行, 列)
WALL_STATE = (1, 1)  # 墙壁：撞到墙壁会触发惩罚


# 2. 核心函数定义
def init_q_table():
    """
    初始化Q表：记录每个状态-动作对的价值（未来总奖励期望）
    形状：(GRID_SIZE, GRID_SIZE, 动作数量)，初始值全为0（智能体初始对环境陌生）
    """
    return np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))


def choose_action(state, q_table, epsilon):
    """
    根据ε-greedy策略选择动作（平衡探索与利用）
    参数：
        state: 当前状态(行, 列)
        q_table: Q表（状态-动作价值表）
        epsilon: 探索率（0≤ε≤1，ε越大越倾向探索）
    返回：
        action: 选择的动作（0-3）
    """
    row, col = state
    # 以概率ε随机选动作（探索未知路径）
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    # 以概率1-ε选当前Q值最大的动作（利用已知最优路径）
    else:
        q_values = q_table[row][col]
        max_q = np.max(q_values)
        # 若有多个最大Q值，随机选一个（避免单一路径依赖）
        max_actions = [a for a in ACTIONS if q_values[a] == max_q]
        return random.choice(max_actions)


def take_action(state, action):
    """
    执行动作，返回下一个状态和即时奖励（环境反馈）
    参数：
        action: 选择的动作（0-3）
        state: 当前状态(行, 列)
    返回：
        next_state: 下一个状态(行, 列)
        reward: 即时奖励（正数鼓励，负数惩罚）
    """
    row, col = state
    # 动作对应的行/列变化（上：行-1，下：行+1，左：列-1，右：列+1）
    dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
    # 限制下一个状态在迷宫范围内（避免出界）
    next_row = max(0, min(GRID_SIZE - 1, row + dr))
    next_col = max(0, min(GRID_SIZE - 1, col + dc))
    next_state = (next_row, next_col)

    # 奖励规则设计（核心：引导智能体走正确路径）
    if next_state == WALL_STATE:
        return state, -1  # 撞墙：奖励-1，状态不变
    elif next_state == END_STATE:
        return next_state, 10  # 到达终点：奖励+10（最大鼓励）
    else:
        return next_state, 0  # 走通道：奖励0（中性）


def train_q_learning(q_table, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.1):
    """
    训练Q-Learning智能体（核心逻辑：试错更新Q表）
    参数：
        q_table: 初始化的Q表
        episodes: 训练轮数（每轮从起点到终点为1轮）
        alpha: 学习率（0<α≤1，控制Q值更新幅度，α越小越稳定）
        gamma: 折扣因子（0<γ≤1，控制未来奖励的权重，γ越大越重视长远收益）
        epsilon: 初始探索率（0.5表示50%概率探索）
        min_epsilon: 探索率最小值（避免ε过小导致无法探索）
        epsilon_decay: 探索率衰减率（每轮后ε减小，逐渐从探索转向利用）
    返回：
        trained_q_table: 训练后的Q表（记录最优状态-动作价值）
    """
    for episode in range(episodes):
        current_state = START_STATE  # 每轮从起点开始
        done = False  # 是否到达终点的标记
        while not done:
            # 1. 选动作（ε-greedy）
            action = choose_action(current_state, q_table, epsilon)
            # 2. 执行动作（环境反馈）
            next_state, reward = take_action(current_state, action)
            # 3. 更新Q表（贝尔曼方程：当前价值=即时奖励+未来最大价值）
            row, col = current_state
            next_row, next_col = next_state
            # 当前状态-动作的Q值
            q_current = q_table[row][col][action]
            # 下一个状态的最大Q值（智能体未来会选最优动作）
            q_next_max = np.max(q_table[next_row][next_col])
            # 贝尔曼方程更新
            q_table[row][col][action] = q_current + alpha * (reward + gamma * q_next_max - q_current)
            # 4. 切换到下一个状态
            current_state = next_state
            # 5. 判断是否到达终点（结束本轮训练）
            if current_state == END_STATE:
                done = True
        # 探索率衰减（训练后期减少探索，专注利用）
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        # 每100轮输出训练进度（方便观察）
        if (episode + 1) % 100 == 0:
            print(f"训练进度：{episode+1}/{episodes} 轮，当前探索率：{epsilon:.2f}")
    return q_table


def test_optimal_path(q_table):
    """
    测试训练后的Q表，输出从起点到终点的最优路径（纯利用，不探索）
    参数：
        q_table: 训练后的Q表
    返回：
        path: 最优路径（状态列表，如[(0,0), (0,1), ..., (2,2)]）
    """
    path = [START_STATE]  # 路径从起点开始
    current_state = START_STATE
    while current_state != END_STATE:
        row, col = current_state
        q_values = q_table[row][col]
        # 选当前状态Q值最大的动作（ε=0，纯利用）
        max_q = np.max(q_values)
        max_actions = [a for a in ACTIONS if q_values[a] == max_q]
        action = random.choice(max_actions)
        # 执行动作（不需要奖励，只关心路径）
        next_state, _ = take_action(current_state, action)
        # 添加到路径
        path.append(next_state)
        # 切换状态
        current_state = next_state
    return path


# 3. 主程序（执行入口）
if __name__ == "__main__":
    # 步骤1：初始化Q表（全0）
    q_table = init_q_table()
    print("初始化后的Q表（起点(0,0)的初始Q值）：")
    print(q_table[0][0])  # 输出起点(0,0)的4个动作Q值（初始全0）

    # 步骤2：训练Q-Learning智能体（关键参数可修改）
    print("\n开始训练...")
    trained_q_table = train_q_learning(
        q_table=q_table,
        episodes=1000,  # 训练1000轮（足够收敛）
        alpha=0.1,      # 学习率（0.1为常用值）
        gamma=0.9,      # 折扣因子（重视长远收益）
        epsilon=0.5,    # 初始探索率（50%探索）
        epsilon_decay=0.001,  # 探索率衰减率（每轮减少0.1%）
        min_epsilon=0.1       # 最小探索率（10%）
    )
    print("训练完成！")

    # 步骤3：输出训练后的Q表（关键状态）
    print("\n训练后的Q表（关键状态，保留2位小数）：")
    print(f"起点(0,0)的Q值：{trained_q_table[0][0].round(2)}")    # 右动作Q值最大
    print(f"中间状态(0,1)的Q值：{trained_q_table[0][1].round(2)}")# 右动作Q值最大
    print(f"中间状态(0,2)的Q值：{trained_q_table[0][2].round(2)}")# 下动作Q值最大
    print(f"中间状态(1,2)的Q值：{trained_q_table[1][2].round(2)}")# 下动作Q值最大
    print(f"终点(2,2)的Q值：{trained_q_table[2][2].round(2)}")    # 终点无动作，全0

    # 步骤4：测试最优路径（输出从起点到终点的最优路线）
    print("\n测试最优路径...")
    optimal_path = test_optimal_path(trained_q_table)
    print(f"从起点{START_STATE}到终点{END_STATE}的最优路径：")
    print("→".join([str(state) for state in optimal_path]))  # 输出路径（如(0,0)→(0,1)→...→(2,2)）