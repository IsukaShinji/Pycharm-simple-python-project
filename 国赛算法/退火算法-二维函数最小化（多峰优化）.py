import random
import math
def rastrigin(x, y):
    """
    二维Rastrigin函数（目标函数，需最小化）
    参数：
        x: 自变量x
        y: 自变量y
    返回：
        函数值f(x,y)
    """
    return 20 + x ** 2 + y ** 2 - 10 * math.cos(2 * math.pi * x) - 10 * math.cos(2 * math.pi * y)

def simulated_annealing(T0=100, alpha=0.9, L=100, T_end=1e-5):
    """
    模拟退火算法实现（解决二维Rastrigin函数最小化问题）
    参数：
        T0: 初始温度（默认100，越大探索越充分）
        alpha: 温度衰减系数（默认0.9，0.8-0.95之间，越小降温越快）
        L: 每个温度下的迭代次数（默认100，越大搜索越充分）
        T_end: 停止温度（默认1e-5，越小收敛越彻底）
    返回：
        best_solution: 全局最优解(x,y)
        best_value: 全局最优目标函数值
    """
    # 1. 初始化当前解（随机生成x,y ∈ [-5.12, 5.12]）
    x = random.uniform(-5.12, 5.12)
    y = random.uniform(-5.12, 5.12)
    current_sol = (x, y)  # 当前解（元组存储）
    current_val = rastrigin(x, y)  # 当前解的目标函数值

    # 2. 初始化最优解（初始时最优解=当前解）
    best_sol = current_sol
    best_val = current_val

    # 3. 温度循环（从高温降至停止温度）
    T = T0
    while T > T_end:
        # 4. 每个温度下的迭代（充分搜索当前温度的解空间）
        for _ in range(L):
            # a. 生成新解：当前解附近正态分布扰动（温度越高，扰动越大）
            new_x = current_sol[0] + random.gauss(0, T ** 0.5)
            new_y = current_sol[1] + random.gauss(0, T ** 0.5)

            # b. 截断新解：确保x,y在定义域[-5.12, 5.12]内
            new_x = max(min(new_x, 5.12), -5.12)
            new_y = max(min(new_y, 5.12), -5.12)
            new_sol = (new_x, new_y)
            new_val = rastrigin(new_x, new_y)

            # c. 计算解的优劣差（新解 - 当前解）
            delta_f = new_val - current_val

            # d. Metropolis准则：判断是否接受新解
            if delta_f <= 0:
                # 新解更优，直接接受
                current_sol = new_sol
                current_val = new_val
                # 更新全局最优解
                if new_val < best_val:
                    best_sol = new_sol
                    best_val = new_val
            else:
                # 新解较差，按概率接受
                accept_prob = math.exp(-delta_f / T)
                if random.random() < accept_prob:
                    current_sol = new_sol
                    current_val = new_val

        # 5. 温度指数衰减
        T *= alpha

    return best_sol, best_val


if __name__ == "__main__":
    # ---------------------- 可调参数（小白推荐范围）----------------------
    T0 = 100  # 初始温度（50-200）
    alpha = 0.9  # 衰减系数（0.8-0.95）
    L = 100  # 每个温度迭代次数（50-200）
    T_end = 1e-5  # 停止温度（1e-4-1e-6）

    # 运行模拟退火算法
    optimal_sol, optimal_val = simulated_annealing(T0, alpha, L, T_end)

    # 输出结果
    print("=" * 60)
    print("模拟退火算法求解二维Rastrigin函数最小值结果：")
    print(f"全局最优解(x, y)：({optimal_sol[0]:.4f}, {optimal_sol[1]:.4f})")
    print(f"全局最优目标函数值：{optimal_val:.4f}")
    print(f"理论全局最优：解(0.0000, 0.0000)，函数值0.0000")
    print("=" * 60)