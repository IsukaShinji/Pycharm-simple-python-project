import numpy as np
import matplotlib.pyplot as plt


def simulate_mm1(arrival_rate, service_rate, num_customers):
    """
    模拟M/M/1排队模型，计算核心性能指标
    参数:
        arrival_rate (float): 顾客到达率(人/小时)
        service_rate (float): 服务台服务率(人/小时)
        num_customers (int): 模拟的顾客数量(建议≥1000，保证稳态)
    返回:
        dict: 包含性能指标的字典(直观展示结果)
        np.ndarray: 顾客到达时间数组
        np.ndarray: 顾客等待时间数组
        np.ndarray: 顾客结束服务时间数组
    """
    # 1. 检查模型稳定性(利用率ρ必须<1)
    utilization = arrival_rate / service_rate
    if utilization >= 1:
        raise ValueError("错误:服务率必须大于到达率(ρ=λ/μ<1)，否则队列无限增长!")

    # 2. 生成顾客到达时间(泊松过程→指数分布的到达间隔)
    # 指数分布scale参数=到达间隔均值(1/arrival_rate 小时)
    inter_arrival_times = np.random.exponential(scale=1 / arrival_rate, size=num_customers)
    arrival_times = np.cumsum(inter_arrival_times)  # 累加得到每个顾客到达时间
    arrival_times -= arrival_times[0]  # 标准化，让第一个顾客在t=0到达

    # 3. 生成顾客服务时间(指数分布，均值=1/service_rate 小时)
    service_times = np.random.exponential(scale=1 / service_rate, size=num_customers)

    # 4. 模拟排队过程(计算开始/结束服务时间)
    start_service_times = np.zeros(num_customers)  # 开始服务时间
    end_service_times = np.zeros(num_customers)  # 结束服务时间
    prev_end_time = 0  # 前一个顾客结束时间(初始为0，服务台空闲)

    for i in range(num_customers):
        # 开始服务时间：取"前顾客结束时间"与"当前顾客到达时间"的较大值
        start_service_times[i] = max(prev_end_time, arrival_times[i])
        end_service_times[i] = start_service_times[i] + service_times[i]
        prev_end_time = end_service_times[i]  # 更新前顾客结束时间

    # 5. 计算核心性能指标
    wait_times = start_service_times - arrival_times  # 队列等待时间
    system_times = end_service_times - arrival_times  # 系统总时间(等待+服务)

    # 转换为分钟（更符合日常认知）
    avg_wait_time_min = np.mean(wait_times) * 60
    avg_system_time_min = np.mean(system_times) * 60

    # 用Little公式计算平均队列长度、系统顾客数
    avg_queue_length = arrival_rate * np.mean(wait_times)
    avg_system_customers = arrival_rate * np.mean(system_times)

    # 整理结果字典
    results = {
        "到达率(人/小时)": round(arrival_rate, 2),
        "服务率(人/小时)": round(service_rate, 2),
        "服务台利用率(%)": round(utilization * 100, 2),
        "平均等待时间(分钟)": round(avg_wait_time_min, 2),
        "平均系统时间(分钟)": round(avg_system_time_min, 2),
        "平均队列长度(人)": round(avg_queue_length, 2),
        "平均系统顾客数(人)": round(avg_system_customers, 2)
    }

    return results, arrival_times, wait_times, end_service_times


if __name__ == "__main__":
    # ---------------------- 1. 设置模拟参数 ----------------------
    lambda_arrival = 10  # 顾客到达率(人/小时)，可修改（如高峰时段设为12）
    mu_service = 15  # 服务台服务率(人/小时)，可修改（如提速后设为20）
    n_customers = 1000  # 模拟顾客数量（≥1000时结果更接近稳态理论值）

    # ---------------------- 2. 运行模拟 ----------------------
    try:
        results, arrival_times, wait_times, end_service_times = simulate_mm1(
            lambda_arrival, mu_service, n_customers
        )
    except ValueError as e:
        print(e)
        exit()

    # ---------------------- 3. 输出模拟结果 ----------------------
    print("=" * 50)
    print("M/M/1 排队模型模拟结果(稳态)")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key:20}: {value}")
    print("=" * 50)

    # ---------------------- 4. 可视化展示（可选） ----------------------
    # 图1：顾客等待时间分布（直方图）
    plt.figure(figsize=(10, 5))
    plt.hist(wait_times * 60, bins=30, edgecolor='black', alpha=0.7)
    plt.title("顾客等待时间分布(M/M/1 模型)", fontsize=12)
    plt.xlabel("等待时间(分钟)", fontsize=10)
    plt.ylabel("顾客数量", fontsize=10)
    plt.grid(alpha=0.3)

    # 图2：队列长度随时间变化（折线图）
    # 生成所有关键时间点（顾客到达/离开时队列长度变化）
    times = np.concatenate([arrival_times, end_service_times])
    times.sort()
    queue_lengths = []
    for t in times:
        arrived = np.sum(arrival_times <= t)  # 已到达顾客数
        departed = np.sum(end_service_times <= t)  # 已离开顾客数
        system_customers = arrived - departed  # 系统中当前顾客数（队列+服务）
        queue_length = max(system_customers - 1, 0)  # 队列长度=系统顾客数-服务中1人
        queue_lengths.append(queue_length)

    plt.figure(figsize=(10, 5))
    plt.plot(times, queue_lengths, linewidth=1.5)
    plt.title("队列长度随时间变化(M/M/1 模型)", fontsize=12)
    plt.xlabel("时间(小时)", fontsize=10)
    plt.ylabel("队列长度(人)", fontsize=10)
    plt.grid(alpha=0.3)

    # 显示所有图表
    plt.show()