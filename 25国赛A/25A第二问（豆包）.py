import numpy as np
import math
import os
from random import random, uniform
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 【1. 手动修改参数区】
# ==============================================================================
num_cyl_z_layers = 3  # 真目标采样层数
num_cyl_points_per_layer = 72  # 每层采样点数
max_iter = 300  # PSO迭代次数
particle_size = 100  # 粒子群规模
w_pso = 0.7  # 惯性权重
c1_pso = 1.5  # 认知因子
c2_pso = 1.5  # 社会因子
num_threads = min(os.cpu_count() or 8, particle_size)  # 并行线程数
dt_smoke_check = 0.001  # 烟幕判定时间步长

# ==============================================================================
# 【2. 固定参数（严格遵循《A题.pdf》）】
# ==============================================================================
# 真目标参数
TRUE_TARGET_CYL = {
    "center_bottom": np.array([0.0, 200.0, 0.0]),
    "radius": 7.0,
    "height": 10.0
}

# 无人机FY1参数
UAV_FY1_INIT = np.array([17800.0, 0.0, 1800.0])  # 初始(x,y,z)
UAV_V_BOUNDS = (70.0, 140.0)  # 速度范围
UAV_THETA_BOUNDS = (0.0, 360.0)  # 方向范围（x轴正向起，逆时针）
UAV_Z_FIXED = UAV_FY1_INIT[2]  # 等高度飞行z值

# 导弹M1参数
M1_INIT = np.array([20000.0, 0.0, 2000.0])  # 初始位置
M1_SPEED = 300.0  # 速度
M1_DIR = (-M1_INIT) / np.linalg.norm(M1_INIT)  # 指向原点单位向量
M1_V = M1_SPEED * M1_DIR  # 速度向量
M1_TOTAL_TIME = np.linalg.norm(M1_INIT) / M1_SPEED  # 到原点时间（≈66.8889s）
M1_VX, M1_VY, M1_VZ = M1_V
M1_INIT_X, M1_INIT_Y, M1_INIT_Z = M1_INIT

# 烟幕参数
SMOKE_RADIUS = 10.0  # 有效半径
SMOKE_SINK_SPEED = 3.0  # 下沉速度
SMOKE_VALID_TIME = 20.0  # 有效时间
g = 9.8  # 重力加速度

# ==============================================================================
# 【3. Numba加速函数（逻辑不变，确保计算精度）】
# ==============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("✅ 启用Numba加速")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba未安装，使用普通计算模式（建议执行：pip install numba）")

if HAS_NUMBA:
    CYL_SAMPLES_NUMBA = None  # 预存真目标采样点


    @njit(fastmath=True, cache=True)
    def calc_missile_pos(t: float) -> Tuple[float, float, float]:
        """计算t时刻导弹位置"""
        if t >= M1_TOTAL_TIME + 1e-8:  # 提高精度阈值，避免提前判定失效
            return np.nan, np.nan, np.nan
        x = M1_INIT_X + M1_VX * t
        y = M1_INIT_Y + M1_VY * t
        z = M1_INIT_Z + M1_VZ * t
        return x, y, z


    @njit(fastmath=True, cache=True)
    def calc_smoke_center(t: float, t_det: float, det_x: float, det_y: float, det_z: float) -> Tuple[
        float, float, float]:
        """计算t时刻烟幕中心位置"""
        if t < t_det - 1e-8 or t > t_det + SMOKE_VALID_TIME + 1e-8:
            return np.nan, np.nan, np.nan
        z = det_z - SMOKE_SINK_SPEED * (t - t_det)
        return det_x, det_y, z


    @njit(fastmath=True, cache=True)
    def segment_intersect(Ax, Ay, Az, Bx, By, Bz, Ox, Oy, Oz, r: float) -> bool:
        """线段（导弹-真目标点）与烟幕球相交判定"""
        ABx = Bx - Ax
        ABy = By - Ay
        ABz = Bz - Az
        AOx = Ax - Ox
        AOy = Ay - Oy
        AOz = Az - Oz

        a = ABx ** 2 + ABy ** 2 + ABz ** 2
        if a < 1e-16:  # 更小的线段长度阈值，避免误判
            return (AOx ** 2 + AOy ** 2 + AOz ** 2) <= r ** 2 + 1e-8

        b = 2 * (AOx * ABx + AOy * ABy + AOz * ABz)
        c = (AOx ** 2 + AOy ** 2 + AOz ** 2) - r ** 2
        delta = b ** 2 - 4 * a * c
        if delta < -1e-8:
            return False

        sqrt_delta = np.sqrt(max(delta, 0.0))
        s1 = (-b - sqrt_delta) / (2 * a)
        s2 = (-b + sqrt_delta) / (2 * a)
        return (s1 <= 1.0 + 1e-8) and (s2 >= 0.0 - 1e-8)


    @njit(fastmath=True, cache=True)
    def fitness_numba(theta: float, v: float, t1: float, dt: float) -> float:
        """适应度计算（有效遮蔽时间）"""
        # 约束检查（保留原逻辑，提高精度阈值）
        if not (UAV_THETA_BOUNDS[0] - 1e-8 <= theta <= UAV_THETA_BOUNDS[1] + 1e-8 and
                UAV_V_BOUNDS[0] - 1e-8 <= v <= UAV_V_BOUNDS[1] + 1e-8 and
                t1 >= -1e-8 and dt >= -1e-8):
            return 0.0

        t_det = t1 + dt
        if t_det > M1_TOTAL_TIME - 1e-8 or t_det + SMOKE_VALID_TIME > M1_TOTAL_TIME + 1e-8:
            return 0.0

        # 计算投放点、起爆点（精度保留）
        theta_rad = math.radians(theta)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        drop_x = UAV_FY1_INIT[0] + v * t1 * cos_theta
        drop_y = UAV_FY1_INIT[1] + v * t1 * sin_theta
        det_x = drop_x + v * dt * cos_theta
        det_y = drop_y + v * dt * sin_theta
        det_z = UAV_Z_FIXED - 0.5 * g * dt ** 2
        if det_z < -1e-8:  # 更小的落地判定阈值
            return 0.0

        # 计算有效时间（逻辑不变）
        t_start = t_det
        t_end = min(t_det + SMOKE_VALID_TIME, M1_TOTAL_TIME)
        if t_start >= t_end - 1e-8:
            return 0.0

        num_steps = int((t_end - t_start) / dt_smoke_check) + 1
        effective_time = 0.0

        # 全遮挡判定
        for i in range(num_steps):
            t = t_start + i * dt_smoke_check
            mx, my, mz = calc_missile_pos(t)
            if np.isnan(mx):
                continue
            sx, sy, sz = calc_smoke_center(t, t_det, det_x, det_y, det_z)
            if np.isnan(sx):
                continue

            all_blocked = True
            for j in range(len(CYL_SAMPLES_NUMBA)):
                cx, cy, cz = CYL_SAMPLES_NUMBA[j]
                if not segment_intersect(mx, my, mz, cx, cy, cz, sx, sy, sz, SMOKE_RADIUS):
                    all_blocked = False
                    break
            if all_blocked:
                effective_time += dt_smoke_check

        return effective_time


# ==============================================================================
# 【4. 工具函数（逻辑不变，确保采样精度）】
# ==============================================================================
def generate_cyl_samples() -> np.ndarray:
    """生成真目标采样点（保留更高精度）"""
    cyl_points = []
    ccx, ccy, ccz = TRUE_TARGET_CYL["center_bottom"]
    r = TRUE_TARGET_CYL["radius"]
    height = TRUE_TARGET_CYL["height"]
    z_layers = np.linspace(ccz, ccz + height, num_cyl_z_layers, dtype=np.float64)
    for z in z_layers:
        thetas = np.linspace(0, 2 * np.pi, num_cyl_points_per_layer, endpoint=False, dtype=np.float64)
        for theta in thetas:
            x = ccx + r * np.cos(theta)
            y = ccy + r * np.sin(theta)
            cyl_points.append((x, y, z))
    return np.array(cyl_points, dtype=np.float64)


def fitness_single(particle: np.ndarray) -> float:
    """单粒子适应度计算（兼容Numba/普通模式）"""
    theta, v, t1, dt = particle
    if HAS_NUMBA:
        return fitness_numba(theta, v, t1, dt)
    else:
        # 普通模式逻辑与Numba完全一致，提高精度阈值
        if not (UAV_THETA_BOUNDS[0] - 1e-8 <= theta <= UAV_THETA_BOUNDS[1] + 1e-8 and
                UAV_V_BOUNDS[0] - 1e-8 <= v <= UAV_V_BOUNDS[1] + 1e-8 and
                t1 >= -1e-8 and dt >= -1e-8):
            return 0.0

        t_det = t1 + dt
        if t_det > M1_TOTAL_TIME - 1e-8 or t_det + SMOKE_VALID_TIME > M1_TOTAL_TIME + 1e-8:
            return 0.0

        # 计算投放点、起爆点
        theta_rad = math.radians(theta)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        drop_x = UAV_FY1_INIT[0] + v * t1 * cos_theta
        drop_y = UAV_FY1_INIT[1] + v * t1 * sin_theta
        det_x = drop_x + v * dt * cos_theta
        det_y = drop_y + v * dt * sin_theta
        det_z = UAV_Z_FIXED - 0.5 * g * dt ** 2
        if det_z < -1e-8:
            return 0.0

        # 计算有效时间
        t_start = t_det
        t_end = min(t_det + SMOKE_VALID_TIME, M1_TOTAL_TIME)
        if t_start >= t_end - 1e-8:
            return 0.0

        num_steps = int((t_end - t_start) / dt_smoke_check) + 1
        effective_time = 0.0
        cyl_samples = generate_cyl_samples()

        # 全遮挡判定
        for i in range(num_steps):
            t = t_start + i * dt_smoke_check
            mx = M1_INIT_X + M1_VX * t
            my = M1_INIT_Y + M1_VY * t
            mz = M1_INIT_Z + M1_VZ * t
            if t >= M1_TOTAL_TIME + 1e-8:
                continue
            if t < t_det - 1e-8 or t > t_det + SMOKE_VALID_TIME + 1e-8:
                continue
            sx, sy, sz = det_x, det_y, det_z - SMOKE_SINK_SPEED * (t - t_det)

            all_blocked = True
            for (cx, cy, cz) in cyl_samples:
                ABx = cx - mx
                ABy = cy - my
                ABz = cz - mz
                AOx = mx - sx
                AOy = my - sy
                AOz = mz - sz
                a = ABx ** 2 + ABy ** 2 + ABz ** 2

                if a < 1e-16:
                    if not ((AOx ** 2 + AOy ** 2 + AOz ** 2) <= SMOKE_RADIUS ** 2 + 1e-8):
                        all_blocked = False
                    continue

                b = 2 * (AOx * ABx + AOy * ABy + AOz * ABz)
                c = (AOx ** 2 + AOy ** 2 + AOz ** 2) - SMOKE_RADIUS ** 2
                delta = b ** 2 - 4 * a * c
                if delta < -1e-8 or not (
                        (-b - np.sqrt(max(delta, 0.0))) / (2 * a) <= 1.0 + 1e-8 and (-b + np.sqrt(max(delta, 0.0))) / (
                        2 * a) >= 0.0 - 1e-8):
                    all_blocked = False
                    break
            if all_blocked:
                effective_time += dt_smoke_check
        return effective_time


def fitness_parallel(particles: np.ndarray) -> np.ndarray:
    """多线程并行计算适应度（逻辑不变）"""
    results = np.zeros(len(particles), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {executor.submit(fitness_single, p): i for i, p in enumerate(particles)}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    return results


# ==============================================================================
# 【5. PSO优化核心（逻辑不变，初始化/更新保留更高精度）】
# ==============================================================================
def init_swarm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """初始化粒子群（使用float64提高精度）"""
    particles = np.zeros((particle_size, 4), dtype=np.float64)
    for i in range(particle_size):
        particles[i, 0] = uniform(*UAV_THETA_BOUNDS)  # 方向
        particles[i, 1] = uniform(*UAV_V_BOUNDS)  # 速度
        particles[i, 2] = uniform(0.0, M1_TOTAL_TIME - 1e-8)  # 投放延迟t1
        # 起爆延迟dt约束（保留更高精度）
        max_dt = min(SMOKE_VALID_TIME, math.sqrt(2 * UAV_Z_FIXED / g), M1_TOTAL_TIME - particles[i, 2] - 1e-8)
        particles[i, 3] = uniform(0.0, max_dt)

    # 初始化速度（float64精度）
    velocities = np.zeros_like(particles, dtype=np.float64)
    vel_ranges = [36.0, 7.0, 6.7, 2.0]  # 方向/速度/t1/dt的速度范围
    for i in range(particle_size):
        for j in range(4):
            velocities[i, j] = uniform(-vel_ranges[j], vel_ranges[j])

    # 计算初始适应度
    pbest_fitness = fitness_parallel(particles)
    pbest = particles.copy()
    gbest_idx = np.argmax(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    return particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness


def update_swarm(
        particles: np.ndarray,
        velocities: np.ndarray,
        pbest: np.ndarray,
        pbest_fitness: np.ndarray,
        gbest: np.ndarray,
        gbest_fitness: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """更新粒子群（保留float64精度）"""
    new_particles = particles.copy()
    new_velocities = velocities.copy()
    new_pbest = pbest.copy()
    new_pbest_fitness = pbest_fitness.copy()

    for i in range(particle_size):
        r1, r2 = random(), random()
        # PSO速度更新（精度保留）
        new_velocities[i] = (w_pso * velocities[i] +
                             c1_pso * r1 * (pbest[i] - particles[i]) +
                             c2_pso * r2 * (gbest - particles[i]))

        # 更新位置并截断（更高精度约束）
        new_particles[i] += new_velocities[i]
        new_particles[i, 0] = np.clip(new_particles[i, 0], UAV_THETA_BOUNDS[0] - 1e-8, UAV_THETA_BOUNDS[1] + 1e-8)
        new_particles[i, 1] = np.clip(new_particles[i, 1], UAV_V_BOUNDS[0] - 1e-8, UAV_V_BOUNDS[1] + 1e-8)
        new_particles[i, 2] = np.clip(new_particles[i, 2], 0.0 - 1e-8, M1_TOTAL_TIME - 1e-8)
        max_dt = min(SMOKE_VALID_TIME, math.sqrt(2 * UAV_Z_FIXED / g), M1_TOTAL_TIME - new_particles[i, 2] - 1e-8)
        new_particles[i, 3] = np.clip(new_particles[i, 3], 0.0 - 1e-8, max_dt)

    # 计算新适应度
    new_fitness = fitness_parallel(new_particles)

    # 更新最优解
    for i in range(particle_size):
        if new_fitness[i] > new_pbest_fitness[i] + 1e-12:  # 避免浮点误差导致的无效更新
            new_pbest[i] = new_particles[i].copy()
            new_pbest_fitness[i] = new_fitness[i]
    gbest_idx = np.argmax(new_pbest_fitness)
    new_gbest = new_pbest[gbest_idx].copy()
    new_gbest_fitness = new_pbest_fitness[gbest_idx]

    return new_particles, new_velocities, new_pbest, new_pbest_fitness, new_gbest, new_gbest_fitness


# ==============================================================================
# 【6. 主程序（重点优化：输出精度+合理性验证）】
# ==============================================================================
def check_rationality(theta: float, drop_y: float, det_y: float) -> str:
    """参数合理性检查：方向与y坐标的匹配性"""
    theta_abs = abs(theta % 360.0)  # 归一化到0~360度
    # 若方向接近0/360度（误差<0.1度），y坐标应接近0
    if (theta_abs < 0.1 or theta_abs > 359.9):
        if abs(drop_y) < 1e-3 and abs(det_y) < 1e-3:
            return "✅ 合理：方向接近x轴，y坐标接近0"
        else:
            return f"⚠️ 警告：方向接近x轴（{theta:.4f}度），但y坐标非0（投放y={drop_y:.6f}）"
    # 若方向与x轴有夹角，y坐标不应为0
    else:
        if abs(drop_y) < 1e-6:
            return f"⚠️ 警告：方向与x轴有夹角（{theta:.4f}度），但投放y坐标接近0（{drop_y:.6f}）"
        else:
            return f"✅ 合理：方向与x轴有夹角（{theta:.4f}度），y坐标非0（投放y={drop_y:.6f}）"


def main():
    # 预初始化
    cyl_samples = generate_cyl_samples()
    global CYL_SAMPLES_NUMBA
    if HAS_NUMBA:
        CYL_SAMPLES_NUMBA = cyl_samples
        fitness_numba(180.0, 100.0, 10.0, 1.0)  # Numba预编译

    # 初始化粒子群
    particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness = init_swarm()

    # PSO迭代（每10次打印最佳时间，保留6位小数）
    print("=" * 80)
    print(f"PSO优化启动（迭代{max_iter}次 | 粒子{particle_size}个 | 采样{num_cyl_z_layers}层×{num_cyl_points_per_layer}点）")
    print("=" * 80)
    for iter in range(max_iter):
        particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness = \
            update_swarm(particles, velocities, pbest, pbest_fitness, gbest, gbest_fitness)
        if (iter + 1) % 10 == 0:
            print(f"迭代{iter + 1:3d}/{max_iter} | 当前最佳遮蔽时间：{gbest_fitness:.8f} 秒")

    # 计算最优参数（保留高精度）
    theta_opt, v_opt, t1_opt, dt_opt = gbest
    theta_rad = math.radians(theta_opt)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    # 投放点计算（显示计算依据）
    drop_x = UAV_FY1_INIT[0] + v_opt * t1_opt * cos_theta
    drop_y = UAV_FY1_INIT[1] + v_opt * t1_opt * sin_theta  # 公式：y = v×t1×sin(theta)
    drop_z = UAV_Z_FIXED
    # 起爆点计算
    det_x = drop_x + v_opt * dt_opt * cos_theta
    det_y = drop_y + v_opt * dt_opt * sin_theta
    det_z = drop_z - 0.5 * g * dt_opt ** 2
    # 合理性检查
    rationality_msg = check_rationality(theta_opt, drop_y, det_y)

    # 最终输出（保留6~8位小数，补充计算依据）
    print("\n" + "=" * 80)
    print("FY1 最优烟幕干扰弹投放策略（高精度输出）")
    print("=" * 80)
    print(f"1. 无人机方向：{theta_opt:.8f} 度（x轴正向起算，逆时针为正）")
    print(f"2. 无人机速度：{v_opt:.8f} 米/秒（约束范围：70.0~140.0 m/s）")
    print(f"3. 投放延迟t1：{t1_opt:.8f} 秒（从受领任务到投放的时间）")
    print(f"4. 起爆延迟dt：{dt_opt:.8f} 秒（从投放到起爆的时间）")
    print("-" * 80)
    print("5. 烟幕干扰弹投放点坐标：")
    print(f"   x: {drop_x:.8f} 米（计算依据：初始x + v×t1×cos(theta)）")
    print(f"   y: {drop_y:.8f} 米（计算依据：初始y + v×t1×sin(theta)）")
    print(f"   z: {drop_z:.8f} 米（固定高度：无人机初始z坐标）")
    print("-" * 80)
    print("6. 烟幕干扰弹起爆点坐标：")
    print(f"   x: {det_x:.8f} 米（计算依据：投放x + v×dt×cos(theta)）")
    print(f"   y: {det_y:.8f} 米（计算依据：投放y + v×dt×sin(theta)）")
    print(f"   z: {det_z:.8f} 米（计算依据：投放z - 0.5×g×dt²）")
    print("-" * 80)
    print(f"7. 有效干扰时长：{gbest_fitness:.8f} 秒")
    print(f"8. 参数合理性检查：{rationality_msg}")
    print("=" * 80)


if __name__ == "__main__":
    main()