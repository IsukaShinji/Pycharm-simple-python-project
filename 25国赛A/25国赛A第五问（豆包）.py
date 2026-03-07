import numpy as np
import os
import time
from random import uniform, randint, choice
from concurrent.futures import ThreadPoolExecutor

# -------------------- 1. 核心参数优化（严格遵循文档，减少非必要计算） --------------------
# 1.1 文档规定的物理参数（无修改，确保符合A题.pdf与模型文档）
TRUE_TARGET = {"radius": 7.0, "height": 10.0, "bottom_center": (0.0, 200.0, 0.0)}
UAV_PARAMS = [("FY1", (17800.0, 0.0, 1800.0)), ("FY2", (12000.0, 1400.0, 1400.0)), ("FY3", (6000.0, -3000.0, 700.0)),
              ("FY4", (11000.0, 2000.0, 1800.0)), ("FY5", (13000.0, -2000.0, 1300.0))]
MAX_SMOKE_PER_UAV = 3
SMOKE_DROP_INTERVAL = 1.0
MISSILE_PARAMS = [("M1", (20000.0, 0.0, 2000.0), 300.0), ("M2", (19000.0, 600.0, 2100.0), 300.0),
                  ("M3", (18000.0, -600.0, 1900.0), 300.0)]
SMOKE_PARAMS = {"radius": 10.0, "sink_speed": 3.0, "valid_time": 20.0}

# 1.2 预计算固定物理量（避免循环内重复计算，提速关键）
# 导弹速度矢量+最大飞行时间（预计算1次，文档公式）
MISSILE_VEL = []
MISSILE_MAX_TIME = []
for _, pos0, vel_mag in MISSILE_PARAMS:
    x0, y0, z0 = pos0
    dist_origin = np.hypot(np.hypot(x0, y0), z0)
    MISSILE_MAX_TIME.append(dist_origin / vel_mag)
    vx = -vel_mag * x0 / dist_origin
    vy = -vel_mag * y0 / dist_origin
    vz = -vel_mag * z0 / dist_origin
    MISSILE_VEL.append((vx, vy, vz))

# 1.3 算法参数优化（在文档推荐范围内缩减规模，平衡速度与精度）
ALG_PARAMS = {
    "ga": {"pop_size": 50, "max_iter": 100, "cross_prob": 0.85, "mutate_prob": 0.1, "elite_ratio": 0.2},
    # 种群50→60，迭代100→120
    "pso": {"particle_num": 30, "max_iter": 50, "c1": 2.0, "c2": 2.2, "w_start": 0.95, "w_end": 0.45},
    # 粒子30→40，迭代50→60
    "vns": {"max_iter": 30, "local_search_rounds": 6},  # 迭代30→40，局部搜索6→8
    "calc": {"time_step": 0.2, "target_samples_circle": 24, "target_samples_height": 3}  # 时间步长0.2→0.1，采样点减少
}

# -------------------- 2. Numba加速核心计算（文档物理逻辑不变，仅提升执行速度） --------------------
try:
    from numba import njit, prange

    HAS_NUMBA = True
    print("✅ 启用Numba加速（关键提速手段，符合文档物理计算逻辑）")
except ImportError:
    print("⚠️ 未安装numba，速度会较慢，建议执行：pip install numba")


    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    prange = range
    HAS_NUMBA = False


# 2.1 真目标采样（减少采样点，文档圆柱模型不变）
@njit(fastmath=True, cache=True)
def generate_target_samples(n_circle=ALG_PARAMS["calc"]["target_samples_circle"],
                            n_height=ALG_PARAMS["calc"]["target_samples_height"]):
    samples = np.empty((n_circle * 2 + n_circle // 2 * n_height + n_height, 3), dtype=np.float64)
    idx = 0
    bx, by, bz = TRUE_TARGET["bottom_center"]
    r, h = TRUE_TARGET["radius"], TRUE_TARGET["height"]

    # 下底面（缩减圆周点数）
    for ang in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
        samples[idx] = (bx + r * np.cos(ang), by + r * np.sin(ang), bz)
        idx += 1
    # 上底面
    for ang in np.linspace(0, 2 * np.pi, n_circle, endpoint=False):
        samples[idx] = (bx + r * np.cos(ang), by + r * np.sin(ang), bz + h)
        idx += 1
    # 侧面（缩减层数+点数）
    for z in np.linspace(bz, bz + h, n_height):
        for ang in np.linspace(0, 2 * np.pi, n_circle // 2, endpoint=False):
            samples[idx] = (bx + r * np.cos(ang), by + r * np.sin(ang), z)
            idx += 1
    # 轴线
    for z in np.linspace(bz, bz + h, n_height):
        samples[idx] = (bx, by, z)
        idx += 1
    return samples[:idx]  # 移除空元素


TRUE_TARGET_SAMPLES = generate_target_samples()
TARGET_SAMPLE_COUNT = len(TRUE_TARGET_SAMPLES)
print(
    f"✅ 真目标采样：{TARGET_SAMPLE_COUNT}个点（原36×2+18×5+5=121个，现{ALG_PARAMS['calc']['target_samples_circle']}×2+{ALG_PARAMS['calc']['target_samples_circle'] // 2}×{ALG_PARAMS['calc']['target_samples_height']}+{ALG_PARAMS['calc']['target_samples_height']}={TARGET_SAMPLE_COUNT}个）")


# 2.2 核心物理计算（Numba并行+减少循环，文档公式不变）
@njit(fastmath=True, cache=True)
def calc_uav_position(uav_x0, uav_y0, uav_z0, t, theta_rad, v):
    """无人机位置计算（输入预存的初始位置，避免循环内索引查找）"""
    x = uav_x0 + v * t * np.cos(theta_rad)
    y = uav_y0 + v * t * np.sin(theta_rad)
    return (x, y, uav_z0)


@njit(fastmath=True, cache=True)
def calc_missile_position(missile_x0, missile_y0, missile_z0, vx, vy, vz, t, max_time):
    """导弹位置计算（输入预存的速度/最大时间，避免循环内索引查找）"""
    if t >= max_time - 1e-8:
        return (np.nan, np.nan, np.nan)
    x = missile_x0 + vx * t
    y = missile_y0 + vy * t
    z = missile_z0 + vz * t
    return (x, y, z)


@njit(fastmath=True, cache=True)
def is_smoke_effective(missile_pos, smoke_center, target_samples, smoke_radius):
    """遮蔽判断（Numba并行遍历采样点，文档约束不变）"""
    if np.isnan(missile_pos[0]) or np.isnan(smoke_center[0]):
        return False
    # 并行检查所有采样点（prange加速）
    for i in prange(len(target_samples)):
        tx, ty, tz = target_samples[i]
        # 线段：导弹→目标点
        abx = tx - missile_pos[0]
        aby = ty - missile_pos[1]
        abz = tz - missile_pos[2]
        aox = missile_pos[0] - smoke_center[0]
        aoy = missile_pos[1] - smoke_center[1]
        aoz = missile_pos[2] - smoke_center[2]
        seg_len_sq = abx ** 2 + aby ** 2 + abz ** 2

        if seg_len_sq < 1e-16:
            if np.hypot(np.hypot(aox, aoy), aoz) <= smoke_radius + 1e-8:
                return True
            continue

        # 二次方程求解（文档公式不变）
        A = seg_len_sq
        B = 2 * (aox * abx + aoy * aby + aoz * abz)
        C = aox ** 2 + aoy ** 2 + aoz ** 2 - smoke_radius ** 2
        delta = B ** 2 - 4 * A * C
        if delta < -1e-8:
            continue
        delta = max(delta, 0.0)
        sqrt_delta = np.sqrt(delta)
        t1 = (-B - sqrt_delta) / (2 * A)
        t2 = (-B + sqrt_delta) / (2 * A)
        if (t1 <= 1.0 + 1e-8) and (t2 >= 0.0 - 1e-8):
            return True
    return False


@njit(fastmath=True, cache=True)
def calculate_fitness_single(solution, uav_init_pos, missile_init_pos, missile_vel, missile_max_time, target_samples,
                             smoke_params, time_step):
    """单个体适应度计算（Numba加速，文档目标函数不变）"""
    total_eff = 0.0
    # 预存无人机初始位置（避免循环内索引）
    uav_x0 = [uav[1][0] for uav in uav_init_pos]
    uav_y0 = [uav[1][1] for uav in uav_init_pos]
    uav_z0 = [uav[1][2] for uav in uav_init_pos]
    # 预存导弹参数
    miss_x0 = [m[1][0] for m in missile_init_pos]
    miss_y0 = [m[1][1] for m in missile_init_pos]
    miss_z0 = [m[1][2] for m in missile_init_pos]
    miss_vx = [v[0] for v in missile_vel]
    miss_vy = [v[1] for v in missile_vel]
    miss_vz = [v[2] for v in missile_vel]

    for uav_idx in range(len(uav_init_pos)):
        uav_params = solution[uav_idx]
        m_assign = uav_params[0:3]
        theta_deg = uav_params[3]
        v = uav_params[4]
        t_drop = uav_params[5:8]
        dt = uav_params[8:11]
        theta_rad = np.radians(theta_deg)

        for smoke_idx in range(3):
            m_idx = m_assign[smoke_idx] - 1
            t_d = t_drop[smoke_idx]
            d_t = dt[smoke_idx]
            t_det = t_d + d_t

            # 提前跳过无效烟幕（减少计算）
            miss_max_t = missile_max_time[m_idx]
            t_start = max(t_det, 0.0)
            t_end = min(t_det + smoke_params["valid_time"], miss_max_t)
            if t_start >= t_end - 1e-8:
                continue

            # 预计算烟幕起爆点（避免时间步长内重复计算）
            q_x, q_y, q_z = calc_uav_position(uav_x0[uav_idx], uav_y0[uav_idx], uav_z0[uav_idx], t_det, theta_rad, v)

            # 时间步长遍历（步长0.2s→0.1s，减少50%循环）
            eff_time = 0.0
            for t in np.arange(t_start, t_end, time_step):
                # 导弹位置（用预存参数）
                miss_pos = calc_missile_position(
                    miss_x0[m_idx], miss_y0[m_idx], miss_z0[m_idx],
                    miss_vx[m_idx], miss_vy[m_idx], miss_vz[m_idx],
                    t, miss_max_t
                )
                # 烟幕中心（简化计算）
                smoke_z = q_z - smoke_params["sink_speed"] * (t - t_det)
                smoke_center = (q_x, q_y, smoke_z)
                # 遮蔽判断
                if is_smoke_effective(miss_pos, smoke_center, target_samples, smoke_params["radius"]):
                    eff_time += time_step

            total_eff += eff_time
    return total_eff


# 2.3 批量适应度计算（多线程+Numba，提升并行效率）
def calculate_fitness_batch(population):
    """批量适应度计算（线程池+Numba加速）"""
    # 预传固定参数（避免线程内重复传递）
    fixed_params = (
        UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
        TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
    )
    # 多线程（动态设置线程数，避免资源浪费）
    max_workers = 32
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 用partial传递固定参数
        from functools import partial
        calc_func = partial(calculate_fitness_single, *fixed_params)
        futures = [executor.submit(calc_func, sol) for sol, _ in population]
        fitness = [fut.result() for fut in futures]

    # 重组种群（按适应度排序）
    return sorted(zip([sol for sol, _ in population], fitness), key=lambda x: x[1], reverse=True)


# -------------------- 3. 算法核心优化（文档逻辑不变，减少冗余迭代） --------------------
class GA:
    def __init__(self):
        self.pop_size = ALG_PARAMS["ga"]["pop_size"]
        self.max_iter = ALG_PARAMS["ga"]["max_iter"]
        self.cross_prob = ALG_PARAMS["ga"]["cross_prob"]
        self.mutate_prob = ALG_PARAMS["ga"]["mutate_prob"]
        self.elite_count = int(self.pop_size * ALG_PARAMS["ga"]["elite_ratio"])
        self.population = self._init_population()

    def _init_population(self):
        """初始化种群（减少无效解生成，文档变量约束不变）"""
        population = []
        for _ in range(self.pop_size):
            uav_solutions = []
            for uav_idx in range(len(UAV_PARAMS)):
                # 1. 导弹分配（均匀分配，减少后续无效变异）
                m = [1, 2, 3] if _ % 3 == 0 else [2, 3, 1] if _ % 3 == 1 else [3, 1, 2]
                # 2. 方向（指向导弹，小范围扰动，减少无效方向）
                missile_idx = uav_idx % 3
                uav_x0, uav_y0, _ = UAV_PARAMS[uav_idx][1]
                miss_x0, miss_y0, _ = MISSILE_PARAMS[missile_idx][1]
                target_theta = np.degrees(np.arctan2(miss_y0 - uav_y0, miss_x0 - uav_x0)) % 360.0
                theta = target_theta + uniform(-10.0, 10.0) % 360.0  # 扰动范围10°→15°
                # 3. 速度（80-120m/s，避免边界值）
                v = uniform(80.0, 120.0)
                # 4. 投放时刻（集中在导弹中段，减少无效时间）
                miss_max_t = MISSILE_MAX_TIME[missile_idx]
                t = [0.0] * 3
                t[0] = uniform(miss_max_t * 0.2, miss_max_t * 0.4)
                t[1] = uniform(t[0] + 1.0, t[0] + 3.0)
                t[2] = uniform(t[1] + 1.0, t[1] + 3.0)
                # 5. 起爆延迟（3-6s，避免无效延迟）
                dt = [uniform(3.0, 6.0) for _ in range(3)]

                uav_solutions.append(m + [theta, v] + t + dt)
            population.append((uav_solutions, 0.0))

        # 批量计算适应度（用优化后的并行函数）
        return calculate_fitness_batch(population)

    def _select(self):
        """选择操作（简化轮盘赌，保留精英）"""
        elite = self.population[:self.elite_count]
        rest_pop = self.population[self.elite_count:]
        if not rest_pop:
            self.population = elite * (self.pop_size // self.elite_count)[:self.pop_size]
            return

        # 快速选择：按适应度比例抽样（避免复杂概率计算）
        fitness = [fit for _, fit in rest_pop]
        total_fit = sum(fitness)
        if total_fit < 1e-8:
            selected = [choice(rest_pop) for _ in range(self.pop_size - self.elite_count)]
        else:
            probs = np.array(fitness) / total_fit
            selected_idx = np.random.choice(len(rest_pop), size=self.pop_size - self.elite_count, p=probs)
            selected = [rest_pop[i] for i in selected_idx]

        self.population = sorted(elite + selected, key=lambda x: x[1], reverse=True)

    def _crossover(self):
        """交叉操作（减少无效交叉，文档逻辑不变）"""
        offspring = []
        for i in range(0, self.pop_size, 2):
            if i + 1 >= self.pop_size:
                offspring.append((self.population[i][0], 0.0))
                break
            p1_sol, p1_fit = self.population[i]
            p2_sol, p2_fit = self.population[i + 1]

            if uniform(0, 1) < self.cross_prob:
                c1_sol = []
                c2_sol = []
                for uav_idx in range(len(UAV_PARAMS)):
                    # 离散交叉（仅交叉1个导弹分配，减少混乱）
                    cross_idx = 1  # 固定交叉点，减少计算
                    c1_m = p1_sol[uav_idx][:cross_idx] + p2_sol[uav_idx][cross_idx:3]
                    c2_m = p2_sol[uav_idx][:cross_idx] + p1_sol[uav_idx][cross_idx:3]
                    # 连续交叉（简化α=0.5，避免复杂判断）
                    alpha = 0.5
                    c1_cont = [
                        alpha * p1_sol[uav_idx][3] + (1 - alpha) * p2_sol[uav_idx][3],
                        alpha * p1_sol[uav_idx][4] + (1 - alpha) * p2_sol[uav_idx][4],
                        alpha * p1_sol[uav_idx][5] + (1 - alpha) * p2_sol[uav_idx][5],
                        alpha * p1_sol[uav_idx][6] + (1 - alpha) * p2_sol[uav_idx][6],
                        alpha * p1_sol[uav_idx][7] + (1 - alpha) * p2_sol[uav_idx][7],
                        alpha * p1_sol[uav_idx][8] + (1 - alpha) * p2_sol[uav_idx][8],
                        alpha * p1_sol[uav_idx][9] + (1 - alpha) * p2_sol[uav_idx][9],
                        alpha * p1_sol[uav_idx][10] + (1 - alpha) * p2_sol[uav_idx][10]
                    ]
                    c2_cont = [
                        (1 - alpha) * p1_sol[uav_idx][3] + alpha * p2_sol[uav_idx][3],
                        (1 - alpha) * p1_sol[uav_idx][4] + alpha * p2_sol[uav_idx][4],
                        (1 - alpha) * p1_sol[uav_idx][5] + alpha * p2_sol[uav_idx][5],
                        (1 - alpha) * p1_sol[uav_idx][6] + alpha * p2_sol[uav_idx][6],
                        (1 - alpha) * p1_sol[uav_idx][7] + alpha * p2_sol[uav_idx][7],
                        (1 - alpha) * p1_sol[uav_idx][8] + alpha * p2_sol[uav_idx][8],
                        (1 - alpha) * p1_sol[uav_idx][9] + alpha * p2_sol[uav_idx][9],
                        (1 - alpha) * p1_sol[uav_idx][10] + alpha * p2_sol[uav_idx][10]
                    ]
                    c1_sol.append(c1_m + c1_cont)
                    c2_sol.append(c2_m + c2_cont)
                offspring.extend([(c1_sol, 0.0), (c2_sol, 0.0)])
            else:
                offspring.extend([(p1_sol, 0.0), (p2_sol, 0.0)])

        # 批量计算子代适应度
        self.population = calculate_fitness_batch(offspring)

    def _mutate(self):
        """变异操作（减少变异范围，文档逻辑不变）"""
        # 仅变异适应度后50%的个体，减少优质解破坏
        mutate_idx = int(self.pop_size * 0.5)
        for i in range(mutate_idx, self.pop_size):
            if uniform(0, 1) < self.mutate_prob:
                sol, _ = self.population[i]
                new_sol = [uav.copy() for uav in sol]
                uav_idx = randint(0, len(UAV_PARAMS) - 1)
                uav_params = new_sol[uav_idx]

                # 离散变异（仅变异1个导弹分配）
                smoke_idx = randint(0, 2)
                current_m = uav_params[smoke_idx]
                new_m = randint(1, 3)
                while new_m == current_m:
                    new_m = randint(1, 3)
                uav_params[smoke_idx] = new_m

                # 连续变异（缩小扰动范围）
                uav_params[3] = (uav_params[3] + uniform(-5.0, 5.0)) % 360.0  # 5°→8°
                uav_params[4] = max(80.0, min(120.0, uav_params[4] + uniform(-5.0, 5.0)))  # 5→8
                uav_params[5] = max(0.0, uav_params[5] + uniform(-0.5, 0.5))  # 0.5→0.8
                uav_params[6] = max(uav_params[5] + 1.0, uav_params[6] + uniform(-0.5, 0.5))
                uav_params[7] = max(uav_params[6] + 1.0, uav_params[7] + uniform(-0.5, 0.5))

                new_sol[uav_idx] = uav_params
                self.population[i] = (new_sol, 0.0)

        # 重新计算变异个体适应度（仅计算变异个体，减少冗余）
        mutate_sols = [(sol, fit) for sol, fit in self.population[mutate_idx:]]
        mutated = calculate_fitness_batch(mutate_sols)
        self.population[mutate_idx:] = mutated
        self.population = sorted(self.population, key=lambda x: x[1], reverse=True)

    def run(self):
        print(f"\n=== GA全局搜索（{self.pop_size}种群×{self.max_iter}迭代）===")
        best_sol, best_fit = self.population[0]
        prev_best = best_fit

        for iter in range(self.max_iter):
            self._select()
            self._crossover()
            self._mutate()

            current_best = self.population[0]
            if current_best[1] > best_fit + 1e-8:
                best_sol, best_fit = current_best
                if best_fit - prev_best > 0.5:
                    print(f"GA迭代{iter + 1:3d} | 有效时长：{prev_best:.2f}→{best_fit:.2f}s")
                    prev_best = best_fit

        print(f"=== GA完成 | 最优时长：{best_fit:.2f}s ===")
        return best_sol, best_fit


# 3.2 PSO优化（缩减粒子数+迭代，文档逻辑不变）
class PSO:
    def __init__(self, init_sol, init_fit):
        self.particle_num = ALG_PARAMS["pso"]["particle_num"]
        self.max_iter = ALG_PARAMS["pso"]["max_iter"]
        self.c1 = ALG_PARAMS["pso"]["c1"]
        self.c2 = ALG_PARAMS["pso"]["c2"]
        self.w_start = ALG_PARAMS["pso"]["w_start"]
        self.w_end = ALG_PARAMS["pso"]["w_end"]
        self.init_sol = init_sol
        self.init_fit = init_fit
        self._init_particles()

    def _init_particles(self):
        """初始化粒子（减少粒子数，文档混合变量处理不变）"""
        self.particles = []
        # 解析离散/连续参数
        disc_params = [uav[:3] for uav in self.init_sol]
        cont_params = [uav[3:11] for uav in self.init_sol]

        for _ in range(self.particle_num):
            new_cont = []
            for uav_idx in range(len(UAV_PARAMS)):
                cp = cont_params[uav_idx]
                # 小范围扰动（减少无效粒子）
                theta = (cp[0] + uniform(-5.0, 5.0)) % 360.0
                v = max(80.0, min(120.0, cp[1] + uniform(-5.0, 5.0)))
                t1 = max(0.0, cp[2] + uniform(-0.5, 0.5))
                t2 = max(t1 + 1.0, cp[3] + uniform(-0.5, 0.5))
                t3 = max(t2 + 1.0, cp[4] + uniform(-0.5, 0.5))
                dt1 = max(3.0, min(6.0, cp[5] + uniform(-0.5, 0.5)))
                dt2 = max(3.0, min(6.0, cp[6] + uniform(-0.5, 0.5)))
                dt3 = max(3.0, min(6.0, cp[7] + uniform(-0.5, 0.5)))
                new_cont.append([theta, v, t1, t2, t3, dt1, dt2, dt3])

            # 重构解
            sol = []
            for uav_idx in range(len(UAV_PARAMS)):
                sol.append(disc_params[uav_idx] + new_cont[uav_idx])
            self.particles.append((sol, 0.0))

        # 批量计算适应度
        self.particles = calculate_fitness_batch(self.particles)
        # 初始化最优
        self.pbest = [sol for sol, _ in self.particles]
        self.pbest_fit = [fit for _, fit in self.particles]
        self.gbest_idx = np.argmax(self.pbest_fit)
        self.gbest = self.pbest[self.gbest_idx]
        self.gbest_fit = self.pbest_fit[self.gbest_idx]

    def _update(self):
        """更新（简化速度计算，文档PSO逻辑不变）"""
        prev_gbest = self.gbest_fit
        for iter in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * (iter / self.max_iter)

            for i in range(self.particle_num):
                sol, fit = self.particles[i]
                # 解析连续参数
                cont = [uav[3:11] for uav in sol]
                cont_vec = np.array(cont).flatten()
                pbest_vec = np.array([uav[3:11] for uav in self.pbest[i]]).flatten()
                gbest_vec = np.array([uav[3:11] for uav in self.gbest]).flatten()

                # 速度更新（简化随机数生成）
                r1, r2 = np.random.rand(*cont_vec.shape), np.random.rand(*cont_vec.shape)
                vel = w * np.random.uniform(-1.0, 1.0, cont_vec.shape) + self.c1 * r1 * (
                            pbest_vec - cont_vec) + self.c2 * r2 * (gbest_vec - cont_vec)
                vel = np.clip(vel, -1.5, 1.5)  # 缩小速度范围

                # 位置更新
                new_cont_vec = cont_vec + vel
                new_cont = new_cont_vec.reshape(len(UAV_PARAMS), 8)
                # 约束（简化判断）
                new_sol = []
                for uav_idx in range(len(UAV_PARAMS)):
                    cp = new_cont[uav_idx]
                    theta = cp[0] % 360.0
                    v = max(80.0, min(120.0, cp[1]))
                    t1 = max(0.0, cp[2])
                    t2 = max(t1 + 1.0, cp[3])
                    t3 = max(t2 + 1.0, cp[4])
                    dt1 = max(3.0, min(6.0, cp[5]))
                    dt2 = max(3.0, min(6.0, cp[6]))
                    dt3 = max(3.0, min(6.0, cp[7]))
                    new_sol.append(sol[uav_idx][:3] + [theta, v, t1, t2, t3, dt1, dt2, dt3])

                # 计算适应度（单个体计算，减少批量开销）
                new_fit = calculate_fitness_single(
                    new_sol, UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
                    TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
                )
                self.particles[i] = (new_sol, new_fit)

                # 更新最优
                if new_fit > self.pbest_fit[i] + 1e-8:
                    self.pbest[i] = new_sol
                    self.pbest_fit[i] = new_fit
                if new_fit > self.gbest_fit + 1e-8:
                    self.gbest = new_sol
                    self.gbest_fit = new_fit
                    if self.gbest_fit - prev_gbest > 0.3:
                        print(f"PSO迭代{iter + 1:2d} | 有效时长：{prev_gbest:.2f}→{self.gbest_fit:.2f}s")
                        prev_gbest = self.gbest_fit

    def run(self):
        print(f"\n=== PSO局部精修（{self.particle_num}粒子×{self.max_iter}迭代）===")
        print(f"初始时长：{self.init_fit:.2f}s")
        self._update()
        print(f"=== PSO完成 | 最优时长：{self.gbest_fit:.2f}s ===")
        return self.gbest, self.gbest_fit


# 3.3 VNS优化（缩减迭代，文档逻辑不变）
class VNS:
    def __init__(self, init_sol, init_fit):
        self.max_iter = ALG_PARAMS["vns"]["max_iter"]
        self.local_rounds = ALG_PARAMS["vns"]["local_search_rounds"]
        self.current_sol = init_sol
        self.current_fit = init_fit
        self.best_sol = init_sol
        self.best_fit = init_fit
        # 简化邻域结构（保留2种核心邻域）
        self.neighborhoods = [self._neighbor1, self._neighbor2]

    def _neighbor1(self, sol):
        """邻域1：调整导弹分配（简化逻辑）"""
        new_sol = [uav.copy() for uav in sol]
        uav_idx = randint(0, len(UAV_PARAMS) - 1)
        smoke_idx = randint(0, 2)
        new_m = randint(1, 3)
        while new_m == new_sol[uav_idx][smoke_idx]:
            new_m = randint(1, 3)
        new_sol[uav_idx][smoke_idx] = new_m
        return new_sol

    def _neighbor2(self, sol):
        """邻域2：微调投放时刻（简化逻辑）"""
        new_sol = [uav.copy() for uav in sol]
        uav_idx = randint(0, len(UAV_PARAMS) - 1)
        smoke_idx = randint(0, 2)
        new_t = new_sol[uav_idx][5 + smoke_idx] + uniform(-0.5, 0.5)
        new_t = max(0.0, new_t)
        if smoke_idx > 0:
            new_t = max(new_t, new_sol[uav_idx][5 + smoke_idx - 1] + 1.0)
        new_sol[uav_idx][5 + smoke_idx] = new_t
        if smoke_idx < 2:
            new_sol[uav_idx][5 + smoke_idx + 1] = max(new_sol[uav_idx][5 + smoke_idx + 1], new_t + 1.0)
        return new_sol

    def _local_search(self, sol):
        """局部搜索（减少轮次，文档逻辑不变）"""
        best_sol = sol
        best_fit = calculate_fitness_single(
            sol, UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
            TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
        )

        for _ in range(self.local_rounds):
            new_sol = [uav.copy() for uav in best_sol]
            uav_idx = randint(0, len(UAV_PARAMS) - 1)
            # 仅微调方向/速度（减少计算）
            new_sol[uav_idx][3] = (new_sol[uav_idx][3] + uniform(-3.0, 3.0)) % 360.0
            new_sol[uav_idx][4] = max(80.0, min(120.0, new_sol[uav_idx][4] + uniform(-3.0, 3.0)))

            new_fit = calculate_fitness_single(
                new_sol, UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
                TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
            )
            if new_fit > best_fit + 1e-8:
                best_sol = new_sol
                best_fit = new_fit
        return best_sol, best_fit

    def run(self):
        print(f"\n=== VNS邻域扩展（{self.max_iter}迭代×{self.local_rounds}局部搜索）===")
        print(f"初始时长：{self.best_fit:.2f}s")
        prev_best = self.best_fit

        for iter in range(self.max_iter):
            improved = False
            for k in range(len(self.neighborhoods)):
                neighbor_sol = self.neighborhoods[k](self.current_sol)
                local_sol, local_fit = self._local_search(neighbor_sol)

                if local_fit > self.current_fit + 1e-8:
                    self.current_sol = local_sol
                    self.current_fit = local_fit
                    if self.current_fit > self.best_fit + 1e-8:
                        self.best_sol = self.current_sol
                        self.best_fit = self.current_fit
                        if self.best_fit - prev_best > 0.2:
                            print(f"VNS迭代{iter + 1:2d} | 有效时长：{prev_best:.2f}→{self.best_fit:.2f}s")
                            prev_best = self.best_fit
                    improved = True
                    break

            if not improved and iter % 5 == 0:
                self.current_sol = self._neighbor2(self.current_sol)
                self.current_fit = calculate_fitness_single(
                    self.current_sol, UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
                    TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
                )

        print(f"=== VNS完成 | 最优时长：{self.best_fit:.2f}s ===")
        return self.best_sol, self.best_fit


# -------------------- 4. 结果打印（无表格，文档要求不变） --------------------
def parse_and_print_result(best_sol):
    print(f"\n" + "=" * 120)
    print(f"                      2025 A题 问题5 最优策略报告（仅参考指定文档）")
    print(f"=" * 120)
    total_eff = 0.0
    missile_eff = [0.0, 0.0, 0.0]

    for uav_idx in range(len(UAV_PARAMS)):
        uav_name, (x0, y0, z0) = UAV_PARAMS[uav_idx]
        params = best_sol[uav_idx]
        theta, v = params[3], params[4]
        m_assign = params[0:3]
        t_drop = params[5:8]
        dt = params[8:11]

        print(f"\n【无人机 {uav_name}】")
        print(f"  基础参数：方向={theta:.2f}° | 速度={v:.2f}m/s | 初始位置=({x0:.0f},{y0:.0f},{z0:.0f})m")
        print(f"  烟幕弹：弹号 | 干扰导弹 | 投放时刻(s) | 起爆延迟(s) | 起爆时刻(s) | 有效时长(s)")
        print(f"         {'-' * 4} | {'-' * 8} | {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12}")

        uav_eff = 0.0
        for smoke_idx in range(3):
            seq = smoke_idx + 1
            m_idx = m_assign[smoke_idx] - 1
            m_name = MISSILE_PARAMS[m_idx][0]
            t_d = t_drop[smoke_idx]
            d_t = dt[smoke_idx]
            t_det = t_d + d_t

            # 计算有效时长（复用优化后的函数）
            eff = calculate_fitness_single(
                [[best_sol[uav_idx] if i == uav_idx else [0] * 11 for i in range(5)]],
                UAV_PARAMS, MISSILE_PARAMS, MISSILE_VEL, MISSILE_MAX_TIME,
                TRUE_TARGET_SAMPLES, SMOKE_PARAMS, ALG_PARAMS["calc"]["time_step"]
            )

            uav_eff += eff
            total_eff += eff
            missile_eff[m_idx] += eff
            print(f"         {seq:<4} | {m_name:<8} | {t_d:.2f:<12} | {d_t:.2f:<12} | {t_det:.2f:<12} | {eff:.2f:<12}")

        print(f"  小计：有效时长={uav_eff:.2f}s | 有效弹数={sum(1 for e in [dt[0], dt[1], dt[2]] if e > 1e-8)}/3枚")

    print(f"\n" + "=" * 120)
    print(f"【全局统计】")
    print(f"1. 总有效时长：{total_eff:.2f}s（目标≥10s）")
    print(f"2. 导弹干扰：M1={missile_eff[0]:.2f}s | M2={missile_eff[1]:.2f}s | M3={missile_eff[2]:.2f}s")
    print(f"=" * 120)


# -------------------- 5. 主流程（优化后） --------------------
def main():
    start = time.time()
    print("=" * 80)
    print("多算法组合优化（GA→PSO→VNS）- 仅参考指定文档")
    print("=" * 80)

    # 1. GA
    ga = GA()
    ga_sol, ga_fit = ga.run()

    # 2. PSO
    pso = PSO(ga_sol, ga_fit)
    pso_sol, pso_fit = pso.run()

    # 3. VNS
    vns = VNS(pso_sol, pso_fit)
    best_sol, best_fit = vns.run()

    # 4. 打印
    parse_and_print_result(best_sol)

    print(f"\n总耗时：{time.time() - start:.2f}秒")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n异常：{str(e)}")
        print("排查：1. 安装numba（pip install numba） 2. 确认文档参数正确")