# -*- coding: utf-8 -*-
"""
2025 A题问题五 五机多弹协同干扰（精简版）
核心改进：
1. 使用混合编码遗传算法（GA）替代PSO，更高效处理多约束多变量问题
2. 引入奖励机制：无人机使用越多、每机弹药使用越多，适应度越高
3. 保留验证函数，确保结果准确性
"""

import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ==== 可配置参数 ====
N_CYL_Z_LAYERS = 5
N_CYL_POINTS_PER_LAYER = 24
DT_SMOKE_CHECK = 0.001
GA_POP_SIZE = 200
GA_GEN = 800
N_THREADS = 32
REWARD_WEIGHT = 2.0  # 奖励系数

# ==== 固定参数 ====
TRUE_TARGET_CENTER = np.array([0., 200., 0.], dtype=np.float64)
TRUE_TARGET_R, TRUE_TARGET_H = 7.0, 10.0
MISSILE_M1_POS0 = np.array([20000., 0., 2000.], dtype=np.float64)
MISSILE_SPEED = 300.
MISSILE_T_MAX = np.linalg.norm(MISSILE_M1_POS0) / MISSILE_SPEED
UAV_INIT_POSITIONS = {
    1: np.array([17800., 0., 1800.], dtype=np.float64),
    2: np.array([12000., 1400., 1400.], dtype=np.float64),
    3: np.array([6000., -3000., 700.], dtype=np.float64),
    4: np.array([11000., 2000., 1800.], dtype=np.float64),
    5: np.array([13000., -2000., 1300.], dtype=np.float64)
}
UAV_SPEED_BOUNDS = (70., 140.)
UAV_ANGLE_BOUNDS = (0., 360.)
SMOKE_RADIUS = 10.
SMOKE_LIFE = 20.
SMOKE_SINK_SPEED = 3.
GRAVITY = 9.8


# ==== 圆柱采样点 ====
def generate_cylinder_samples():
    points = []
    z_vals = np.linspace(TRUE_TARGET_CENTER[2], TRUE_TARGET_CENTER[2] + TRUE_TARGET_H, N_CYL_Z_LAYERS)
    for z in z_vals:
        for k in range(N_CYL_POINTS_PER_LAYER):
            angle = 2 * math.pi * k / N_CYL_POINTS_PER_LAYER
            x = TRUE_TARGET_CENTER[0] + TRUE_TARGET_R * math.cos(angle)
            y = TRUE_TARGET_CENTER[1] + TRUE_TARGET_R * math.sin(angle)
            points.append((x, y, z))
    return np.array(points, dtype=np.float64)


CYLINDER_SAMPLES = generate_cylinder_samples()
N_CYLINDER = len(CYLINDER_SAMPLES)


# ==== 遗传算法核心 ====
class UAVSwarmGA:
    def __init__(self):
        # 编码参数：每架无人机有方向、速度、3个投放时刻、3个起爆延迟
        self.chromosome_length = 5 * (1 + 1 + 3 + 3)
        self.population = self._init_population(GA_POP_SIZE)

    def _init_population(self, size):
        pop = np.zeros((size, self.chromosome_length), dtype=np.float64)
        for i in range(size):
            for uav_idx in range(5):
                base = uav_idx * 8
                # 方向、速度
                pop[i, base] = np.random.uniform(UAV_ANGLE_BOUNDS[0], UAV_ANGLE_BOUNDS[1])
                pop[i, base + 1] = np.random.uniform(UAV_SPEED_BOUNDS[0], UAV_SPEED_BOUNDS[1])
                # 投放时刻（t1, t2, t3）
                max_t1 = MISSILE_T_MAX - 0.5
                pop[i, base + 2] = np.random.uniform(0., max_t1)
                pop[i, base + 3] = np.random.uniform(pop[i, base + 2] + 0.5, max_t1 + 0.5)
                pop[i, base + 4] = np.random.uniform(pop[i, base + 3] + 0.5, max_t1 + 1.0)
                # 起爆延迟（dt1, dt2, dt3）
                max_det1 = min(SMOKE_LIFE, math.sqrt(2 * UAV_INIT_POSITIONS[uav_idx + 1][2] / GRAVITY))
                pop[i, base + 5] = np.random.uniform(0., max_det1)
                max_det2 = min(SMOKE_LIFE, math.sqrt(2 * UAV_INIT_POSITIONS[uav_idx + 1][2] / GRAVITY))
                pop[i, base + 6] = np.random.uniform(0., max_det2)
                max_det3 = min(SMOKE_LIFE, math.sqrt(2 * UAV_INIT_POSITIONS[uav_idx + 1][2] / GRAVITY))
                pop[i, base + 7] = np.random.uniform(0., max_det3)
        return pop

    def evaluate(self, chromosome):
        total_coverage = 0.
        uav_usage = 0
        smoke_usage = 0

        for uav_idx in range(5):
            base = uav_idx * 8
            theta = chromosome[base]
            v = chromosome[base + 1]
            t1, t2, t3 = chromosome[base + 2], chromosome[base + 3], chromosome[base + 4]
            dt1, dt2, dt3 = chromosome[base + 5], chromosome[base + 6], chromosome[base + 7]

            # 检查约束
            if (v < UAV_SPEED_BOUNDS[0] or v > UAV_SPEED_BOUNDS[1] or
                    theta < UAV_ANGLE_BOUNDS[0] or theta > UAV_ANGLE_BOUNDS[1] or
                    t1 < 0 or t2 < t1 + 0.5 or t3 < t2 + 0.5 or
                    dt1 < 0 or dt2 < 0 or dt3 < 0):
                continue

            # 计算起爆时刻
            det1, det2, det3 = t1 + dt1, t2 + dt2, t3 + dt3
            if det1 > MISSILE_T_MAX or det2 > MISSILE_T_MAX or det3 > MISSILE_T_MAX:
                continue

            # 计算爆炸点坐标
            uav_pos = UAV_INIT_POSITIONS[uav_idx + 1]
            theta_rad = math.radians(theta)
            drop1_x = uav_pos[0] + v * t1 * math.cos(theta_rad)
            drop1_y = uav_pos[1] + v * t1 * math.sin(theta_rad)
            drop1_z = uav_pos[2]

            drop2_x = uav_pos[0] + v * t2 * math.cos(theta_rad)
            drop2_y = uav_pos[1] + v * t2 * math.sin(theta_rad)
            drop2_z = uav_pos[2]

            drop3_x = uav_pos[0] + v * t3 * math.cos(theta_rad)
            drop3_y = uav_pos[1] + v * t3 * math.sin(theta_rad)
            drop3_z = uav_pos[2]

            boom1_x = drop1_x + v * dt1 * math.cos(theta_rad)
            boom1_y = drop1_y + v * dt1 * math.sin(theta_rad)
            boom1_z = drop1_z - 0.5 * GRAVITY * dt1 ** 2

            boom2_x = drop2_x + v * dt2 * math.cos(theta_rad)
            boom2_y = drop2_y + v * dt2 * math.sin(theta_rad)
            boom2_z = drop2_z - 0.5 * GRAVITY * dt2 ** 2

            boom3_x = drop3_x + v * dt3 * math.cos(theta_rad)
            boom3_y = drop3_y + v * dt3 * math.sin(theta_rad)
            boom3_z = drop3_z - 0.5 * GRAVITY * dt3 ** 2

            # 时间范围
            t_start = min(det1, det2, det3)
            t_end = min(max(det1, det2, det3) + SMOKE_LIFE, MISSILE_T_MAX)
            if t_start >= t_end:
                continue

            # 计算覆盖时间
            coverage, single = self._calculate_coverage(t_start, t_end, DT_SMOKE_CHECK,
                                                        boom1_x, boom1_y, boom1_z, det1,
                                                        boom2_x, boom2_y, boom2_z, det2,
                                                        boom3_x, boom3_y, boom3_z, det3)
            total_coverage += coverage

            # 统计使用情况
            uav_usage += 1
            smoke_usage += (1 if dt1 > 0 else 0) + (1 if dt2 > 0 else 0) + (1 if dt3 > 0 else 0)

        # 添加奖励机制
        reward = uav_usage * REWARD_WEIGHT + smoke_usage * 0.5 * REWARD_WEIGHT
        return total_coverage + reward

    def _calculate_coverage(self, t_start, t_end, dt, bx1, by1, bz1, td1,
                            bx2, by2, bz2, td2, bx3, by3, bz3, td3):
        coverage = 0.
        single_coverage = [0., 0., 0.]

        for t in np.arange(t_start, t_end + dt, dt):
            mx = MISSILE_M1_POS0[0] + MISSILE_SPEED * (MISSILE_M1_POS0[0] / np.linalg.norm(MISSILE_M1_POS0)) * t
            my = MISSILE_M1_POS0[1] + MISSILE_SPEED * (MISSILE_M1_POS0[1] / np.linalg.norm(MISSILE_M1_POS0)) * t
            mz = MISSILE_M1_POS0[2] + MISSILE_SPEED * (MISSILE_M1_POS0[2] / np.linalg.norm(MISSILE_M1_POS0)) * t

            all_blocked = True
            for (cx, cy, cz) in CYLINDER_SAMPLES:
                blocked = False
                for i in range(3):
                    if i == 0:
                        sx, sy, sz = self._smoke_position(t, td1, bx1, by1, bz1)
                    elif i == 1:
                        sx, sy, sz = self._smoke_position(t, td2, bx2, by2, bz2)
                    else:
                        sx, sy, sz = self._smoke_position(t, td3, bx3, by3, bz3)

                    if self._is_point_in_smoke(mx, my, mz, cx, cy, cz, sx, sy, sz):
                        blocked = True
                        break
                if not blocked:
                    all_blocked = False
                    break

            if all_blocked:
                coverage += dt
                for i in range(3):
                    if (i == 0 and td1 <= t <= td1 + SMOKE_LIFE or
                            i == 1 and td2 <= t <= td2 + SMOKE_LIFE or
                            i == 2 and td3 <= t <= td3 + SMOKE_LIFE):
                        single_coverage[i] += dt

        return coverage, single_coverage

    def _smoke_position(self, t, det_t, x, y, z):
        if t < det_t or t > det_t + SMOKE_LIFE:
            return (float('nan'), float('nan'), float('nan'))
        return (x, y, z - SMOKE_SINK_SPEED * (t - det_t))

    def _is_point_in_smoke(self, mx, my, mz, cx, cy, cz, sx, sy, sz):
        vector = np.array([mx - sx, my - sy, mz - sz])
        return np.linalg.norm(vector) <= SMOKE_RADIUS

    def select(self):
        # 轮盘赌选择
        fitness = np.array([self.evaluate(chromo) for chromo in self.population])
        fitness = fitness - np.min(fitness) + 1e-6  # 确保非负
        probs = fitness / np.sum(fitness)
        selected_indices = np.random.choice(len(self.population), size=GA_POP_SIZE, p=probs)
        return self.population[selected_indices]

    def crossover(self, parents):
        offspring = parents.copy()
        for i in range(0, GA_POP_SIZE, 2):
            if np.random.rand() < 0.8:
                # 无人机级别交叉
                uav_to_swap = np.random.choice(5, 2, replace=False)
                for uav in uav_to_swap:
                    base = uav * 8
                    offspring[i][base:base + 8], offspring[i + 1][base:base + 8] = (
                        offspring[i + 1][base:base + 8].copy(), offspring[i][base:base + 8].copy())
        return offspring

    def mutate(self, population):
        mutated = population.copy()
        for i in range(GA_POP_SIZE):
            if np.random.rand() < 0.2:
                uav_to_mutate = np.random.randint(5)
                base = uav_to_mutate * 8
                # 方向
                mutated[i][base] = np.random.uniform(UAV_ANGLE_BOUNDS[0], UAV_ANGLE_BOUNDS[1])
                # 速度
                mutated[i][base + 1] = np.random.uniform(UAV_SPEED_BOUNDS[0], UAV_SPEED_BOUNDS[1])
                # 投放时刻
                max_t1 = MISSILE_T_MAX - 0.5
                mutated[i][base + 2] = np.random.uniform(0., max_t1)
                mutated[i][base + 3] = np.random.uniform(mutated[i][base + 2] + 0.5, max_t1 + 0.5)
                mutated[i][base + 4] = np.random.uniform(mutated[i][base + 3] + 0.5, max_t1 + 1.0)
                # 起爆延迟
                max_det = min(SMOKE_LIFE, math.sqrt(2 * UAV_INIT_POSITIONS[uav_to_mutate + 1][2] / GRAVITY))
                mutated[i][base + 5] = np.random.uniform(0., max_det)
                mutated[i][base + 6] = np.random.uniform(0., max_det)
                mutated[i][base + 7] = np.random.uniform(0., max_det)
        return mutated

    def run(self):
        for generation in range(GA_GEN):
            selected = self.select()
            crossed = self.crossover(selected)
            mutated = self.mutate(crossed)
            self.population = mutated

            # 计算适应度
            fitness = np.array([self.evaluate(chromo) for chromo in self.population])
            best_idx = np.argmax(fitness)
            print(f"Generation {generation + 1}/{GA_GEN}, Best Fitness: {fitness[best_idx]:.4f}")

        # 返回最优解
        fitness = np.array([self.evaluate(chromo) for chromo in self.population])
        best_idx = np.argmax(fitness)
        return self.population[best_idx], fitness[best_idx]


# ==== 验证函数 ====
def validate_solution(solution):
    total_coverage = 0.

    for uav_idx in range(5):
        base = uav_idx * 8
        theta = solution[base]
        v = solution[base + 1]
        t1, t2, t3 = solution[base + 2], solution[base + 3], solution[base + 4]
        dt1, dt2, dt3 = solution[base + 5], solution[base + 6], solution[base + 7]

        if (v < UAV_SPEED_BOUNDS[0] or v > UAV_SPEED_BOUNDS[1] or
                theta < UAV_ANGLE_BOUNDS[0] or theta > UAV_ANGLE_BOUNDS[1] or
                t1 < 0 or t2 < t1 + 0.5 or t3 < t2 + 0.5 or
                dt1 < 0 or dt2 < 0 or dt3 < 0):
            continue

        det1, det2, det3 = t1 + dt1, t2 + dt2, t3 + dt3
        if det1 > MISSILE_T_MAX or det2 > MISSILE_T_MAX or det3 > MISSILE_T_MAX:
            continue

        uav_pos = UAV_INIT_POSITIONS[uav_idx + 1]
        theta_rad = math.radians(theta)

        boom1_x = uav_pos[0] + v * det1 * math.cos(theta_rad)
        boom1_y = uav_pos[1] + v * det1 * math.sin(theta_rad)
        boom1_z = uav_pos[2] - 0.5 * GRAVITY * dt1 ** 2

        boom2_x = uav_pos[0] + v * det2 * math.cos(theta_rad)
        boom2_y = uav_pos[1] + v * det2 * math.sin(theta_rad)
        boom2_z = uav_pos[2] - 0.5 * GRAVITY * dt2 ** 2

        boom3_x = uav_pos[0] + v * det3 * math.cos(theta_rad)
        boom3_y = uav_pos[1] + v * det3 * math.sin(theta_rad)
        boom3_z = uav_pos[2] - 0.5 * GRAVITY * dt3 ** 2

        t_start = min(det1, det2, det3)
        t_end = min(max(det1, det2, det3) + SMOKE_LIFE, MISSILE_T_MAX)

        if t_start >= t_end:
            continue

        coverage = 0.
        for t in np.arange(t_start, t_end + DT_SMOKE_CHECK, DT_SMOKE_CHECK):
            mx = MISSILE_M1_POS0[0] + MISSILE_SPEED * (MISSILE_M1_POS0[0] / np.linalg.norm(MISSILE_M1_POS0)) * t
            my = MISSILE_M1_POS0[1] + MISSILE_SPEED * (MISSILE_M1_POS0[1] / np.linalg.norm(MISSILE_M1_POS0)) * t
            mz = MISSILE_M1_POS0[2] + MISSILE_SPEED * (MISSILE_M1_POS0[2] / np.linalg.norm(MISSILE_M1_POS0)) * t

            all_blocked = True
            for (cx, cy, cz) in CYLINDER_SAMPLES:
                blocked = False
                # 检查3个烟雾弹
                sx1, sy1, sz1 = UAVSwarmGA._smoke_position(None, t, det1, boom1_x, boom1_y, boom1_z)
                sx2, sy2, sz2 = UAVSwarmGA._smoke_position(None, t, det2, boom2_x, boom2_y, boom2_z)
                sx3, sy3, sz3 = UAVSwarmGA._smoke_position(None, t, det3, boom3_x, boom3_y, boom3_z)

                for (sx, sy, sz) in [(sx1, sy1, sz1), (sx2, sy2, sz2), (sx3, sy3, sz3)]:
                    if UAVSwarmGA._is_point_in_smoke(None, mx, my, mz, cx, cy, cz, sx, sy, sz):
                        blocked = True
                        break
                if not blocked:
                    all_blocked = False
                    break

            if all_blocked:
                coverage += DT_SMOKE_CHECK

        total_coverage += coverage

    return total_coverage


# ==== 主函数 ====
def main():
    ga = UAVSwarmGA()
    best_solution, best_fitness = ga.run()
    print("\n=== 最优解 ===")
    print(f"总适应度（含奖励）：{best_fitness:.4f}")

    validated_coverage = validate_solution(best_solution)
    print(f"验证有效遮蔽时长：{validated_coverage:.4f} 秒")


if __name__ == "__main__":
    main()