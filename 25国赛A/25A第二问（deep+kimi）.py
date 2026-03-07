# -*- coding: utf-8 -*-
"""
问题二：FY1 单弹干扰 M1  多线程 + Numba 批处理加速版
运行前：pip install numba pandas openpyxl
"""
import numpy as np
import pandas as pd
import math, os, getpass, time
from random import random, uniform
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

# --------------------------【1. 手动修改区】--------------------------
num_cyl_points_per_layer = 72        # 圆柱面每层采样点数
cyl_z_layers = [0, 5, 10]            # 采样高度层
dt_smoke_check = 0.001               # 遮蔽判定时间步长（s）
max_iter = 300                       # PSO 迭代次数
particle_size = 100                  # 粒子数
w_pso, c1_pso, c2_pso = 0.7, 1.5, 1.5
num_threads = 16                     # 线程池规模
# ---------------------------------------------------------------------

# --------------------------【2. 常量区】-------------------------------
O = np.array([0, 0, 0], dtype=np.float64)
TRUE_TARGET_CYL = {"center_bottom": np.array([0, 200, 0], dtype=np.float64), "radius": 7, "height": 10}
UAV_FY1_INIT = np.array([17800, 0, 1800], dtype=np.float64)
UAV_V_BOUNDS = (70.0, 140.0)
UAV_THETA_BOUNDS = (0.0, 360.0)
M1_INIT = np.array([20000, 0, 2000], dtype=np.float64)
M1_SPEED = 300.0
M1_DIR = (O - M1_INIT) / np.linalg.norm(M1_INIT)
M1_V = M1_SPEED * M1_DIR
M1_TOTAL_TIME = np.linalg.norm(M1_INIT - O) / M1_SPEED
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_VALID_TIME = 20.0
g = 9.8

def get_desktop_path():
    user = getpass.getuser()
    return f"C:\\Users\\{user}\\Desktop" if os.name == 'nt' else f"/Users/{user}/Desktop"

RESULT_PATH = os.path.join(get_desktop_path(), "问题二最优投放策略.xlsx")

# 预生成圆柱采样点
def generate_cyl_samples():
    pts = []
    center, r = TRUE_TARGET_CYL["center_bottom"], TRUE_TARGET_CYL["radius"]
    for z in cyl_z_layers:
        for theta in np.linspace(0, 2*np.pi, num_cyl_points_per_layer, endpoint=False):
            pts.append([center[0] + r*np.cos(theta),
                        center[1] + r*np.sin(theta),
                        z])
    return np.asarray(pts, dtype=np.float64)

CYL_SAMPLES = generate_cyl_samples()

# 预生成导弹位置数组（0~67 s，步长 0.001 s）
t_all = np.arange(0, 67.0, dt_smoke_check, dtype=np.float64)
missile_pos_all = M1_INIT + M1_V * t_all.reshape(-1, 1)

# --------------------------【3. Numba 批处理】-------------------------
@njit(fastmath=True, parallel=True)
def batch_fitness(particles, cyl, t_samples, mpos):
    M = particles.shape[0]
    N_cyl = cyl.shape[0]
    N_t = t_samples.shape[0]
    fit = np.zeros(M, dtype=np.float64)

    for i in prange(M):
        th, v, t1, dt = particles[i]
        if not (0 <= th <= 360 and 70 <= v <= 140 and t1 >= 0 and dt >= 0):
            continue
        t_det = t1 + dt
        if t_det > 66.999: continue

        th_rad = np.radians(th)
        cos_th, sin_th = np.cos(th_rad), np.sin(th_rad)

        # 烟幕弹投放点
        drop = np.array([17800 + v * t1 * cos_th, v * t1 * sin_th, 1800], dtype=np.float64)
        # 烟幕弹起爆点（平抛运动 + 自由下落）
        det = np.array([
            drop[0] + v * dt * cos_th,
            drop[1] + v * dt * sin_th,
            drop[2] - 0.5 * g * (dt ** 2)
        ], dtype=np.float64)
        if det[2] < 0: continue

        t_start = t_det
        t_end = min(t_det + SMOKE_VALID_TIME, 66.999)
        n_steps = int((t_end - t_start) / dt_smoke_check) + 1
        total = 0.0

        for j in range(n_steps):
            t = t_start + j * dt_smoke_check
            idx = int(t / dt_smoke_check + 0.5)
            if idx < 0 or idx >= N_t:
                continue
            mpos_t = mpos[idx]
            z_smoke = det[2] - SMOKE_SINK_SPEED * (t - t_det)
            if z_smoke < 0:
                z_smoke = 0
            sc = np.array([det[0], det[1], z_smoke], dtype=np.float64)

            all_blocked = True
            for k in range(N_cyl):
                cp = cyl[k]
                AB = cp - mpos_t
                AO = mpos_t - sc
                a = np.dot(AB, AB)
                b = 2.0 * np.dot(AO, AB)
                c = np.dot(AO, AO) - SMOKE_RADIUS ** 2
                delta = b * b - 4 * a * c
                if delta < 0:
                    all_blocked = False
                    break
                sq = np.sqrt(delta)
                s1 = (-b - sq) / (2 * a)
                s2 = (-b + sq) / (2 * a)
                if not (s1 <= 1.0 and s2 >= 0):
                    all_blocked = False
                    break
            if all_blocked:
                total += dt_smoke_check
        fit[i] = total
    return fit

# --------------------------【4. PSO 框架】-----------------------------
def init_swarm():
    pts = np.zeros((particle_size, 4), dtype=np.float64)
    for i in range(particle_size):
        pts[i, 0] = uniform(*UAV_THETA_BOUNDS)
        pts[i, 1] = uniform(*UAV_V_BOUNDS)
        pts[i, 2] = uniform(0, M1_TOTAL_TIME - 1)
        max_dt = min(SMOKE_VALID_TIME, M1_TOTAL_TIME - pts[i, 2])
        pts[i, 3] = uniform(0, max_dt)
    return pts

def pso():
    particles = init_swarm()
    vel = np.zeros_like(particles)
    pbest = particles.copy()
    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        chunks = np.array_split(particles, num_threads * 4)
        res = exe.map(lambda c: batch_fitness(c, CYL_SAMPLES, t_all, missile_pos_all), chunks)
        pbest_fit = np.concatenate(list(res))
    gbest_idx = np.argmax(pbest_fit)
    gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]

    print("=" * 80)
    print("PSO 开始  多线程+Numba并行")
    print(f"粒子={particle_size}  迭代={max_iter}  线程={num_threads}")
    print("=" * 80)

    for it in range(max_iter):
        for i in range(particle_size):
            r1, r2 = random(), random()
            vel[i] = w_pso * vel[i] + c1_pso * r1 * (pbest[i] - particles[i]) + c2_pso * r2 * (gbest - particles[i])
            particles[i] += vel[i]
            particles[i, 0] = np.clip(particles[i, 0], *UAV_THETA_BOUNDS)
            particles[i, 1] = np.clip(particles[i, 1], *UAV_V_BOUNDS)
            particles[i, 2] = np.clip(particles[i, 2], 0, M1_TOTAL_TIME - 0.1)
            max_d = M1_TOTAL_TIME - particles[i, 2] - 1e-6
            particles[i, 3] = np.clip(particles[i, 3], 0, max_d)

        with ThreadPoolExecutor(max_workers=num_threads) as exe:
            chunks = np.array_split(particles, num_threads * 4)
            res = exe.map(lambda c: batch_fitness(c, CYL_SAMPLES, t_all, missile_pos_all), chunks)
            fit = np.concatenate(list(res))

        better = fit > pbest_fit
        pbest[better] = particles[better]
        pbest_fit[better] = fit[better]
        gbest_idx = np.argmax(pbest_fit)
        gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]

        if (it + 1) % 10 == 0 or it == max_iter - 1:
            print(f"Iter {it + 1:3d}/{max_iter}  遮蔽时长={gbest_fit:.6f} s")

    return gbest, gbest_fit

# --------------------------【5. 主程序】-------------------------------
def main():
    gbest, gbest_fit = pso()
    theta_opt, v_opt, t1_opt, dt_opt = gbest
    t_det_opt = t1_opt + dt_opt
    th_rad = np.radians(theta_opt)
    cos_th, sin_th = np.cos(th_rad), np.sin(th_rad)

    drop = np.array([17800 + v_opt * t1_opt * cos_th, v_opt * t1_opt * sin_th, 1800], dtype=np.float64)
    det = np.array([
        drop[0] + v_opt * dt_opt * cos_th,
        drop[1] + v_opt * dt_opt * sin_th,
        drop[2] - 0.5 * g * (dt_opt ** 2)
    ], dtype=np.float64)

    print("\n" + "=" * 80)
    print("最优投放策略")
    print("=" * 80)
    print(f"无人机运动方向(度): {theta_opt:.6f}")
    print(f"无人机运动速度(m/s): {v_opt:.6f}")
    print(f"烟幕干扰弹投放点的x坐标(m): {drop[0]:.6f}")
    print(f"烟幕干扰弹投放点的y坐标(m): {drop[1]:.6f}")
    print(f"烟幕干扰弹投放点的z坐标(m): {drop[2]:.6f}")
    print(f"烟幕干扰弹起爆点的x坐标(m): {det[0]:.6f}")
    print(f"烟幕干扰弹起爆点的y坐标(m): {det[1]:.6f}")
    print(f"烟幕干扰弹起爆点的z坐标(m): {det[2]:.6f}")
    print(f"有效干扰时长(s): {gbest_fit:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()