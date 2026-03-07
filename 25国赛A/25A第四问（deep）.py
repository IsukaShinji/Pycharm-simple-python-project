# -*- coding: utf-8 -*-
"""
2025 A-prob4 极速版·进度条
Numba 矢量化内核 + tqdm 实时进度
"""
import time
import math
import numpy as np
from random import uniform
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # pip install tqdm

# ===================  可改参数 ===================
NUM_CYL_Z_LAYERS = 5
NUM_CYL_POINTS_PER_LAYER = 24
DT_SMOKE_CHECK = 0.001 # 步长
PSO_PARTICLE = 320
PSO_ITER = 800
N_THREAD = 32
W0, W1 = 0.9, 0.4
C1, C2 = 1.5, 3.0
# ================================================

TRUE_CYL_CENTRE = np.array([0., 200., 0.])
TRUE_CYL_R = 7.0
TRUE_CYL_H = 10.0

M1_POS0 = np.array([20000., 0., 2000.])
M1_VEL = 300. * (-M1_POS0) / np.linalg.norm(M1_POS0)
M1_T_MAX = np.linalg.norm(M1_POS0) / 300.

FY1_INIT = np.array([17800., 0., 1800.])
FY2_INIT = np.array([12000., 1400., 1400.])
FY3_INIT = np.array([6000., -3000., 700.])

V_BOUNDS = (70., 140.)
THETA_BOUNDS = (0., 360.)
SMOKE_R = 10.0
SMOKE_R_SQ = SMOKE_R ** 2
SMOKE_SINK = 3.0
SMOKE_LIFE = 20.0
G = 9.8

# -------------- 圆柱采样 --------------
zs = np.linspace(TRUE_CYL_CENTRE[2], TRUE_CYL_CENTRE[2] + TRUE_CYL_H, NUM_CYL_Z_LAYERS)
angs = np.linspace(0, 2 * np.pi, NUM_CYL_POINTS_PER_LAYER, endpoint=False)
CYL_SAMPLES = np.array([[TRUE_CYL_CENTRE[0] + TRUE_CYL_R * math.cos(a),
                         TRUE_CYL_CENTRE[1] + TRUE_CYL_R * math.sin(a), z]
                        for z in zs for a in angs], dtype=np.float64)

# -------------- Numba 矢量化内核 --------------
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def _eval_kernel(t_arr, fy1_mask, xq2, yq2, zq2, td2, xq3, yq3, zq3, td3):
    """矢量化计算并集遮蔽时长，返回总秒数"""
    n = t_arr.size
    total = 0.0
    for i in prange(n):
        t = t_arr[i]
        if fy1_mask[i]:
            total += DT_SMOKE_CHECK
            continue
        mx = M1_POS0[0] + M1_VEL[0] * t
        my = M1_POS0[1] + M1_VEL[1] * t
        mz = M1_POS0[2] + M1_VEL[2] * t

        # FY2 烟幕
        sx2 = xq2
        sy2 = yq2
        sz2 = zq2 - SMOKE_SINK * (t - td2)
        fy2_ok = (td2 <= t <= td2 + SMOKE_LIFE) and sz2 >= 0
        # FY3 烟幕
        sx3 = xq3
        sy3 = yq3
        sz3 = zq3 - SMOKE_SINK * (t - td3)
        fy3_ok = (td3 <= t <= td3 + SMOKE_LIFE) and sz3 >= 0

        all_cyl = True
        for j in range(CYL_SAMPLES.shape[0]):
            cx, cy, cz = CYL_SAMPLES[j, 0], CYL_SAMPLES[j, 1], CYL_SAMPLES[j, 2]
            covered = False
            if fy2_ok:
                if _seg_hit(mx, my, mz, cx, cy, cz, sx2, sy2, sz2):
                    covered = True
            if not covered and fy3_ok:
                if _seg_hit(mx, my, mz, cx, cy, cz, sx3, sy3, sz3):
                    covered = True
            if not covered:
                all_cyl = False
                break
        if all_cyl:
            total += DT_SMOKE_CHECK
    return total

@njit(fastmath=True)
def _seg_hit(ax, ay, az, bx, by, bz, ox, oy, oz):
    abx, aby, abz = bx - ax, by - ay, bz - az
    aox, aoy, aoz = ax - ox, ay - oy, az - oz
    a = abx ** 2 + aby ** 2 + abz ** 2
    if a < 1e-16:
        return (aox ** 2 + aoy ** 2 + aoz ** 2) <= SMOKE_R_SQ + 1e-8
    b = 2.0 * (aox * abx + aoy * aby + aoz * abz)
    c = aox ** 2 + aoy ** 2 + aoz ** 2 - SMOKE_R_SQ
    d = b ** 2 - 4.0 * a * c
    if d < 0:
        return False
    sq = math.sqrt(d)
    s1, s2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
    return (s1 <= 1.0 + 1e-8) and (s2 >= -1e-8)

# -------------- 预计算 FY1 固定区间 --------------
FY1_THETA = 179.3012
FY1_V = 140.0
FY1_T1 = 0.0
FY1_DT = 0.7180

def pre_fy1():
    steps = int(M1_T_MAX / DT_SMOKE_CHECK) + 1
    mask = np.zeros(steps, dtype=np.bool_)
    theta = math.radians(FY1_THETA)
    cv, sv = math.cos(theta), math.sin(theta)
    tdet = FY1_T1 + FY1_DT
    xq = FY1_INIT[0] + FY1_V * tdet * cv
    yq = FY1_INIT[1] + FY1_V * tdet * sv
    zq = FY1_INIT[2] - 0.5 * G * FY1_DT ** 2
    t_arr = np.arange(steps) * DT_SMOKE_CHECK
    for i in range(steps):
        t = t_arr[i]
        mx = M1_POS0[0] + M1_VEL[0] * t
        my = M1_POS0[1] + M1_VEL[1] * t
        mz = M1_POS0[2] + M1_VEL[2] * t
        sx = xq
        sy = yq
        sz = zq - SMOKE_SINK * (t - tdet)
        if not (tdet <= t <= tdet + SMOKE_LIFE and sz >= 0):
            continue
        ok = True
        for cx, cy, cz in CYL_SAMPLES:
            if not _seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                ok = False
                break
        mask[i] = ok
    return t_arr, mask

T_ARR, FY1_MASK = pre_fy1()

# -------------- 适应度 = 矢量化内核调用 --------------
def fitness(x):
    theta2, v2, t2, dt2, theta3, v3, t3, dt3 = x
    if not (THETA_BOUNDS[0] <= theta2 <= THETA_BOUNDS[1] and
            V_BOUNDS[0] <= v2 <= V_BOUNDS[1] and
            THETA_BOUNDS[0] <= theta3 <= THETA_BOUNDS[1] and
            V_BOUNDS[0] <= v3 <= V_BOUNDS[1]):
        return 0.
    td2, td3 = t2 + dt2, t3 + dt3
    if min(t2, t3, dt2, dt3) < 0 or max(td2, td3) >= M1_T_MAX - 1e-8:
        return 0.
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    xq2 = FY2_INIT[0] + v2 * td2 * math.cos(theta2)
    yq2 = FY2_INIT[1] + v2 * td2 * math.sin(theta2)
    zq2 = FY2_INIT[2] - 0.5 * G * dt2 ** 2
    xq3 = FY3_INIT[0] + v3 * td3 * math.cos(theta3)
    yq3 = FY3_INIT[1] + v3 * td3 * math.sin(theta3)
    zq3 = FY3_INIT[2] - 0.5 * G * dt3 ** 2
    if zq2 < 0 or zq3 < 0:
        return 0.
    total = _eval_kernel(T_ARR, FY1_MASK, xq2, yq2, zq2, td2,
                         xq3, yq3, zq3, td3)
    if max(td2, td3) + SMOKE_LIFE >= 55.:
        total *= 1.05
    return total

# -------------- 多线程评估 --------------
def eval_pop(pop):
    fit = np.empty(len(pop))
    with ThreadPoolExecutor(max_workers=32) as ex:
        fut = {ex.submit(fitness, p): i for i, p in enumerate(pop)}
        for f in fut:
            fit[fut[f]] = f.result()
    return fit

# -------------- PSO 主流程 --------------
def main():
    print('=' * 70)
    print('2025 A-prob4 极速版 | FY1 固定 | 矢量化内核 | 进度条')
    print('=' * 70)
    # 初始化
    pop = np.empty((PSO_PARTICLE, 8))
    for i in range(PSO_PARTICLE):
        pop[i] = [uniform(*THETA_BOUNDS), uniform(*V_BOUNDS),
                  uniform(0, M1_T_MAX - 1), uniform(0, min(SMOKE_LIFE, M1_T_MAX)),
                  uniform(*THETA_BOUNDS), uniform(*V_BOUNDS),
                  uniform(0, M1_T_MAX - 1), uniform(0, min(SMOKE_LIFE, M1_T_MAX))]
    vel = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_fit = eval_pop(pop)
    gbest_idx = np.argmax(pbest_fit)
    gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]

    # 迭代
    with tqdm(range(1, PSO_ITER + 1), ncols=80) as bar:
        for itr in bar:
            w = W0 - (W0 - W1) * itr / PSO_ITER
            r1, r2 = np.random.rand(PSO_PARTICLE, 8), np.random.rand(PSO_PARTICLE, 8)
            vel = w * vel + C1 * r1 * (pbest - pop) + C2 * r2 * (gbest - pop)
            pop = np.clip(pop + vel, 0, None)
            # 边界修正
            pop[:, 0] %= 360
            pop[:, 1] = np.clip(pop[:, 1], *V_BOUNDS)
            pop[:, 2] = np.clip(pop[:, 2], 0, M1_T_MAX - 1e-3)
            pop[:, 3] = np.clip(pop[:, 3], 0, np.minimum(SMOKE_LIFE, M1_T_MAX - pop[:, 2]))
            pop[:, 4] %= 360
            pop[:, 5] = np.clip(pop[:, 5], *V_BOUNDS)
            pop[:, 6] = np.clip(pop[:, 6], 0, M1_T_MAX - 1e-3)
            pop[:, 7] = np.clip(pop[:, 7], 0, np.minimum(SMOKE_LIFE, M1_T_MAX - pop[:, 6]))

            fit = eval_pop(pop)
            mask = fit > pbest_fit
            pbest[mask], pbest_fit[mask] = pop[mask], fit[mask]
            best_idx = np.argmax(pbest_fit)
            if pbest_fit[best_idx] > gbest_fit:
                gbest, gbest_fit = pbest[best_idx].copy(), pbest_fit[best_idx]
            bar.set_postfix(best=f'{gbest_fit:.6f}s')

    # 结果
    theta2, v2, t2, dt2, theta3, v3, t3, dt3 = gbest
    print('\n' + '=' * 70)
    print('FY2/FY3 最优参数（FY1 已固定）')
    print('=' * 70)
    print(f'FY2  theta={theta2:.6f}°  v={v2:.6f}m/s  t2={t2:.6f}s  dt2={dt2:.6f}s')
    print(f'FY3  theta={theta3:.6f}°  v={v3:.6f}m/s  t3={t3:.6f}s  dt3={dt3:.6f}s')
    print(f'并集总遮蔽时长={gbest_fit:.6f}s')
    print('=' * 70)

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'总耗时：{time.time() - t0:.2f}s')