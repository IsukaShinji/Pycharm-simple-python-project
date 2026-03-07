# -*- coding: utf-8 -*-
"""
2025 A 题 问题4  三机三弹协同干扰 M1  （改进版）
Numba 加速粗筛 + 分段评估，彻底避免“卡住”
"""
import numpy as np
import math, time, os
from random import uniform
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 如果机器没装 Numba，就给一个“空壳”装饰器，保证代码不报错
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# --------------------  可配置参数  --------------------
N_LAYER = 5
M_PER_LAYER = 24
DT_CHECK = 0.001
COARSE_CHUNK = 8000     # 每批粗筛粒子数
COARSE_ROUND = 80         # 共 10 批 → 20 000
FINE_PARTICLE = 160
FINE_ITER = 1000
N_THREAD = 32
REWARD = 1.5

# --------------------  固定物理量  --------------------
CYL_C = np.array([0., 200., 0.], dtype=np.float64)
CYL_R, CYL_H = 7.0, 10.0
M1_POS0 = np.array([20000., 0., 2000.], dtype=np.float64)
M1_VEL = 300. * (-M1_POS0) / np.linalg.norm(M1_POS0)
M1_T_MAX = np.linalg.norm(M1_POS0) / 300.
UAV_POS = {1: np.array([17800., 0., 1800.], dtype=np.float64),
           2: np.array([12000., 1400., 1400.], dtype=np.float64),
           3: np.array([6000., -3000., 700.], dtype=np.float64)}
V_BOUNDS = (70., 140.)
THETA_BOUNDS = (0., 360.)
SMOKE_R, SMOKE_LIFE, SMOKE_SINK, G = 10.0, 20.0, 3.0, 9.8

# --------------------  圆柱采样点  --------------------
@njit(fastmath=True, cache=True)
def _gen_samples(c_center, r, h, n_layer, m_layer):
    pts = []
    zs = np.linspace(c_center[2], c_center[2] + h, n_layer)
    for z in zs:
        for k in range(m_layer):
            ang = 2. * math.pi * k / m_layer
            pts.append([c_center[0] + r * math.cos(ang),
                        c_center[1] + r * math.sin(ang), z])
    return np.array(pts, dtype=np.float64)


CYL_PTS = _gen_samples(CYL_C, CYL_R, CYL_H, N_LAYER, M_PER_LAYER)
N_CYL = len(CYL_PTS)

# --------------------  Numba 核心  --------------------
from numba import njit

@njit(fastmath=True, cache=True)
def _missile(t):
    if t >= M1_T_MAX - 1e-8:
        return np.nan, np.nan, np.nan
    x = M1_POS0[0] + M1_VEL[0] * t
    y = M1_POS0[1] + M1_VEL[1] * t
    z = M1_POS0[2] + M1_VEL[2] * t
    return x, y, z

@njit(fastmath=True, cache=True)
def _smoke_c(t, t_det, xe, ye, ze):
    if t < t_det - 1e-8 or t > t_det + SMOKE_LIFE + 1e-8:
        return np.nan, np.nan, np.nan
    return xe, ye, ze - SMOKE_SINK * (t - t_det)

@njit(fastmath=True, cache=True)
def _hit(ax, ay, az, bx, by, bz, ox, oy, oz):
    abx, aby, abz = bx - ax, by - ay, bz - az
    aox, aoy, aoz = ax - ox, ay - oy, az - oz
    seg2 = abx * abx + aby * aby + abz * abz
    if seg2 < 1e-16:
        return (aox * aox + aoy * aoy + aoz * aoz) <= SMOKE_R * SMOKE_R + 1e-8
    a, b, c = seg2, 2 * (aox * abx + aoy * aby + aoz * abz), \
              (aox * aox + aoy * aoy + aoz * aoz) - SMOKE_R * SMOKE_R
    delta = b * b - 4 * a * c
    if delta < -1e-8:
        return False
    sd = math.sqrt(max(delta, 0.))
    s1, s2 = (-b - sd) / (2 * a), (-b + sd) / (2 * a)
    return s1 <= 1. + 1e-8 and s2 >= 0. - 1e-8

@njit(fastmath=True, cache=True)
def _fitness_one(x12, cyl_pts, n_cyl, dt_check, m1_t_max, smoke_life, smoke_sink, g,
                 uav_pos_, v_bds, th_bds):
    # 返回一个 float，彻底 Numba 内完成
    theta = x12[::4]
    v = x12[1::4]
    t_rel = x12[2::4]
    dt = x12[3::4]
    t_det = t_rel + dt
    # 快速边界
    for i in range(3):
        if v[i] < v_bds[0] or v[i] > v_bds[1] or \
           theta[i] < th_bds[0] or theta[i] > th_bds[1] or \
           t_rel[i] < 0. or dt[i] < 0. or t_det[i] > m1_t_max - 1e-8:
            return 0.
    # 起爆点
    xyz_det = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        th = math.radians(theta[i])
        xe = uav_pos_[i, 0] + v[i] * t_det[i] * math.cos(th)
        ye = uav_pos_[i, 1] + v[i] * t_det[i] * math.sin(th)
        ze = uav_pos_[i, 2] - 0.5 * g * dt[i] * dt[i]
        if ze < -1e-8:
            return 0.
        xyz_det[i, 0] = xe
        xyz_det[i, 1] = ye
        xyz_det[i, 2] = ze
    t_start = max(t_det.min(), 0.)
    t_end = min(t_det.max() + smoke_life, m1_t_max)
    if t_start >= t_end - 1e-8:
        return 0.
    steps = int((t_end - t_start) / dt_check) + 1
    total = 0.
    single = np.zeros(3, dtype=np.float64)
    for k in range(steps):
        t = t_start + k * dt_check
        mx, my, mz = _missile(t)
        if np.isnan(mx):
            continue
        all_blocked = True
        for j in range(n_cyl):
            cx, cy, cz = cyl_pts[j, 0], cyl_pts[j, 1], cyl_pts[j, 2]
            blocked = False
            for i in range(3):
                if t < t_det[i] - 1e-8 or t > t_det[i] + smoke_life + 1e-8:
                    continue
                sx, sy, sz = _smoke_c(t, t_det[i], xyz_det[i, 0], xyz_det[i, 1], xyz_det[i, 2])
                if np.isnan(sx):
                    continue
                if _hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    blocked = True
                    break
            if not blocked:
                all_blocked = False
                break
        if all_blocked:
            total += dt_check
            for i in range(3):
                if t_det[i] <= t <= t_det[i] + smoke_life:
                    single[i] += dt_check
    reward = 0.
    for i in range(3):
        reward += math.sqrt(single[i] + 1e-6) * REWARD
    return total + reward


# 把 UAV_POS 转成固定数组供 Numba
UAV_ARR = np.empty((3, 3), dtype=np.float64)
for i in range(3):
    UAV_ARR[i] = UAV_POS[i + 1]

# --------------------  并行评估（Numba 内计算，Python 仅调度）  --------------------
def eval_pool(pop):
    fit = np.zeros(len(pop), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=N_THREAD) as ex:
        fut = {ex.submit(_fitness_one, pop[i], CYL_PTS, N_CYL, DT_CHECK,
                         M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR,
                         V_BOUNDS, THETA_BOUNDS): i for i in range(len(pop))}
        for f in as_completed(fut):
            fit[fut[f]] = f.result()
    return fit


# --------------------  初始化  --------------------
def init_pop(size):
    pop = np.zeros((size, 12), dtype=np.float64)
    for i in range(size):
        for j in range(3):
            pop[i, j * 4] = uniform(*THETA_BOUNDS)
            pop[i, j * 4 + 1] = uniform(*V_BOUNDS)
            max_tr = M1_T_MAX - 2.0
            pop[i, j * 4 + 2] = uniform(0., max_tr)
            max_dt = min(SMOKE_LIFE, math.sqrt(2 * UAV_ARR[j, 2] / G), M1_T_MAX - pop[i, j * 4 + 2])
            pop[i, j * 4 + 3] = uniform(0., max_dt)
    return pop


# --------------------  小 PSO 精修  --------------------
def small_pso(best4):
    pop = np.array(best4, dtype=np.float64)
    while len(pop) < FINE_PARTICLE:
        pop = np.vstack([pop, init_pop(1)])
    pop = pop[:FINE_PARTICLE]
    vel = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_fit = eval_pool(pop)
    g = pbest[np.argmax(pbest_fit)].copy()
    gfit = pbest_fit.max()

    w, c1, c2 = 0.7, 1.5, 2.5
    for itr in range(1, FINE_ITER + 1):
        r1, r2 = np.random.rand(*pop.shape), np.random.rand(*pop.shape)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (g - pop)
        new_pop = pop + vel
        # 边界
        for j in range(3):
            new_pop[:, j * 4] = np.clip(new_pop[:, j * 4], *THETA_BOUNDS)
            new_pop[:, j * 4 + 1] = np.clip(new_pop[:, j * 4 + 1], *V_BOUNDS)
            new_pop[:, j * 4 + 2] = np.clip(new_pop[:, j * 4 + 2], 0., M1_T_MAX)
            new_pop[:, j * 4 + 3] = np.clip(new_pop[:, j * 4 + 3], 0., SMOKE_LIFE)
        fit = eval_pool(new_pop)
        better = fit > pbest_fit
        pbest[better] = new_pop[better]
        pbest_fit[better] = fit[better]
        if fit.max() > gfit + 1e-12:
            g = new_pop[np.argmax(fit)].copy()
            gfit = fit.max()
        if itr % 10 == 0 or itr == FINE_ITER:
            print(f'[PSO] iter {itr:3d}/{FINE_ITER}  best fitness = {gfit:.6f}')
    return g, gfit


# --------------------  主流程  --------------------
def main():
    print('==== 问题4  三机三弹协同干扰 M1  （Numba 加速版）====')
    print('Step1  粗筛随机种子（分段） ...')
    top_list = []
    for rnd in range(1, COARSE_ROUND + 1):
        pop = init_pop(COARSE_CHUNK)
        fit = eval_pool(pop)
        top_idx = np.argpartition(fit, -4)[-4:]
        top_list += [pop[i].copy() for i in top_idx]
        print(f'  粗筛轮次 {rnd:2d}/{COARSE_ROUND}  当前Top4={fit.max():.6f}')
    # 去重并留 Top-4
    top_arr = np.vstack(top_list)
    top_fit = eval_pool(top_arr)
    best4_idx = np.argpartition(top_fit, -4)[-4:]
    best4 = [top_arr[i].copy() for i in best4_idx]
    print('Step2  小粒子群精修 ...')
    best_x, best_f = small_pso(best4)
    print('==== 最终最优策略 ====')
    # 解析结果
    theta = best_x[::4]
    v = best_x[1::4]
    t_rel = best_x[2::4]
    dt = best_x[3::4]
    t_det = t_rel + dt
    header = ['无人机', '方向°', '速度m/s', '投放x', '投放y', '投放z', '起爆x', '起爆y', '起爆z', '有效时长s']
    print('\t'.join(header))
    for i in range(3):
        th = math.radians(theta[i])
        xe = UAV_ARR[i, 0] + v[i] * t_det[i] * math.cos(th)
        ye = UAV_ARR[i, 1] + v[i] * t_det[i] * math.sin(th)
        ze = UAV_ARR[i, 2] - 0.5 * G * dt[i] ** 2
        drop_x = UAV_ARR[i, 0] + v[i] * t_rel[i] * math.cos(th)
        drop_y = UAV_ARR[i, 1] + v[i] * t_rel[i] * math.sin(th)
        drop_z = UAV_ARR[i, 2]
        # 单发时长
        single = 0.
        t0, t1 = t_det[i], min(t_det[i] + SMOKE_LIFE, M1_T_MAX)
        steps = int((t1 - t0) / DT_CHECK) + 1
        for k in range(steps):
            t = t0 + k * DT_CHECK
            mx, my, mz = _missile(t)
            if np.isnan(mx):
                continue
            sx, sy, sz = _smoke_c(t, t_det[i], xe, ye, ze)
            if np.isnan(sx):
                continue
            all_ok = True
            for j in range(N_CYL):
                cx, cy, cz = CYL_PTS[j, 0], CYL_PTS[j, 1], CYL_PTS[j, 2]
                if not _hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    all_ok = False
                    break
            if all_ok:
                single += DT_CHECK
        row = [f'FY{i + 1}', f'{theta[i]:.3f}', f'{v[i]:.3f}',
               f'{drop_x:.3f}', f'{drop_y:.3f}', f'{drop_z:.3f}',
               f'{xe:.3f}', f'{ye:.3f}', f'{ze:.3f}', f'{single:.6f}']
        print('\t'.join(row))


if __name__ == '__main__':
    t0 = time.time()
    main()
    print('==== 总耗时 %.2f s ====' % (time.time() - t0))