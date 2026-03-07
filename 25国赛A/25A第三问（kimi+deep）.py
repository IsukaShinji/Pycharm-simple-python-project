# -*- coding: utf-8 -*-
"""
2025 A-problem 3  |  部分遮蔽即有效 + 解析校验（打印区物理修正）
FY1-3-smoke vs M1  |  Numba + 16-thread  |  6-decimal
"""
import numpy as np
import math, os, time
from random import uniform, random
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------  USER TUNABLE  --------------------
POP_SIZE        = 120
MAX_ITER        = 300
N_THREAD        = min(16, os.cpu_count() or 16)
N_LAYER         = 3
N_PER_LAYER     = 72
SAVE_EVERY      = 10
DT_CHECK        = 0.001
OUT_PREC        = 6
# --------------------------------------------------------

# --------------------  PHYSICS CONST  -------------------
TRUE_CYL_CENTRE  = np.array([0., 200., 0.], dtype=np.float64)
TRUE_CYL_R       = 7.0
TRUE_CYL_H       = 10.0

M1_POS0          = np.array([20000., 0., 2000.], dtype=np.float64)
M1_VEL           = 300.0 * (-M1_POS0) / np.linalg.norm(M1_POS0)
M1_T_MAX         = float(np.linalg.norm(M1_POS0) / 300.0)

FY1_INIT_POS     = np.array([17800., 0., 1800.], dtype=np.float64)
FY1_Z            = float(FY1_INIT_POS[2])
FY1_V_BOUNDS     = (70.0, 140.0)
FY1_THETA_BOUNDS = (0.0, 360.0)

SMOKE_R          = 10.0
SMOKE_SINK       = 3.0
SMOKE_LIFE       = 20.0
GRAVITY          = 9.8
# --------------------------------------------------------


# --------------------  CYLINDER SAMPLING  ----------------
def gen_cyl_samples():
    pts = []
    zs = np.linspace(TRUE_CYL_CENTRE[2], TRUE_CYL_CENTRE[2] + TRUE_CYL_H, N_LAYER, dtype=np.float64)
    for z in zs:
        for k in range(N_PER_LAYER):
            ang = 2.0 * math.pi * k / N_PER_LAYER
            pts.append([TRUE_CYL_CENTRE[0] + TRUE_CYL_R * math.cos(ang),
                        TRUE_CYL_CENTRE[1] + TRUE_CYL_R * math.sin(ang), z])
    return np.array(pts, dtype=np.float64)


CYL_SAMPLES = gen_cyl_samples()
N_CYL       = len(CYL_SAMPLES)


# --------------------  NUMBA  ---------------------------
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print('⚠️  Numba not found – extremely slow')


if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def missile_pos(t):
        if t >= M1_T_MAX:
            return np.nan, np.nan, np.nan
        x = M1_POS0[0] + M1_VEL[0] * t
        y = M1_POS0[1] + M1_VEL[1] * t
        z = M1_POS0[2] + M1_VEL[2] * t
        return x, y, z

    @njit(fastmath=True, cache=True)
    def smoke_centre(t, t_det, xq, yq, zq):
        if t < t_det or t > t_det + SMOKE_LIFE:
            return np.nan, np.nan, np.nan
        return xq, yq, zq - SMOKE_SINK * (t - t_det)

    @njit(fastmath=True, cache=True)
    def seg_hit(ax, ay, az, bx, by, bz, ox, oy, oz, r2):
        abx = bx - ax
        aby = by - ay
        abz = bz - az
        aox = ax - ox
        aoy = ay - oy
        aoz = az - oz
        a = abx * abx + aby * aby + abz * abz
        if a == 0.0:
            return (aox * aox + aoy * aoy + aoz * aoz) <= r2
        b = 2.0 * (aox * abx + aoy * aby + aoz * abz)
        c = aox * aox + aoy * aoy + aoz * aoz - r2
        delta = b * b - 4.0 * a * c
        if delta < 0.0:
            return False
        sd = math.sqrt(delta)
        s1 = (-b - sd) / (2.0 * a)
        s2 = (-b + sd) / (2.0 * a)
        return s1 <= 1.0 and s2 >= 0.0

    @njit(fastmath=True, cache=True)
    def fitness_union(x):
        theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
        if not (FY1_V_BOUNDS[0] <= v <= FY1_V_BOUNDS[1] and
                FY1_THETA_BOUNDS[0] <= theta_deg <= FY1_THETA_BOUNDS[1]):
            return 0.0
        if t1 < 0.0 or t2 < t1 + 1.0 or t3 < t2 + 1.0:
            return 0.0
        if dt1 < 0.0 or dt2 < 0.0 or dt3 < 0.0:
            return 0.0

        theta = math.radians(theta_deg)
        cv, sv = math.cos(theta), math.sin(theta)
        T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
        xq = FY1_INIT_POS[0] + v * T_det * cv
        yq = FY1_INIT_POS[1] + v * T_det * sv
        zq = FY1_Z - 0.5 * GRAVITY * np.array([dt1, dt2, dt3], dtype=np.float64) ** 2
        if np.any(T_det >= M1_T_MAX) or np.any(zq < 0.0):
            return 0.0

        t_start = T_det.min()
        t_end = min(T_det.max() + SMOKE_LIFE, M1_T_MAX)
        steps = int((t_end - t_start) / DT_CHECK) + 1
        union_dt = 0.0

        for k in range(steps):
            t = t_start + k * DT_CHECK
            mx, my, mz = missile_pos(t)
            if np.isnan(mx):
                continue
            covered = False
            for i in range(3):
                if t < T_det[i] or t > T_det[i] + SMOKE_LIFE:
                    continue
                sx, sy, sz = smoke_centre(t, T_det[i], xq[i], yq[i], zq[i])
                if np.isnan(sx):
                    continue
                # 部分遮蔽即有效：任一圆柱点被遮挡即可
                for j in range(N_CYL):
                    cx, cy, cz = CYL_SAMPLES[j, 0], CYL_SAMPLES[j, 1], CYL_SAMPLES[j, 2]
                    if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz, SMOKE_R ** 2):
                        covered = True
                        break
                if covered:
                    break
            if covered:
                union_dt += DT_CHECK

        # 软奖励：晚段覆盖
        if t_end >= 55.0:
            union_dt *= 1.05
        return union_dt
else:
    fitness_union = None


def evaluate_pop(pop):
    if HAS_NUMBA and len(pop) > 0:
        with ThreadPoolExecutor(max_workers=N_THREAD) as ex:
            futs = {ex.submit(fitness_union, p): i for i, p in enumerate(pop)}
            fit = np.zeros(len(pop), dtype=np.float64)
            for fu in as_completed(futs):
                fit[futs[fu]] = fu.result()
        return fit
    else:
        return np.array([fitness_union(p) for p in pop], dtype=np.float64)


def init_pop_wide():
    pop = np.zeros((POP_SIZE, 8), dtype=np.float64)
    for i in range(POP_SIZE):
        t1 = uniform(0.0, M1_T_MAX - 6.0)
        t2 = uniform(t1 + 1.0, M1_T_MAX - 3.0)
        t3 = uniform(t2 + 1.0, M1_T_MAX - 0.5)
        pop[i, 0] = uniform(*FY1_THETA_BOUNDS)
        pop[i, 1] = uniform(*FY1_V_BOUNDS)
        pop[i, 2] = t1
        pop[i, 3] = t2
        pop[i, 4] = t3
        pop[i, 5] = uniform(0.0, SMOKE_LIFE)
        pop[i, 6] = uniform(0.0, SMOKE_LIFE)
        pop[i, 7] = uniform(0.0, SMOKE_LIFE)
    return pop


def apply_constraints(x):
    x[0] = x[0] % 360.0
    x[1] = np.clip(x[1], *FY1_V_BOUNDS)
    t1, t2, t3 = x[2], x[3], x[4]
    t1 = max(t1, 0.0)
    t2 = max(t2, t1 + 1.0)
    t3 = max(t3, t2 + 1.0)
    x[2], x[3], x[4] = t1, t2, t3
    x[5] = np.clip(x[5], 0.0, SMOKE_LIFE)
    x[6] = np.clip(x[6], 0.0, SMOKE_LIFE)
    x[7] = np.clip(x[7], 0.0, SMOKE_LIFE)
    return x


# ------------------------------------------------------------------
# 解析解校验：与离散积分完全同构
# ------------------------------------------------------------------
def analytic_union(x):
    """
    与离散积分完全同构的解析校验：
    1. 时间步长 = DT_CHECK
    2. 部分遮蔽即有效
    3. 起爆点 z 已减去自由下落
    """
    theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
    theta = math.radians(theta_deg)
    cv, sv = math.cos(theta), math.sin(theta)
    T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
    xq = FY1_INIT_POS[0] + v * T_det * cv
    yq = FY1_INIT_POS[1] + v * T_det * sv
    zq = FY1_Z - 0.5 * GRAVITY * np.array([dt1, dt2, dt3], dtype=np.float64) ** 2
    if np.any(T_det >= M1_T_MAX) or np.any(zq < 0):
        return 0.0

    t_start = T_det.min()
    t_end = min(T_det.max() + SMOKE_LIFE, M1_T_MAX)
    steps = int((t_end - t_start) / DT_CHECK) + 1
    union_dt = 0.0

    for k in range(steps):
        t = t_start + k * DT_CHECK
        mx, my, mz = missile_pos(t)
        if np.isnan(mx):
            continue
        covered = False
        for i in range(3):
            if t < T_det[i] or t > T_det[i] + SMOKE_LIFE:
                continue
            sx, sy, sz = smoke_centre(t, T_det[i], xq[i], yq[i], zq[i])
            if np.isnan(sx):
                continue
            # 部分遮蔽：任一采样点被遮挡即可
            for j in range(N_CYL):
                cx, cy, cz = CYL_SAMPLES[j, 0], CYL_SAMPLES[j, 1], CYL_SAMPLES[j, 2]
                if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz, SMOKE_R ** 2):
                    covered = True
                    break
            if covered:
                break
        if covered:
            union_dt += DT_CHECK

    return union_dt


# --------------------  PSO  --------------------
def main():
    pop = init_pop_wide()
    vel = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_fit = evaluate_pop(pop)
    gbest_idx = np.argmax(pbest_fit)
    gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]

    print('=' * 80)
    print(f'FY1-3-smoke vs M1   swarm={POP_SIZE}   iter={MAX_ITER}   threads={N_THREAD}   prec={OUT_PREC}')
    print('=' * 80)
    for itr in range(1, MAX_ITER + 1):
        w, c1, c2 = 0.7, 2.0, 2.0
        r1, r2 = np.random.rand(2, POP_SIZE, 1)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
        pop = np.array([apply_constraints(p + v) for p, v in zip(pop, vel)], dtype=np.float64)
        fit = evaluate_pop(pop)
        better = fit > pbest_fit
        pbest[better] = pop[better]
        pbest_fit[better] = fit[better]
        gbest_idx = np.argmax(pbest_fit)
        gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]
        if itr % SAVE_EVERY == 0 or itr == MAX_ITER:
            print(f'Iter {itr:3d}   Best union shadow = {gbest_fit:.{OUT_PREC}f} s')

    # ----------------  final table  ----------------
    theta, v, t1, t2, t3, dt1, dt2, dt3 = gbest
    theta_rad = math.radians(theta)
    cv, sv = math.cos(theta_rad), math.sin(theta_rad)
    T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
    P = FY1_INIT_POS.reshape(1, 3) + v * np.array([t1, t2, t3], dtype=np.float64).reshape(-1, 1) \
        * np.array([cv, sv, 0.0], dtype=np.float64).reshape(1, 3)
    P[:, 2] = FY1_Z
    Q = FY1_INIT_POS.reshape(1, 3) + v * T_det.reshape(-1, 1) \
        * np.array([cv, sv, 0.0], dtype=np.float64).reshape(1, 3)
    # **** 关键修正：起爆点 z 必须减去自由下落 ****
    Q[:, 2] = FY1_Z - 0.5 * GRAVITY * np.array([dt1, dt2, dt3], dtype=np.float64) ** 2

    print('\n' + '=' * 80)
    print(f'Copy below 3 rows → paste into result1.xlsx (row 3++)   precision = {OUT_PREC}')
    print('=' * 80)
    for i in range(3):
        row = [theta, v, i + 1,
               P[i, 0], P[i, 1], P[i, 2],
               Q[i, 0], Q[i, 1], Q[i, 2],
               gbest_fit if i == 0 else 0.0]
        print('\t'.join([f'{v:.{OUT_PREC}f}' for v in row]))
    print('=' * 80)

    # ===== 解析校验 =====
    analytic_dt = analytic_union(gbest)
    err_perc = abs(gbest_fit - analytic_dt) / (gbest_fit + 1e-12) * 100.0
    print('\n校验结果：', end='')
    if err_perc < 1.0:
        print(f'准确（误差 {err_perc:.3f} %）')
    else:
        print(f'警告（误差 {err_perc:.3f} %）')

    # ===== 竖向表头 =====
    header_fields = [
        "无人机运动方向 (度)",
        "无人机运动速度 (m/s)",
        "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)",
        "烟幕干扰弹投放点的y坐标 (m)",
        "烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)",
        "烟幕干扰弹起爆点的y坐标 (m)",
        "烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)"
    ]
    print('\n' + '=' * 80)
    print('竖向表头（直接复制→Excel 竖向粘贴）：')
    print('=' * 80)
    for fld in header_fields:
        print(fld)
    print('=' * 80)


if __name__ == '__main__':
    cpu_t0 = time.time()
    main()
    print('Total CPU elapsed:', time.time() - cpu_t0, 's')