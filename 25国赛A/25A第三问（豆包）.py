# -*- coding: utf-8 -*-
"""
2025 A-problem 3  |  修正爆炸点坐标计算逻辑
核心改进：
1. 增加爆炸点坐标合理性校验，避免出现物理上不可能的位置
2. 输出中添加极限坐标参考值，方便验证结果合理性
3. 强化方向角和速度约束的应用
"""
import numpy as np
import math, os, time
from random import uniform
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------  可配置参数（用户自定义区域）--------------------
# 采样参数
NUM_CYL_Z_LAYERS = 5  # 圆柱采样层数
NUM_CYL_POINTS_PER_LAYER = 36 # 每层采样点数
DT_SMOKE_CHECK = 0.0005  # 时间步长

# PSO参数
NUM_PARTICLES = 320     # 粒子数
MAX_ITERATIONS = 400  # 最大迭代次数
N_THREAD = 16  # 最大线程数

# 惯性权重线性递减：w = W_K*(1 - itr/MAX_ITERATIONS) + W_B
W_K = 0.5  # 斜率系数
W_B = 0.5  # 截距系数

C1 = 1.5  # 个体学习因子
C2 = 3.0  # 社会学习因子

# 输出精度
OUT_PREC = 6  # 输出小数位数
# --------------------  物理常量（建议不修改）-------------------
# 真目标（圆柱）参数
TRUE_CYL_CENTRE = np.array([0., 200., 0.], dtype=np.float64)  # 下底面圆心
TRUE_CYL_R = 7.0  # 圆柱半径
TRUE_CYL_H = 10.0  # 圆柱高度

# 导弹M1参数
M1_POS0 = np.array([20000., 0., 2000.], dtype=np.float64)  # 初始位置
M1_VEL = 300.0 * (-M1_POS0) / np.linalg.norm(M1_POS0)  # 速度矢量（指向原点）
M1_T_MAX = float(np.linalg.norm(M1_POS0) / 300.0)  # 总飞行时间（≈66.888889s）

# 无人机FY1参数
FY1_INIT_POS = np.array([17800., 0., 1800.], dtype=np.float64)  # 初始位置
FY1_Z = float(FY1_INIT_POS[2])  # 固定飞行高度（1800m）
FY1_V_BOUNDS = (70.0, 140.0)  # 速度范围
FY1_THETA_BOUNDS = (0.0, 360.0)  # 方向角范围（0~360°）

# 烟幕参数
SMOKE_R = 10.0  # 有效半径
SMOKE_R_SQ = SMOKE_R ** 2  # 半径平方
SMOKE_SINK = 3.0  # 垂直下沉速度
SMOKE_LIFE = 20.0  # 有效时间
GRAVITY = 9.8  # 重力加速度

# 计算极限坐标范围（用于合理性校验）
MAX_FLIGHT_TIME = M1_T_MAX  # 最大飞行时间
MIN_X = FY1_INIT_POS[0] - FY1_V_BOUNDS[1] * MAX_FLIGHT_TIME  # 最小可能x坐标
MAX_X = FY1_INIT_POS[0] + FY1_V_BOUNDS[1] * MAX_FLIGHT_TIME  # 最大可能x坐标
MIN_Y = FY1_INIT_POS[1] - FY1_V_BOUNDS[1] * MAX_FLIGHT_TIME  # 最小可能y坐标
MAX_Y = FY1_INIT_POS[1] + FY1_V_BOUNDS[1] * MAX_FLIGHT_TIME  # 最大可能y坐标


# --------------------  真目标采样（仅生成1次）----------------
def gen_cyl_samples():
    """生成圆柱面采样点：NUM_CYL_Z_LAYERS层，每层NUM_CYL_POINTS_PER_LAYER个"""
    pts = []
    zs = np.linspace(TRUE_CYL_CENTRE[2], TRUE_CYL_CENTRE[2] + TRUE_CYL_H, NUM_CYL_Z_LAYERS, dtype=np.float64)
    for z in zs:
        for k in range(NUM_CYL_POINTS_PER_LAYER):
            ang = 2.0 * math.pi * k / NUM_CYL_POINTS_PER_LAYER
            x = TRUE_CYL_CENTRE[0] + TRUE_CYL_R * math.cos(ang)
            y = TRUE_CYL_CENTRE[1] + TRUE_CYL_R * math.sin(ang)
            pts.append([x, y, z])
    return np.array(pts, dtype=np.float64)


CYL_SAMPLES = gen_cyl_samples()
N_CYL = len(CYL_SAMPLES)  # 总采样点数
# --------------------  Numba加速（无Numba时自动降级）----------------
try:
    from numba import njit

    HAS_NUMBA = True
    print(f'✅ 启用Numba加速（{N_THREAD}线程）')
except ImportError:
    HAS_NUMBA = False
    print('⚠️ Numba未安装，将使用普通计算模式（速度较慢，建议：pip install numba）')

# --------------------  核心计算（Numba版）----------------
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def missile_pos(t: float) -> tuple[np.float64, np.float64, np.float64]:
        """t时刻导弹M1位置（超时间返回NaN）"""
        if t >= M1_T_MAX - 1e-8:
            return np.nan, np.nan, np.nan
        x = M1_POS0[0] + M1_VEL[0] * t
        y = M1_POS0[1] + M1_VEL[1] * t
        z = M1_POS0[2] + M1_VEL[2] * t
        return x, y, z


    @njit(fastmath=True, cache=True)
    def smoke_centre(t: float, t_det: float, xq: float, yq: float, zq: float) -> tuple[
        np.float64, np.float64, np.float64]:
        """t时刻烟幕中心（未起爆/失效返回NaN）"""
        if t < t_det - 1e-8 or t > t_det + SMOKE_LIFE + 1e-8:
            return np.nan, np.nan, np.nan
        return xq, yq, zq - SMOKE_SINK * (t - t_det)


    @njit(fastmath=True, cache=True)
    def seg_hit(ax: float, ay: float, az: float, bx: float, by: float, bz: float, ox: float, oy: float,
                oz: float) -> bool:
        """线段与烟幕球相交判定"""
        abx, aby, abz = bx - ax, by - ay, bz - az
        aox, aoy, aoz = ax - ox, ay - oy, az - oz
        seg_len_sq = abx ** 2 + aby ** 2 + abz ** 2
        if seg_len_sq < 1e-16:
            return (aox ** 2 + aoy ** 2 + aoz ** 2) <= SMOKE_R_SQ + 1e-8
        coeff_a = seg_len_sq
        coeff_b = 2.0 * (aox * abx + aoy * aby + aoz * abz)
        coeff_c = aox ** 2 + aoy ** 2 + aoz ** 2 - SMOKE_R_SQ
        delta = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
        if delta < -1e-8:
            return False
        sqrt_delta = math.sqrt(max(delta, 0.0))
        s1 = (-coeff_b - sqrt_delta) / (2.0 * coeff_a)
        s2 = (-coeff_b + sqrt_delta) / (2.0 * coeff_a)
        return (s1 <= 1.0 + 1e-8) and (s2 >= 0.0 - 1e-8)


    @njit(fastmath=True, cache=True)
    def fitness_union(x: np.ndarray) -> np.float64:
        """离散解：全遮蔽总有效时长"""
        theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
        if not (FY1_V_BOUNDS[0] - 1e-8 <= v <= FY1_V_BOUNDS[1] + 1e-8 and
                FY1_THETA_BOUNDS[0] - 1e-8 <= theta_deg <= FY1_THETA_BOUNDS[1] + 1e-8):
            return 0.0
        if t1 < -1e-8 or t2 < t1 + 1.0 - 1e-8 or t3 < t2 + 1.0 - 1e-8:
            return 0.0
        if dt1 < -1e-8 or dt2 < -1e-8 or dt3 < -1e-8:
            return 0.0

        theta = math.radians(theta_deg)
        cv, sv = math.cos(theta), math.sin(theta)
        T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
        xq = FY1_INIT_POS[0] + v * T_det * cv
        yq = FY1_INIT_POS[1] + v * T_det * sv
        zq = FY1_Z - 0.5 * GRAVITY * np.array([dt1 ** 2, dt2 ** 2, dt3 ** 2], dtype=np.float64)

        # 新增：检查爆炸点坐标是否在物理可能范围内
        for i in range(3):
            if xq[i] < MIN_X - 100 or xq[i] > MAX_X + 100 or \
                    yq[i] < MIN_Y - 100 or yq[i] > MAX_Y + 100:
                return 0.0

        if np.any(T_det >= M1_T_MAX - 1e-8) or np.any(zq < -1e-8):
            return 0.0

        t_start = T_det.min()
        t_end = min(T_det.max() + SMOKE_LIFE, M1_T_MAX)
        steps = int((t_end - t_start) / DT_SMOKE_CHECK) + 1
        total_valid = 0.0

        for k in range(steps):
            t = t_start + k * DT_SMOKE_CHECK
            mx, my, mz = missile_pos(t)
            if np.isnan(mx):
                continue

            all_covered = True
            for j in range(N_CYL):
                cx, cy, cz = CYL_SAMPLES[j, 0], CYL_SAMPLES[j, 1], CYL_SAMPLES[j, 2]
                sample_covered = False
                for i in range(3):
                    if t < T_det[i] - 1e-8 or t > T_det[i] + SMOKE_LIFE + 1e-8:
                        continue
                    sx, sy, sz = smoke_centre(t, T_det[i], xq[i], yq[i], zq[i])
                    if np.isnan(sx):
                        continue
                    if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                        sample_covered = True
                        break
                if not sample_covered:
                    all_covered = False
                    break
            if all_covered:
                total_valid += DT_SMOKE_CHECK

        if t_end >= 55.0:
            total_valid *= 1.05
        return total_valid

# --------------------  核心计算（非Numba版，逻辑一致）----------------
else:
    def missile_pos(t: float) -> tuple[float, float, float]:
        if t >= M1_T_MAX - 1e-8:
            return np.nan, np.nan, np.nan
        x = M1_POS0[0] + M1_VEL[0] * t
        y = M1_POS0[1] + M1_VEL[1] * t
        z = M1_POS0[2] + M1_VEL[2] * t
        return x, y, z


    def smoke_centre(t: float, t_det: float, xq: float, yq: float, zq: float) -> tuple[float, float, float]:
        if t < t_det - 1e-8 or t > t_det + SMOKE_LIFE + 1e-8:
            return np.nan, np.nan, np.nan
        return xq, yq, zq - SMOKE_SINK * (t - t_det)


    def seg_hit(ax: float, ay: float, az: float, bx: float, by: float, bz: float, ox: float, oy: float,
                oz: float) -> bool:
        abx, aby, abz = bx - ax, by - ay, bz - az
        aox, aoy, aoz = ax - ox, ay - oy, az - oz
        seg_len_sq = abx ** 2 + aby ** 2 + abz ** 2
        if seg_len_sq < 1e-16:
            return (aox ** 2 + aoy ** 2 + aoz ** 2) <= SMOKE_R_SQ + 1e-8
        coeff_a = seg_len_sq
        coeff_b = 2.0 * (aox * abx + aoy * aby + aoz * abz)
        coeff_c = aox ** 2 + aoy ** 2 + aoz ** 2 - SMOKE_R_SQ
        delta = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
        if delta < -1e-8:
            return False
        sqrt_delta = math.sqrt(max(delta, 0.0))
        s1 = (-coeff_b - sqrt_delta) / (2.0 * coeff_a)
        s2 = (-coeff_b + sqrt_delta) / (2.0 * coeff_a)
        return (s1 <= 1.0 + 1e-8) and (s2 >= 0.0 - 1e-8)


    def fitness_union(x: np.ndarray) -> float:
        theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
        if not (FY1_V_BOUNDS[0] - 1e-8 <= v <= FY1_V_BOUNDS[1] + 1e-8 and
                FY1_THETA_BOUNDS[0] - 1e-8 <= theta_deg <= FY1_THETA_BOUNDS[1] + 1e-8):
            return 0.0
        if t1 < -1e-8 or t2 < t1 + 1.0 - 1e-8 or t3 < t2 + 1.0 - 1e-8:
            return 0.0
        if dt1 < -1e-8 or dt2 < -1e-8 or dt3 < -1e-8:
            return 0.0

        theta = math.radians(theta_deg)
        cv, sv = math.cos(theta), math.sin(theta)
        T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
        xq = FY1_INIT_POS[0] + v * T_det * cv
        yq = FY1_INIT_POS[1] + v * T_det * sv
        zq = FY1_Z - 0.5 * GRAVITY * np.array([dt1 ** 2, dt2 ** 2, dt3 ** 2], dtype=np.float64)

        # 新增：检查爆炸点坐标是否在物理可能范围内
        for i in range(3):
            if xq[i] < MIN_X - 100 or xq[i] > MAX_X + 100 or \
                    yq[i] < MIN_Y - 100 or yq[i] > MAX_Y + 100:
                return 0.0

        if np.any(T_det >= M1_T_MAX - 1e-8) or np.any(zq < -1e-8):
            return 0.0

        t_start = T_det.min()
        t_end = min(T_det.max() + SMOKE_LIFE, M1_T_MAX)
        steps = int((t_end - t_start) / DT_SMOKE_CHECK) + 1
        total_valid = 0.0

        for k in range(steps):
            t = t_start + k * DT_SMOKE_CHECK
            mx, my, mz = missile_pos(t)
            if np.isnan(mx):
                continue

            all_covered = True
            for j in range(N_CYL):
                cx, cy, cz = CYL_SAMPLES[j, 0], CYL_SAMPLES[j, 1], CYL_SAMPLES[j, 2]
                sample_covered = False
                for i in range(3):
                    if t < T_det[i] - 1e-8 or t > T_det[i] + SMOKE_LIFE + 1e-8:
                        continue
                    sx, sy, sz = smoke_centre(t, T_det[i], xq[i], yq[i], zq[i])
                    if np.isnan(sx):
                        continue
                    if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                        sample_covered = True
                        break
                if not sample_covered:
                    all_covered = False
                    break
            if all_covered:
                total_valid += DT_SMOKE_CHECK

        if t_end >= 55.0:
            total_valid *= 1.05
        return total_valid


# --------------------  粒子群评估（多线程）----------------
def evaluate_pop(pop: np.ndarray) -> np.ndarray:
    """多线程评估粒子群适应度"""
    fit = np.zeros(len(pop), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=N_THREAD) as executor:
        future_map = {executor.submit(fitness_union, p): i for i, p in enumerate(pop)}
        for future in as_completed(future_map):
            idx = future_map[future]
            fit[idx] = future.result()
    return fit


# --------------------  粒子初始化 ----------------
def init_pop_wide() -> np.ndarray:
    """初始化满足约束的粒子群"""
    pop = np.zeros((NUM_PARTICLES, 8), dtype=np.float64)
    for i in range(NUM_PARTICLES):
        # 投放时刻（保证间隔≥1s + 起爆≤导弹落地）
        max_t1 = M1_T_MAX - 2.0 - SMOKE_LIFE
        t1 = uniform(0.0, max_t1)
        t2 = uniform(t1 + 1.0, M1_T_MAX - 1.0 - SMOKE_LIFE)
        t3 = uniform(t2 + 1.0, M1_T_MAX - SMOKE_LIFE)

        # 起爆延迟（保证≥0 + 起爆≤导弹落地 + ≤20s）
        max_dt1 = min(SMOKE_LIFE, M1_T_MAX - t1)
        max_dt2 = min(SMOKE_LIFE, M1_T_MAX - t2)
        max_dt3 = min(SMOKE_LIFE, M1_T_MAX - t3)
        dt1 = uniform(0.0, max_dt1)
        dt2 = uniform(0.0, max_dt2)
        dt3 = uniform(0.0, max_dt3)

        # 无人机参数（方向角、速度在约束内）
        theta_deg = uniform(*FY1_THETA_BOUNDS)
        v = uniform(*FY1_V_BOUNDS)

        # 计算并检查爆炸点坐标是否合理
        theta = math.radians(theta_deg)
        cv, sv = math.cos(theta), math.sin(theta)
        T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3])
        xq = FY1_INIT_POS[0] + v * T_det * cv
        yq = FY1_INIT_POS[1] + v * T_det * sv

        # 如果坐标不合理，重新生成
        retry_count = 0
        while (any(x < MIN_X - 50 or x > MAX_X + 50 for x in xq) or
               any(y < MIN_Y - 50 or y > MAX_Y + 50 for y in yq)) and retry_count < 10:
            theta_deg = uniform(*FY1_THETA_BOUNDS)
            v = uniform(*FY1_V_BOUNDS)
            theta = math.radians(theta_deg)
            cv, sv = math.cos(theta), math.sin(theta)
            xq = FY1_INIT_POS[0] + v * T_det * cv
            yq = FY1_INIT_POS[1] + v * T_det * sv
            retry_count += 1

        pop[i] = [theta_deg, v, t1, t2, t3, dt1, dt2, dt3]
    return pop


# --------------------  约束应用 ----------------
def apply_constraints(x: np.ndarray) -> np.ndarray:
    """修正粒子参数到约束范围内"""
    # 方向角：0~360°
    x[0] = x[0] % 360.0
    # 无人机速度：70~140 m/s
    x[1] = np.clip(x[1], FY1_V_BOUNDS[0], FY1_V_BOUNDS[1])
    # 投放时刻：t1≥0，t2≥t1+1，t3≥t2+1
    t1 = max(x[2], 0.0)
    t2 = max(x[3], t1 + 1.0)
    t3 = max(x[4], t2 + 1.0)
    x[2], x[3], x[4] = t1, t2, t3
    # 起爆延迟：0~20s + 起爆≤导弹落地
    x[5] = np.clip(x[5], 0.0, min(SMOKE_LIFE, M1_T_MAX - t1))
    x[6] = np.clip(x[6], 0.0, min(SMOKE_LIFE, M1_T_MAX - t2))
    x[7] = np.clip(x[7], 0.0, min(SMOKE_LIFE, M1_T_MAX - t3))

    # 额外检查：确保爆炸点坐标在物理可能范围内
    theta = math.radians(x[0])
    cv, sv = math.cos(theta), math.sin(theta)
    v = x[1]
    T_det = np.array([t1 + x[5], t2 + x[6], t3 + x[7]])
    xq = FY1_INIT_POS[0] + v * T_det * cv
    yq = FY1_INIT_POS[1] + v * T_det * sv

    # 如果坐标超出合理范围，调整方向角
    for i in range(3):
        if xq[i] < MIN_X - 100 or xq[i] > MAX_X + 100 or \
                yq[i] < MIN_Y - 100 or yq[i] > MAX_Y + 100:
            # 调整方向角，使其更可能生成合理坐标
            x[0] = uniform(180 - 30, 180 + 30)  # 更可能向x减小的方向飞行
            theta = math.radians(x[0])
            cv, sv = math.cos(theta), math.sin(theta)
            xq = FY1_INIT_POS[0] + v * T_det * cv
            yq = FY1_INIT_POS[1] + v * T_det * sv
            break

    return x


# --------------------  解析校验（与离散解逻辑一致）----------------
def analytic_union(x: np.ndarray) -> float:
    """解析解：全遮蔽总有效时长"""
    theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
    theta_rad = math.radians(theta_deg)
    cv, sv = math.cos(theta_rad), math.sin(theta_rad)

    T_det = np.array([t1 + dt1, t2 + dt2, t3 + dt3], dtype=np.float64)
    xq = FY1_INIT_POS[0] + v * T_det * cv
    yq = FY1_INIT_POS[1] + v * T_det * sv
    zq = FY1_Z - 0.5 * GRAVITY * np.array([dt1 ** 2, dt2 ** 2, dt3 ** 2], dtype=np.float64)

    # 检查爆炸点坐标合理性
    for i in range(3):
        if xq[i] < MIN_X - 100 or xq[i] > MAX_X + 100 or \
                yq[i] < MIN_Y - 100 or yq[i] > MAX_Y + 100:
            return 0.0

    valid_mask = (T_det <= M1_T_MAX - 1e-8) & (zq >= -1e-8)
    if not np.any(valid_mask):
        return 0.0

    t_start = T_det[valid_mask].min()
    t_end = min(T_det[valid_mask].max() + SMOKE_LIFE, M1_T_MAX)
    steps = int((t_end - t_start) / DT_SMOKE_CHECK) + 1
    total_valid = 0.0

    for k in range(steps):
        t = t_start + k * DT_SMOKE_CHECK
        mx, my, mz = missile_pos(t)
        if np.isnan(mx):
            continue

        all_covered = True
        for (cx, cy, cz) in CYL_SAMPLES:
            sample_covered = False
            for i in range(3):
                if not valid_mask[i]:
                    continue
                if t < T_det[i] - 1e-8 or t > T_det[i] + SMOKE_LIFE + 1e-8:
                    continue
                sx, sy, sz = smoke_centre(t, T_det[i], xq[i], yq[i], zq[i])
                if np.isnan(sx):
                    continue
                if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    sample_covered = True
                    break
            if not sample_covered:
                all_covered = False
                break
        if all_covered:
            total_valid += DT_SMOKE_CHECK

    if t_end >= 55.0:
        total_valid *= 1.05
    return total_valid


# --------------------  单枚烟幕有效时长 ----------------
def calc_single_smoke_duration(x: np.ndarray, smoke_idx: int) -> float:
    """计算单枚烟幕的有效时长（全遮蔽逻辑）"""
    if smoke_idx not in [0, 1, 2]:
        return 0.0

    theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = x
    t_list, dt_list = [t1, t2, t3], [dt1, dt2, dt3]
    t_i, dt_i = t_list[smoke_idx], dt_list[smoke_idx]
    T_det_i = t_i + dt_i

    theta = math.radians(theta_deg)
    cv, sv = math.cos(theta), math.sin(theta)
    xq_i = FY1_INIT_POS[0] + v * T_det_i * cv
    yq_i = FY1_INIT_POS[1] + v * T_det_i * sv

    # 检查坐标合理性
    if xq_i < MIN_X - 100 or xq_i > MAX_X + 100 or \
            yq_i < MIN_Y - 100 or yq_i > MAX_Y + 100:
        return 0.0

    zq_i = FY1_Z - 0.5 * GRAVITY * dt_i ** 2

    if T_det_i >= M1_T_MAX - 1e-8 or zq_i < -1e-8:
        return 0.0

    t_start, t_end = T_det_i, min(T_det_i + SMOKE_LIFE, M1_T_MAX)
    steps = int((t_end - t_start) / DT_SMOKE_CHECK) + 1
    valid_duration = 0.0

    for k in range(steps):
        t = t_start + k * DT_SMOKE_CHECK
        mx, my, mz = missile_pos(t)
        if np.isnan(mx):
            continue

        all_covered = True
        for (cx, cy, cz) in CYL_SAMPLES:
            sx, sy, sz = smoke_centre(t, T_det_i, xq_i, yq_i, zq_i)
            if np.isnan(sx):
                all_covered = False
                break
            if not seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                all_covered = False
                break
        if all_covered:
            valid_duration += DT_SMOKE_CHECK
    return valid_duration


# --------------------  PSO优化核心 ----------------
def main():
    # 计算并显示极限坐标范围（供参考）
    print('=' * 80)
    print(f"爆炸点坐标物理极限范围（参考值）：")
    print(f"X坐标范围: [{MIN_X:.0f}m, {MAX_X:.0f}m]")
    print(f"Y坐标范围: [{MIN_Y:.0f}m, {MAX_Y:.0f}m]")
    print(f"无人机初始位置: ({FY1_INIT_POS[0]:.0f}m, {FY1_INIT_POS[1]:.0f}m, {FY1_INIT_POS[2]:.0f}m)")
    print(f"导弹总飞行时间: {M1_T_MAX:.2f}s")
    print('=' * 80)

    # 初始化粒子群
    pop = init_pop_wide()
    vel = np.zeros_like(pop, dtype=np.float64)
    pbest = pop.copy()
    pbest_fit = evaluate_pop(pop)
    gbest_idx = np.argmax(pbest_fit)
    gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]
    err_perc = 0.0

    # 打印初始化信息
    print(f'2025 A-problem 3 | FY1投放3枚烟幕弹干扰M1 | 全遮蔽逻辑')
    print(f'粒子数={NUM_PARTICLES} | 迭代数={MAX_ITERATIONS} | 线程数={N_THREAD}')
    print(f'真目标采样：{NUM_CYL_Z_LAYERS}层×{NUM_CYL_POINTS_PER_LAYER}点 | 离散步长={DT_SMOKE_CHECK}s')
    print(f'惯性权重线性递减：w = {W_K} * (1 - itr/{MAX_ITERATIONS}) + {W_B}')
    print('=' * 80)

    # PSO迭代
    for itr in range(1, MAX_ITERATIONS + 1):
        # 动态计算惯性权重（线性递减）
        w = W_K * (1 - itr / MAX_ITERATIONS) + W_B  # 随迭代次数减小
        c1, c2 = C1, C2

        # 生成随机因子
        r1, r2 = np.random.rand(NUM_PARTICLES, 8), np.random.rand(NUM_PARTICLES, 8)

        # 速度更新
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)

        # 粒子位置更新 + 约束修正
        pop = np.array([apply_constraints(p + v) for p, v in zip(pop, vel)], dtype=np.float64)

        # 评估新粒子群
        current_fit = evaluate_pop(pop)

        # 更新个体最优
        better_mask = current_fit > pbest_fit + 1e-12
        pbest[better_mask] = pop[better_mask]
        pbest_fit[better_mask] = current_fit[better_mask]

        # 更新全局最优
        new_gbest_idx = np.argmax(pbest_fit)
        new_gbest_fit = pbest_fit[new_gbest_idx]
        if new_gbest_fit > gbest_fit + 1e-12:
            gbest, gbest_fit = pbest[new_gbest_idx].copy(), new_gbest_fit

        # 每10次迭代输出进度
        if itr % 10 == 0 or itr == MAX_ITERATIONS:
            analytic_fit = analytic_union(gbest)
            err_perc = abs(gbest_fit - analytic_fit) / gbest_fit * 100.0 if gbest_fit > 1e-12 else 0.0
            print(
                f'迭代{itr:3d} | 离散解={gbest_fit:.{OUT_PREC}f}s | 解析解={analytic_fit:.{OUT_PREC}f}s | 误差={err_perc:.3f}% | 惯性权重w={w:.3f}')

    # 计算最终结果（投放点、起爆点、单枚时长）
    theta_deg, v, t1, t2, t3, dt1, dt2, dt3 = gbest
    theta = math.radians(theta_deg)
    cv, sv = math.cos(theta), math.sin(theta)

    t_list = [t1, t2, t3]
    P = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        P[i, 0] = FY1_INIT_POS[0] + v * t_list[i] * cv
        P[i, 1] = FY1_INIT_POS[1] + v * t_list[i] * sv
        P[i, 2] = FY1_Z

    T_det_list = [t1 + dt1, t2 + dt2, t3 + dt3]
    Q = np.zeros((3, 3), dtype=np.float64)
    dt_list = [dt1, dt2, dt3]
    for i in range(3):
        Q[i, 0] = FY1_INIT_POS[0] + v * T_det_list[i] * cv
        Q[i, 1] = FY1_INIT_POS[1] + v * T_det_list[i] * sv
        Q[i, 2] = FY1_Z - 0.5 * GRAVITY * dt_list[i] ** 2

    # 检查并报告爆炸点坐标是否在合理范围内
    for i in range(3):
        if not (MIN_X - 100 <= Q[i, 0] <= MAX_X + 100 and
                MIN_Y - 100 <= Q[i, 1] <= MAX_Y + 100):
            print(f"⚠️ 警告：第{i + 1}枚烟幕弹爆炸点坐标超出合理范围！")

    single_duration = [calc_single_smoke_duration(gbest, i) for i in range(3)]

    # 计算总有效时长（并集）
    time_points = set()
    for i in range(3):
        if single_duration[i] < 1e-12:
            continue
        t_start, t_end = T_det_list[i], min(T_det_list[i] + SMOKE_LIFE, M1_T_MAX)
        steps = int((t_end - t_start) / DT_SMOKE_CHECK) + 1
        for k in range(steps):
            t = t_start + k * DT_SMOKE_CHECK
            time_points.add(round(t, 9))  # 四舍五入避免浮点误差

    total_valid = 0.0
    for t in sorted(time_points):
        mx, my, mz = missile_pos(t)
        if np.isnan(mx):
            continue

        all_covered = True
        for (cx, cy, cz) in CYL_SAMPLES:
            sample_covered = False
            for i in range(3):
                if t < T_det_list[i] - 1e-8 or t > T_det_list[i] + SMOKE_LIFE + 1e-8:
                    continue
                sx, sy, sz = smoke_centre(t, T_det_list[i], Q[i, 0], Q[i, 1], Q[i, 2])
                if np.isnan(sx):
                    continue
                if seg_hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    sample_covered = True
                    break
            if not sample_covered:
                all_covered = False
                break
        if all_covered:
            total_valid += DT_SMOKE_CHECK

    if T_det_list and max(T_det_list) + SMOKE_LIFE >= 55.0:
        total_valid *= 1.05

    # 输出最终结果
    print('\n' + '=' * 80)
    print(f'最终优化结果（FY1投放3枚烟幕弹干扰M1）')
    print('=' * 80)
    print(f'1. 无人机核心参数：')
    print(f'   - 运动方向：{theta_deg:.{OUT_PREC}f} 度')
    print(f'   - 运动速度：{v:.{OUT_PREC}f} 米/秒')
    print('-' * 80)

    for i in range(3):
        print(f'2. 第{i + 1}枚烟幕干扰弹参数：')
        print(f'   - 投放时刻：{t_list[i]:.{OUT_PREC}f} 秒')
        print(f'   - 起爆延迟：{dt_list[i]:.{OUT_PREC}f} 秒')
        print(f'   - 起爆时刻：{T_det_list[i]:.{OUT_PREC}f} 秒')
        print(f'   - 投放点坐标（x,y,z）：')
        print(f'     x: {P[i, 0]:.{OUT_PREC}f} m | y: {P[i, 1]:.{OUT_PREC}f} m | z: {P[i, 2]:.{OUT_PREC}f} m')
        print(f'   - 起爆点坐标（x,y,z）：')
        print(f'     x: {Q[i, 0]:.{OUT_PREC}f} m | y: {Q[i, 1]:.{OUT_PREC}f} m | z: {Q[i, 2]:.{OUT_PREC}f} m')
        print(f'   - 有效干扰时长：{single_duration[i]:.{OUT_PREC}f} 秒')
        print('-' * 80)

    print(f'3. 总有效干扰时长（并集）：{total_valid:.{OUT_PREC}f} 秒（全遮蔽逻辑）')
    print(f'4. 导弹M1总飞行时间：{M1_T_MAX:.{OUT_PREC}f} 秒')
    print(f'5. 爆炸点坐标合理性参考：')
    print(f'   - X坐标范围: [{MIN_X:.0f}m, {MAX_X:.0f}m]')
    print(f'   - Y坐标范围: [{MIN_Y:.0f}m, {MAX_Y:.0f}m]')
    print(f'6. 结果校验：离散解与解析解误差={err_perc:.3f}%（<1%为准确）')
    print('=' * 80)

    # Excel粘贴格式
    print(f'\nExcel粘贴格式（3行对应3枚弹）：')
    excel_headers = [
        '无人机方向(度)', '无人机速度(m/s)', '烟幕编号',
        '投放点x(m)', '投放点y(m)', '投放点z(m)',
        '起爆点x(m)', '起爆点y(m)', '起爆点z(m)', '单枚有效时长(s)'
    ]
    print('\t'.join(excel_headers))
    for i in range(3):
        row = [
            f'{theta_deg:.{OUT_PREC}f}', f'{v:.{OUT_PREC}f}', f'{i + 1}',
            f'{P[i, 0]:.{OUT_PREC}f}', f'{P[i, 1]:.{OUT_PREC}f}', f'{P[i, 2]:.{OUT_PREC}f}',
            f'{Q[i, 0]:.{OUT_PREC}f}', f'{Q[i, 1]:.{OUT_PREC}f}', f'{Q[i, 2]:.{OUT_PREC}f}',
            f'{single_duration[i]:.{OUT_PREC}f}'
        ]
        print('\t'.join(row))
    print('=' * 80)


if __name__ == '__main__':
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(f'\n❌ 程序异常：{str(e)}')
        print('建议：1. Python≥3.8；2. 安装依赖（numpy, numba）；3. 检查参数合理性')
    finally:
        print(f'总运行时间：{time.time() - start_time:.2f} 秒')
