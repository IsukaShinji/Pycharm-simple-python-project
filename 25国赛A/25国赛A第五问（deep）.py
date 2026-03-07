# -*- coding: utf-8 -*-
"""
2025 A 题 问题4 三机三弹协同干扰 M1（长时间优化版）
⚡ 专门优化以获得更长的有效干扰时间
"""
import numpy as np
import math, time, os
from random import uniform
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ========================  用户可调参数区（BEGIN）  ========================
# 1. 线程数：根据CPU核心数调整
N_THREAD = 32

# 2. 圆柱采样密度（保持适中以保证精度和速度平衡）
N_LAYER = 5  # 层数（对应真目标圆柱高度方向采样）
M_PER_LAYER = 30  # 每层点数（对应真目标圆柱圆周采样）

# 3. 时间步长（平衡计算精度与速度，A题物理时间尺度下0.001s过细，建议0.01s）
DT_CHECK = 0.01

# 4. 粗筛阶段 - 增加筛选强度
COARSE_CHUNK = 5000  # 每批粒子数
COARSE_ROUND = 40  # 批次数 → 总采样数 = COARSE_CHUNK * COARSE_ROUND
COARSE_MIN_FIT = 5.0  # 绝对阈值：仅保留有效时间≥5s的粒子（参考A题干扰需求）
USE_ABS_THRESHOLD = True  # 使用绝对阈值筛选
COARSE_KEEP_RATIO = 0.1  # 非绝对阈值时的保留比例（补充定义，避免未定义错误）

# 5. 精修阶段 - 增加迭代次数和粒子数
FINE_PARTICLE = 400  # 精修粒子数（PSO种群规模）
FINE_ITER = 1000  # 精修迭代数（PSO迭代次数）

# 6. 奖励系数（鼓励多无人机协同干扰，符合A题多机协同需求）
REWARD = 1.2  # 协同奖励权重

# 7. 中间过程打印频率
VERBOSE_STEP = 20  # 每N代打印一次迭代信息

# 8. 长时间优化专用参数（针对A题“有效遮蔽时间尽可能长”的目标）
LONG_TIME_FOCUS = True  # 开启长时间优化模式
TIME_WEIGHT = 2.0  # 时间权重因子（放大长时间解的优势）
# ========================  用户可调参数区（END）  ========================

# --------------------  固定物理量（严格匹配A题.pdf与第五问模型文档）  --------------------
# 真目标参数（A题.pdf：半径7m、高10m圆柱，下底面圆心(0,200,0)）
CYL_C = np.array([0., 200., 0.], dtype=np.float64)
CYL_R, CYL_H = 7.0, 10.0

# M1导弹参数（A题.pdf：初始位置(20000,0,2000)，速度300m/s指向原点）
M1_POS0 = np.array([20000., 0., 2000.], dtype=np.float64)
M1_VEL = 300. * (-M1_POS0) / np.linalg.norm(M1_POS0)  # 速度矢量（指向原点）
M1_T_MAX = np.linalg.norm(M1_POS0) / 300.  # 导弹最大飞行时间（到原点）

# 3架无人机初始位置（A题.pdf：FY1、FY2、FY3）
UAV_POS = {
    1: np.array([17800., 0., 1800.], dtype=np.float64),  # FY1
    2: np.array([12000., 1400., 1400.], dtype=np.float64), # FY2
    3: np.array([6000., -3000., 700.], dtype=np.float64)   # FY3
}

# 无人机约束（第五问模型文档：速度70-140m/s，方向0-360°）
V_BOUNDS = (70., 140.)
THETA_BOUNDS = (0., 360.)

# 烟幕参数（A题.pdf：半径10m、有效时间20s、下沉速度3m/s；重力加速度9.8m/s²）
SMOKE_R, SMOKE_LIFE, SMOKE_SINK, G = 10.0, 20.0, 3.0, 9.8

# --------------------  Numba 加速区（第五问模型文档推荐的计算优化手段）  --------------------
try:
    from numba import njit
    HAS_NUMBA = True
    logging.info("✅ 启用Numba加速（符合第五问模型文档计算优化建议）")
except ImportError:
    HAS_NUMBA = False
    # 定义伪njit装饰器（无Numba时兼容运行，不影响语法正确性）
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True, cache=True)
def _gen_samples(c_center, r, h, n_layer, m_layer):
    """生成真目标圆柱采样点（第五问模型文档“真目标建模”要求）"""
    pts = []
    # 高度方向采样（从下底面到上底面）
    zs = np.linspace(c_center[2], c_center[2] + h, n_layer)
    for z in zs:
        # 圆周方向采样
        for k in range(m_layer):
            ang = 2. * math.pi * k / m_layer
            pts.append([
                c_center[0] + r * math.cos(ang),
                c_center[1] + r * math.sin(ang),
                z
            ])
    return np.array(pts, dtype=np.float64)


# 生成真目标采样点（全局变量，避免重复计算）
CYL_PTS = _gen_samples(CYL_C, CYL_R, CYL_H, N_LAYER, M_PER_LAYER)
N_CYL = len(CYL_PTS)

# 无人机初始位置数组（便于批量计算）
UAV_ARR = np.empty((3, 3), dtype=np.float64)
for i in range(3):
    UAV_ARR[i] = UAV_POS[i + 1]


@njit(fastmath=True, cache=True)
def _missile(t):
    """计算M1导弹在t时刻的位置（A题.pdf：匀速直线运动）"""
    if t >= M1_T_MAX - 1e-8:  # 导弹到达目标后失效
        return np.nan, np.nan, np.nan
    x = M1_POS0[0] + M1_VEL[0] * t
    y = M1_POS0[1] + M1_VEL[1] * t
    z = M1_POS0[2] + M1_VEL[2] * t
    return x, y, z


@njit(fastmath=True, cache=True)
def _smoke_c(t, t_det, xe, ye, ze):
    """计算烟幕云团在t时刻的中心位置（A题.pdf：匀速下沉，无水平运动）"""
    # 烟幕仅在起爆后20s内有效
    if t < t_det - 1e-8 or t > t_det + SMOKE_LIFE + 1e-8:
        return np.nan, np.nan, np.nan
    # 烟幕中心x/y坐标与起爆点一致，z坐标随时间下沉
    return xe, ye, ze - SMOKE_SINK * (t - t_det)


@njit(fastmath=True, cache=True)
def _hit(ax, ay, az, bx, by, bz, ox, oy, oz):
    """判断线段（ax,ay,az）-（bx,by,bz）是否与球体（球心(ox,oy,oz)，半径SMOKE_R）相交（第五问模型文档“有效遮蔽约束”）"""
    # 向量计算
    abx, aby, abz = bx - ax, by - ay, bz - az
    aox, aoy, aoz = ax - ox, ay - oy, az - oz
    seg_len_sq = abx**2 + aby**2 + abz**2

    # 线段长度为0（点）的情况
    if seg_len_sq < 1e-16:
        return (aox**2 + aoy**2 + aoz**2) <= SMOKE_R**2 + 1e-8

    # 求解二次方程（判断线段与球的交点）
    a = seg_len_sq
    b = 2 * (aox * abx + aoy * aby + aoz * abz)
    c = (aox**2 + aoy**2 + aoz**2) - SMOKE_R**2
    delta = b**2 - 4 * a * c

    # 无实根（无交点）
    if delta < -1e-8:
        return False
    # 计算根并判断是否在线段上
    sqrt_delta = math.sqrt(max(delta, 0.))
    s1, s2 = (-b - sqrt_delta) / (2 * a), (-b + sqrt_delta) / (2 * a)
    return s1 <= 1. + 1e-8 and s2 >= 0. - 1e-8


@njit(fastmath=True, cache=True)
def _fitness_one(x12, cyl_pts, n_cyl, dt_check, m1_t_max, smoke_life, smoke_sink, g,
                 uav_pos_, v_bds, th_bds, reward, time_weight, long_time_focus):
    """计算单个粒子的适应度（第五问模型文档“目标函数”：总有效干扰时长+协同奖励）"""
    # 解析粒子参数：每架无人机4个参数（方向、速度、投放时刻、起爆延迟），3架共12个参数
    theta = x12[::4]    # 3架无人机的飞行方向（°）
    v = x12[1::4]        # 3架无人机的飞行速度（m/s）
    t_rel = x12[2::4]    # 3架无人机的烟幕投放时刻（s）
    dt = x12[3::4]       # 3架无人机的烟幕起爆延迟（s）
    t_det = t_rel + dt  # 3架无人机的烟幕起爆时刻（s）

    # 1. 物理约束检查（第五问模型文档“约束条件”）
    for i in range(3):
        # 速度在有效范围
        if v[i] < v_bds[0] or v[i] > v_bds[1]:
            return 0.
        # 方向在有效范围
        if theta[i] < th_bds[0] or theta[i] > th_bds[1]:
            return 0.
        # 投放时刻非负，起爆延迟非负，起爆时刻不超过导弹最大飞行时间
        if t_rel[i] < 0. or dt[i] < 0. or t_det[i] > m1_t_max - 1e-8:
            return 0.

    # 2. 计算烟幕起爆点（第五问模型文档“起爆点公式”：无人机等高度飞行）
    xyz_det = np.empty((3, 3), dtype=np.float64)  # 3架无人机的起爆点(x,y,z)
    for i in range(3):
        th_rad = math.radians(theta[i])  # 角度转弧度
        # 起爆点x/y：无人机从初始位置匀速飞行到起爆时刻的位置
        xe = uav_pos_[i, 0] + v[i] * t_det[i] * math.cos(th_rad)
        ye = uav_pos_[i, 1] + v[i] * t_det[i] * math.sin(th_rad)
        # 起爆点z：无人机等高度飞行，减去烟幕下落距离（第五问模型文档假设）
        ze = uav_pos_[i, 2] - 0.5 * g * dt[i]**2
        # 烟幕起爆点z坐标非负（地面约束）
        if ze < -1e-8:
            return 0.
        xyz_det[i] = [xe, ye, ze]

    # 3. 确定有效时间窗口（烟幕有效时间与导弹飞行时间的交集）
    t_start = max(t_det.min(), 0.)  # 有效时间起始（取最晚起爆时刻与0的最大值）
    t_end = min(t_det.max() + smoke_life, m1_t_max)  # 有效时间结束（取最早烟幕失效时刻与导弹最大飞行时间的最小值）
    if t_start >= t_end - 1e-8:  # 无有效时间窗口
        return 0.

    # 4. 计算总有效干扰时长与单无人机贡献（第五问模型文档“有效遮蔽判断”）
    steps = int((t_end - t_start) / dt_check) + 1
    total_eff = 0.  # 总有效干扰时长
    single_eff = np.zeros(3, dtype=np.float64)  # 每架无人机的单独有效时长

    for k in range(steps):
        t = t_start + k * dt_check
        # 计算当前时刻导弹位置
        mx, my, mz = _missile(t)
        if np.isnan(mx):  # 导弹已到达目标，无需干扰
            continue

        # 检查真目标所有采样点是否被至少一个烟幕遮蔽
        all_blocked = True
        for j in range(n_cyl):
            cx, cy, cz = cyl_pts[j]  # 真目标第j个采样点
            blocked = False
            # 遍历3架无人机的烟幕
            for i in range(3):
                # 跳过当前时刻无效的烟幕
                if t < t_det[i] - 1e-8 or t > t_det[i] + smoke_life + 1e-8:
                    continue
                # 计算当前时刻烟幕中心位置
                sx, sy, sz = _smoke_c(t, t_det[i], xyz_det[i, 0], xyz_det[i, 1], xyz_det[i, 2])
                if np.isnan(sx):
                    continue
                # 判断导弹到目标采样点的视线是否被烟幕遮蔽
                if _hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    blocked = True
                    break
            if not blocked:  # 存在未被遮蔽的采样点，当前时刻无效
                all_blocked = False
                break

        # 累加有效时长
        if all_blocked:
            total_eff += dt_check
            # 记录每架无人机的贡献（当前时刻生效的烟幕）
            for i in range(3):
                if t_det[i] <= t <= t_det[i] + smoke_life:
                    single_eff[i] += dt_check

    # 5. 适应度计算（长时间优化+协同奖励，符合A题“有效时间尽可能长”目标）
    if long_time_focus:
        # 基础适应度（总有效时长）
        base_fit = total_eff
        # 长时间奖励（指数放大，鼓励更长有效时长）
        time_bonus = math.exp(time_weight * (total_eff / m1_t_max))
        # 协同奖励（鼓励多无人机参与干扰）
        coop_bonus = 0.
        for i in range(3):
            if single_eff[i] > 0.1:  # 至少贡献0.1s的无人机才计入协同
                coop_bonus += reward * math.sqrt(single_eff[i])
        return base_fit * time_bonus + coop_bonus
    else:
        # 标准模式（总有效时长+协同奖励）
        coop_reward = sum(reward * math.sqrt(eff + 1e-6) for eff in single_eff)
        return total_eff + coop_reward


def eval_pool(pop):
    """批量评估种群适应度（多线程加速，符合第五问模型文档“并行计算”建议）"""
    fit = np.zeros(len(pop), dtype=np.float64)
    # 多线程评估，避免单线程计算过慢
    with ThreadPoolExecutor(max_workers=N_THREAD) as executor:
        # 提交任务：每个粒子对应一个评估任务
        future_dict = {
            executor.submit(
                _fitness_one,
                pop[i], CYL_PTS, N_CYL, DT_CHECK,
                M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR,
                V_BOUNDS, THETA_BOUNDS, REWARD, TIME_WEIGHT, LONG_TIME_FOCUS
            ): i for i in range(len(pop))
        }
        # 收集结果
        for future in as_completed(future_dict):
            idx = future_dict[future]
            fit[idx] = future.result()
    return fit


def init_pop(size):
    """初始化种群（符合第五问模型文档“混合变量初始化”逻辑）"""
    pop = np.zeros((size, 12), dtype=np.float64)  # 3架无人机×4个参数=12维
    for i in range(size):
        for j in range(3):  # 遍历3架无人机
            # 1. 飞行方向（0-360°随机）
            pop[i, j * 4] = uniform(*THETA_BOUNDS)
            # 2. 飞行速度（70-140m/s随机）
            pop[i, j * 4 + 1] = uniform(*V_BOUNDS)
            # 3. 投放时刻（0到“导弹最大飞行时间-2s”随机，预留起爆延迟时间）
            max_t_rel = M1_T_MAX - 2.0
            pop[i, j * 4 + 2] = uniform(0., max_t_rel)
            # 4. 起爆延迟（0到“烟幕寿命/自由下落时间/导弹剩余时间”的最小值）
            max_dt = min(
                SMOKE_LIFE,  # 烟幕最大有效时间
                math.sqrt(2 * UAV_ARR[j, 2] / G),  # 烟幕从无人机高度自由下落时间
                M1_T_MAX - pop[i, j * 4 + 2]  # 导弹剩余飞行时间
            )
            pop[i, j * 4 + 3] = uniform(0., max_dt)
    return pop


def small_pso(best4):
    """PSO精修（第五问模型文档推荐的“局部精修算法”）"""
    # 初始化种群（确保种群规模达到FINE_PARTICLE）
    pop = np.array(best4, dtype=np.float64)
    while len(pop) < FINE_PARTICLE:
        pop = np.vstack([pop, init_pop(1)])  # 不足时补充随机粒子
    pop = pop[:FINE_PARTICLE]  # 截断到目标种群规模

    # PSO初始化
    vel = np.zeros_like(pop)  # 速度初始化
    pbest = pop.copy()       # 个体最优初始化
    pbest_fit = eval_pool(pop)  # 个体最优适应度
    # 全局最优初始化（适应度最大的个体）
    gbest_idx = np.argmax(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # 自适应PSO参数（第五问模型文档“自适应策略”）
    w_start, w_end = 0.9, 0.4  # 惯性权重：从探索到收敛
    c1_start, c1_end = 2.0, 0.5  # 认知系数：从个体探索到全局收敛
    c2_start, c2_end = 2.0, 2.5  # 社会系数：从全局引导到强化收敛

    for itr in range(1, FINE_ITER + 1):
        # 1. 自适应更新PSO参数
        progress = itr / FINE_ITER  # 迭代进度（0→1）
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress

        # 2. 更新速度（PSO核心公式）
        r1, r2 = np.random.rand(*pop.shape), np.random.rand(*pop.shape)  # 0-1随机数
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)

        # 3. 速度约束（避免参数更新幅度过大）
        vel_max = np.array([
            THETA_BOUNDS[1] - THETA_BOUNDS[0],  # 方向最大变化（360°）
            V_BOUNDS[1] - V_BOUNDS[0],          # 速度最大变化（70m/s）
            M1_T_MAX * 0.1,                    # 投放时刻最大变化（10%导弹飞行时间）
            SMOKE_LIFE * 0.1                   # 起爆延迟最大变化（10%烟幕寿命）
        ] * 3)  # 3架无人机，重复3次
        # 逐参数限制速度
        for i in range(len(vel)):
            for j in range(12):
                if abs(vel[i, j]) > vel_max[j % 4]:
                    vel[i, j] = np.sign(vel[i, j]) * vel_max[j % 4]

        # 4. 更新位置（新粒子）
        new_pop = pop + vel
        # 位置约束（确保参数符合物理意义）
        for j in range(3):  # 遍历3架无人机
            new_pop[:, j * 4] = np.clip(new_pop[:, j * 4], *THETA_BOUNDS)    # 方向约束
            new_pop[:, j * 4 + 1] = np.clip(new_pop[:, j * 4 + 1], *V_BOUNDS)  # 速度约束
            new_pop[:, j * 4 + 2] = np.clip(new_pop[:, j * 4 + 2], 0., M1_T_MAX)  # 投放时刻约束
            new_pop[:, j * 4 + 3] = np.clip(new_pop[:, j * 4 + 3], 0., SMOKE_LIFE)  # 起爆延迟约束

        # 5. 评估新种群适应度
        new_fit = eval_pool(new_pop)

        # 6. 更新个体最优（更优则替换）
        better_mask = new_fit > pbest_fit
        pbest[better_mask] = new_pop[better_mask]
        pbest_fit[better_mask] = new_fit[better_mask]

        # 7. 更新全局最优（更优则替换）
        current_max_fit = new_fit.max()
        if current_max_fit > gbest_fit + 1e-12:  # 避免浮点误差导致的无效更新
            current_max_idx = np.argmax(new_fit)
            gbest = new_pop[current_max_idx].copy()
            gbest_fit = current_max_fit

        # 8. 打印迭代信息（按VERBOSE_STEP频率）
        if itr % VERBOSE_STEP == 0 or itr == FINE_ITER:
            # 计算全局最优解的真实有效时间（解析解验证，第五问模型文档“验证机制”）
            real_eff_time = analytic_check(gbest, CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
            print(f'[PSO] 迭代 {itr:3d}/{FINE_ITER} | 适应度={gbest_fit:.6f} | 真实有效时间={real_eff_time:.6f}s')

    return gbest, gbest_fit


@njit(fastmath=True, cache=True)
def analytic_check(x12, cyl_pts, n_cyl, dt_check, m1_t_max, smoke_life, smoke_sink, g, uav_pos_):
    """解析解验证（第五问模型文档“双重验证机制”，确保物理真实性）"""
    # 解析粒子参数
    theta = x12[::4]
    v = x12[1::4]
    t_rel = x12[2::4]
    dt = x12[3::4]
    t_det = t_rel + dt

    # 计算烟幕起爆点
    xyz_det = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        th_rad = math.radians(theta[i])
        xe = uav_pos_[i, 0] + v[i] * t_det[i] * math.cos(th_rad)
        ye = uav_pos_[i, 1] + v[i] * t_det[i] * math.sin(th_rad)
        ze = uav_pos_[i, 2] - 0.5 * g * dt[i]**2
        xyz_det[i] = [xe, ye, ze]

    # 确定有效时间窗口
    t_start = max(t_det.min(), 0.)
    t_end = min(t_det.max() + smoke_life, m1_t_max)
    if t_start >= t_end - 1e-8:
        return 0.0

    # 计算真实有效时间（仅统计物理有效时长，无奖励）
    real_eff_time = 0.0
    for t in np.arange(t_start, t_end, dt_check):
        mx, my, mz = _missile(t)
        if np.isnan(mx):
            continue

        # 检查真目标是否被完全遮蔽
        all_blocked = True
        for j in range(n_cyl):
            cx, cy, cz = cyl_pts[j]
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
            real_eff_time += dt_check

    return real_eff_time


def self_correction(best_x):
    """自修正函数（针对A题“长时间优化”目标，微调参数以延长有效时间）"""
    corrected = best_x.copy()
    # 解析当前参数
    theta = corrected[::4]
    v = corrected[1::4]
    t_rel = corrected[2::4]
    dt = corrected[3::4]
    t_det = t_rel + dt

    # 计算当前真实有效时间（作为修正基准）
    current_eff = analytic_check(corrected, CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
    best_eff = current_eff
    best_corr = corrected.copy()

    # 定义修正策略（基于A题物理规律：延长烟幕覆盖、优化无人机轨迹）
    correction_strategies = [
        # 策略1：延长起爆延迟（增加烟幕下落时间，扩大覆盖范围）
        lambda param, idx, val: min(val * 1.05, SMOKE_LIFE) if param == 3 else val,
        # 策略2：优化飞行速度（微调速度以调整烟幕投放位置）
        lambda param, idx, val: np.clip(val * 1.02, V_BOUNDS[0], V_BOUNDS[1]) if param == 1 else val,
        # 策略3：调整投放时刻（提前投放以延长覆盖时间）
        lambda param, idx, val: max(val * 0.98, 0.) if param == 2 else val,
        # 策略4：调整飞行方向（向导弹轨迹方向微调，优化遮蔽位置）
        lambda param, idx, val: (val + 2.0) % 360.0 if param == 0 else val
    ]

    # 遍历每架无人机的每个参数，尝试所有策略
    for uav_idx in range(3):
        for param_idx in range(4):  # 0=方向,1=速度,2=投放时刻,3=起爆延迟
            for strategy in correction_strategies:
                # 生成候选修正解
                candidate = corrected.copy()
                param_pos = uav_idx * 4 + param_idx
                candidate[param_pos] = strategy(param_idx, uav_idx, candidate[param_pos])
                # 验证候选解的物理约束
                if param_idx == 1:  # 速度约束
                    candidate[param_pos] = np.clip(candidate[param_pos], V_BOUNDS[0], V_BOUNDS[1])
                elif param_idx == 2:  # 投放时刻约束
                    candidate[param_pos] = np.clip(candidate[param_pos], 0., M1_T_MAX)
                elif param_idx == 3:  # 起爆延迟约束
                    candidate[param_pos] = np.clip(candidate[param_pos], 0., SMOKE_LIFE)
                # 计算候选解的真实有效时间
                candidate_eff = analytic_check(candidate, CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
                # 保留更优解
                if candidate_eff > best_eff:
                    best_eff = candidate_eff
                    best_corr = candidate.copy()

    return best_corr


def main():
    print('==== 2025 A题 问题4 三机三弹协同干扰 M1（长时间优化版）====')
    print('核心参数来源：A题.pdf | 第五问模型建立与求解内容.docx')
    print('可调参数：N_THREAD / COARSE_* / FINE_* / REWARD / TIME_WEIGHT\n')

    # Step1：粗筛随机种子（高阈值筛选，第五问模型文档“粗筛阶段”）
    print('Step1 粗筛随机种子（高阈值筛选，保留有效时间≥5s的粒子）...')
    keep_particles = []  # 保存粗筛后的有效粒子
    for round_idx in range(1, COARSE_ROUND + 1):
        # 生成一批随机粒子
        pop = init_pop(COARSE_CHUNK)
        # 评估这批粒子的适应度
        fit = eval_pool(pop)

        # 根据筛选策略保留有效粒子
        if USE_ABS_THRESHOLD:
            # 绝对阈值：保留适应度≥COARSE_MIN_FIT的粒子
            valid_mask = fit >= COARSE_MIN_FIT
            keep_particles.append(pop[valid_mask])
            # 计算当前批次最优粒子的真实有效时间
            best_idx = np.argmax(fit)
            best_eff = analytic_check(pop[best_idx], CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
            print(f'  粗筛轮次 {round_idx:2d}/{COARSE_ROUND} | 保留粒子数：{valid_mask.sum()} | 本批最优有效时间：{best_eff:.6f}s')
        else:
            # 比例阈值：保留适应度前COARSE_KEEP_RATIO的粒子
            keep_count = int(COARSE_CHUNK * COARSE_KEEP_RATIO)
            best_idx = np.argpartition(fit, -keep_count)[-keep_count:]
            keep_particles.append(pop[best_idx])
            # 计算当前批次最优粒子的真实有效时间
            best_eff = analytic_check(pop[np.argmax(fit)], CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
            print(f'  粗筛轮次 {round_idx:2d}/{COARSE_ROUND} | 保留粒子数：{keep_count} | 本批最优有效时间：{best_eff:.6f}s')

    # 处理粗筛结果（若无有效粒子，补充随机粒子）
    if keep_particles:
        keep_particles = np.vstack(keep_particles)
    else:
        keep_particles = init_pop(COARSE_CHUNK)  # 兜底：无有效粒子时生成一批随机粒子
    print(f'Step1 粗筛完成 | 总保留有效粒子数：{len(keep_particles)}\n')

    # Step2：PSO精修（第五问模型文档“精修阶段”）
    print('Step2 PSO精修（优化长时间有效干扰策略）...')
    best_x, best_fit = small_pso(keep_particles)
    print(f'Step2 PSO精修完成 | 精修后最优适应度：{best_fit:.6f}\n')

    # Step3：自修正（进一步延长有效时间，符合A题“尽可能长”目标）
    print('Step3 自修正（微调参数以延长有效时间）...')
    for corr_iter in range(3):  # 多次修正迭代
        best_x = self_correction(best_x)
        current_eff = analytic_check(best_x, CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
        print(f'  自修正迭代 {corr_iter + 1}/3 | 当前有效时间：{current_eff:.6f}s')
    print(f'Step3 自修正完成\n')

    # Step4：输出最终最优策略（符合A题“输出投放策略”要求）
    print('==== 最终最优干扰策略（基于A题.pdf与第五问模型文档） ====')
    # 解析最终最优参数
    theta = best_x[::4]    # 3架无人机的飞行方向（°）
    v = best_x[1::4]        # 3架无人机的飞行速度（m/s）
    t_rel = best_x[2::4]    # 3架无人机的烟幕投放时刻（s）
    dt = best_x[3::4]       # 3架无人机的烟幕起爆延迟（s）
    t_det = t_rel + dt      # 3架无人机的烟幕起爆时刻（s）

    # 输出表头
    header = [
        '无人机编号', '飞行方向(°)', '飞行速度(m/s)',
        '投放点X(m)', '投放点Y(m)', '投放点Z(m)',
        '起爆点X(m)', '起爆点Y(m)', '起爆点Z(m)',
        '单独有效时长(s)'
    ]
    print('\t'.join(header))

    # 计算总真实有效时间与单无人机贡献
    total_eff = 0.0
    for uav_idx in range(3):
        uav_name = f'FY{uav_idx + 1}'  # 无人机编号：FY1、FY2、FY3
        th_rad = math.radians(theta[uav_idx])

        # 计算投放点（无人机投放烟幕时的位置，第五问模型文档“投放点公式”）
        drop_x = UAV_ARR[uav_idx, 0] + v[uav_idx] * t_rel[uav_idx] * math.cos(th_rad)
        drop_y = UAV_ARR[uav_idx, 1] + v[uav_idx] * t_rel[uav_idx] * math.sin(th_rad)
        drop_z = UAV_ARR[uav_idx, 2]  # 等高度飞行，投放点Z=初始Z

        # 计算起爆点（第五问模型文档“起爆点公式”）
        det_x = UAV_ARR[uav_idx, 0] + v[uav_idx] * t_det[uav_idx] * math.cos(th_rad)
        det_y = UAV_ARR[uav_idx, 1] + v[uav_idx] * t_det[uav_idx] * math.sin(th_rad)
        det_z = UAV_ARR[uav_idx, 2] - 0.5 * G * dt[uav_idx]**2  # 减去烟幕下落距离

        # 计算该无人机的单独有效时长
        single_eff = 0.0
        t_start = t_det[uav_idx]
        t_end = min(t_det[uav_idx] + SMOKE_LIFE, M1_T_MAX)
        steps = int((t_end - t_start) / DT_CHECK) + 1
        for k in range(steps):
            t = t_start + k * DT_CHECK
            mx, my, mz = _missile(t)
            if np.isnan(mx):
                continue
            # 计算当前时刻烟幕中心位置
            sx, sy, sz = _smoke_c(t, t_det[uav_idx], det_x, det_y, det_z)
            if np.isnan(sx):
                continue
            # 检查是否至少遮蔽一个真目标采样点
            for j in range(N_CYL):
                cx, cy, cz = CYL_PTS[j]
                if _hit(mx, my, mz, cx, cy, cz, sx, sy, sz):
                    single_eff += DT_CHECK
                    break
        total_eff += single_eff

        # 输出该无人机的详细参数
        row = [
            uav_name,
            f'{theta[uav_idx]:.3f}',
            f'{v[uav_idx]:.3f}',
            f'{drop_x:.3f}',
            f'{drop_y:.3f}',
            f'{drop_z:.3f}',
            f'{det_x:.3f}',
            f'{det_y:.3f}',
            f'{det_z:.3f}',
            f'{single_eff:.6f}'
        ]
        print('\t'.join(row))

    # 最终解析解验证（确保结果物理真实）
    final_eff = analytic_check(best_x, CYL_PTS, N_CYL, DT_CHECK, M1_T_MAX, SMOKE_LIFE, SMOKE_SINK, G, UAV_ARR)
    print(f'\n==== 最终结果验证（符合第五问模型文档“双重验证”） ====')
    print(f'总真实有效干扰时间：{final_eff:.6f}s')
    # 参考A题合理干扰时长（基于导弹飞行时间约66-67s，有效时间≥5s为合理）
    if final_eff >= 5.0:
        print('✅ 结果符合A题干扰需求（有效时间≥5s）')
        if final_eff >= 8.0:
            print('🎉 结果优秀（有效时间≥8s，长时间干扰目标达成）')
    else:
        print('❌ 结果未达A题干扰需求（有效时间<5s，建议调整粗筛参数或增加迭代次数）')


if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f'\n==== 程序总耗时：{total_time:.2f}s ====')