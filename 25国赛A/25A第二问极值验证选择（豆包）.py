import numpy as np
import math
from typing import Tuple

# ==============================================================================
# 固定参数（与原优化代码完全一致，确保验证逻辑统一）
# ==============================================================================
# 真目标参数
TRUE_TARGET_CYL = {
    "center_bottom": np.array([0.0, 200.0, 0.0]),
    "radius": 7.0,
    "height": 10.0
}

# 无人机参数
UAV_FY1_INIT = np.array([17800.0, 0.0, 1800.0])
UAV_V_BOUNDS = (70.0, 140.0)
UAV_THETA_BOUNDS = (0.0, 360.0)
UAV_Z_FIXED = UAV_FY1_INIT[2]

# 导弹参数
M1_INIT = np.array([20000.0, 0.0, 2000.0])
M1_SPEED = 300.0
M1_DIR = (-M1_INIT) / np.linalg.norm(M1_INIT)
M1_V = M1_SPEED * M1_DIR
M1_TOTAL_TIME = np.linalg.norm(M1_INIT) / M1_SPEED
M1_VX, M1_VY, M1_VZ = M1_V
M1_INIT_X, M1_INIT_Y, M1_INIT_Z = M1_INIT

# 烟幕参数
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_VALID_TIME = 20.0
g = 9.8

# 验证精度参数（可适当提高精度，如0.0001s）
dt_validation = 0.0001  # 比优化时的步长更小，验证更严格


# ==============================================================================
# 核心验证函数（与原代码逻辑一致，确保结果可比）
# ==============================================================================
def generate_cyl_samples(num_z_layers: int = 3, num_points_per_layer: int = 72) -> np.ndarray:
    """生成真目标采样点（与原代码相同）"""
    cyl_points = []
    ccx, ccy, ccz = TRUE_TARGET_CYL["center_bottom"]
    r = TRUE_TARGET_CYL["radius"]
    height = TRUE_TARGET_CYL["height"]
    z_layers = np.linspace(ccz, ccz + height, num_z_layers)
    for z in z_layers:
        thetas = np.linspace(0, 2 * np.pi, num_points_per_layer, endpoint=False)
        for theta in thetas:
            x = ccx + r * np.cos(theta)
            y = ccy + r * np.sin(theta)
            cyl_points.append((x, y, z))
    return np.array(cyl_points, dtype=np.float64)


def calc_missile_pos(t: float) -> Tuple[float, float, float]:
    """计算t时刻导弹位置"""
    if t >= M1_TOTAL_TIME + 1e-6:
        return np.nan, np.nan, np.nan
    x = M1_INIT_X + M1_VX * t
    y = M1_INIT_Y + M1_VY * t
    z = M1_INIT_Z + M1_VZ * t
    return x, y, z


def calc_smoke_center(t: float, t_det: float, det_x: float, det_y: float, det_z: float) -> Tuple[float, float, float]:
    """计算t时刻烟幕中心位置"""
    if t < t_det - 1e-6 or t > t_det + SMOKE_VALID_TIME + 1e-6:
        return np.nan, np.nan, np.nan
    z = det_z - SMOKE_SINK_SPEED * (t - t_det)
    return det_x, det_y, z


def segment_intersect(Ax, Ay, Az, Bx, By, Bz, Ox, Oy, Oz, r: float) -> bool:
    """线段（导弹-真目标点）与烟幕球相交判定"""
    ABx = Bx - Ax
    ABy = By - Ay
    ABz = Bz - Az
    AOx = Ax - Ox
    AOy = Ay - Oy
    AOz = Az - Oz

    a = ABx **2 + ABy** 2 + ABz **2
    if a < 1e-12:
        return (AOx** 2 + AOy **2 + AOz** 2) <= r **2 + 1e-6

    b = 2 * (AOx * ABx + AOy * ABy + AOz * ABz)
    c = (AOx** 2 + AOy **2 + AOz** 2) - r **2
    delta = b** 2 - 4 * a * c
    if delta < -1e-12:
        return False

    sqrt_delta = np.sqrt(max(delta, 0.0))
    s1 = (-b - sqrt_delta) / (2 * a)
    s2 = (-b + sqrt_delta) / (2 * a)
    return (s1 <= 1.0 + 1e-6) and (s2 >= 0.0 - 1e-6)


def validate_strategy(
    theta: float, v: float,
    drop_x: float, drop_y: float, drop_z: float,
    det_x: float, det_y: float, det_z: float,
    num_z_layers: int = 3, num_points_per_layer: int = 72
) -> float:
    """
    验证给定策略的有效遮蔽时间
    :param theta: 无人机方向（度）
    :param v: 无人机速度（m/s）
    :param drop_x/y/z: 投放点坐标
    :param det_x/y/z: 起爆点坐标
    :return: 有效遮蔽时间（s）
    """
    # 1. 约束检查
    if not (UAV_THETA_BOUNDS[0] <= theta <= UAV_THETA_BOUNDS[1] and
            UAV_V_BOUNDS[0] <= v <= UAV_V_BOUNDS[1]):
        return 0.0  # 参数超出约束范围

    # 2. 计算投放延迟t1和起爆延迟dt（反推时间参数，用于验证时间窗口）
    theta_rad = math.radians(theta)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    # 从投放点反推t1：drop = init + v*t1*(cosθ, sinθ, 0)
    delta_x_drop = drop_x - UAV_FY1_INIT[0]
    delta_y_drop = drop_y - UAV_FY1_INIT[1]
    t1 = (delta_x_drop * cos_theta + delta_y_drop * sin_theta) / (v + 1e-12)  # 投影计算t1
    if t1 < 0 or drop_z != UAV_Z_FIXED:
        return 0.0  # 投放点无效

    # 从起爆点反推dt：det = drop + v*dt*(cosθ, sinθ, 0)，且det_z = drop_z - 0.5*g*dt²
    delta_x_det = det_x - drop_x
    delta_y_det = det_y - drop_y
    dt = (delta_x_det * cos_theta + delta_y_det * sin_theta) / (v + 1e-12)  # 投影计算dt
    if dt < 0 or not np.isclose(det_z, drop_z - 0.5 * g * dt**2, atol=1e-3):
        return 0.0  # 起爆点无效

    # 3. 时间窗口计算
    t_det = t1 + dt
    if t_det > M1_TOTAL_TIME - 1e-6 or t_det + SMOKE_VALID_TIME > M1_TOTAL_TIME + 1e-6:
        return 0.0  # 烟幕有效时导弹已到达

    t_start = t_det
    t_end = min(t_det + SMOKE_VALID_TIME, M1_TOTAL_TIME)
    if t_start >= t_end - 1e-6:
        return 0.0

    # 4. 真目标采样点
    cyl_samples = generate_cyl_samples(num_z_layers, num_points_per_layer)

    # 5. 逐时间步验证遮蔽效果
    num_steps = int((t_end - t_start) / dt_validation) + 1
    effective_time = 0.0

    for i in range(num_steps):
        t = t_start + i * dt_validation
        # 导弹位置
        mx, my, mz = calc_missile_pos(t)
        if np.isnan(mx):
            continue
        # 烟幕位置
        sx, sy, sz = calc_smoke_center(t, t_det, det_x, det_y, det_z)
        if np.isnan(sx):
            continue

        # 检查所有采样点是否被遮蔽
        all_blocked = True
        for (cx, cy, cz) in cyl_samples:
            if not segment_intersect(mx, my, mz, cx, cy, cz, sx, sy, sz, SMOKE_RADIUS):
                all_blocked = False
                break
        if all_blocked:
            effective_time += dt_validation

    return effective_time


# ==============================================================================
# 主函数：验证示例参数
# ==============================================================================
def main():
    # 示例1：有效时间3.06s的参数（替换为你的优化结果）
    strategy1 = {
        "theta": 360.0000,
        "v": 70.0000,
        "drop_x": 17944.26,
        "drop_y": -0.00,
        "drop_z": 1800.00,
        "det_x": 17944.26,
        "det_y": -0.00,
        "det_z": 1800.00
    }

    # 示例2：有效时间4.58s的参数（替换为你的优化结果）
    strategy2 = {
        "theta": 180.0000,  # 示例值，需替换为实际优化结果
        "v": 100.0000,
        "drop_x": 17600.00,
        "drop_y": 0.00,
        "drop_z": 1800.00,
        "det_x": 17880.00,
        "det_y": 12.00,
        "det_z": 1795.00
    }

    # 验证两个策略
    print("=" * 70)
    print("烟幕干扰策略独立验证结果")
    print("=" * 70)

    # 验证策略1
    t1 = validate_strategy(**strategy1)
    print(f"策略1（优化结果3.06s）验证：")
    print(f"无人机运动方向(度): {strategy1['theta']:.4f}")
    print(f"无人机运动速度(m/s): {strategy1['v']:.4f}")
    print(f"烟幕干扰弹投放点的x坐标(m): {strategy1['drop_x']:.2f}")
    print(f"烟幕干扰弹投放点的y坐标(m): {strategy1['drop_y']:.2f}")
    print(f"烟幕干扰弹投放点的z坐标(m): {strategy1['drop_z']:.2f}")
    print(f"烟幕干扰弹起爆点的x坐标(m): {strategy1['det_x']:.2f}")
    print(f"烟幕干扰弹起爆点的y坐标(m): {strategy1['det_y']:.2f}")
    print(f"烟幕干扰弹起爆点的z坐标(m): {strategy1['det_z']:.2f}")
    print(f"验证有效干扰时长(s): {t1:.6f}")
    print("-" * 70)

    # 验证策略2
    t2 = validate_strategy(** strategy2)
    print(f"策略2（优化结果4.58s）验证：")
    print(f"无人机运动方向(度): {strategy2['theta']:.4f}")
    print(f"无人机运动速度(m/s): {strategy2['v']:.4f}")
    print(f"烟幕干扰弹投放点的x坐标(m): {strategy2['drop_x']:.2f}")
    print(f"烟幕干扰弹投放点的y坐标(m): {strategy2['drop_y']:.2f}")
    print(f"烟幕干扰弹投放点的z坐标(m): {strategy2['drop_z']:.2f}")
    print(f"烟幕干扰弹起爆点的x坐标(m): {strategy2['det_x']:.2f}")
    print(f"烟幕干扰弹起爆点的y坐标(m): {strategy2['det_y']:.2f}")
    print(f"烟幕干扰弹起爆点的z坐标(m): {strategy2['det_z']:.2f}")
    print(f"验证有效干扰时长(s): {t2:.6f}")
    print("=" * 70)

    # 选择最优策略
    if t1 >= t2 and t1 > 1e-6:
        print(f"结论：策略1有效（{t1:.6f}s），优于策略2")
    elif t2 > t1 and t2 > 1e-6:
        print(f"结论：策略2有效（{t2:.6f}s），优于策略1")
    else:
        print("结论：两个策略均无效，请重新优化")


if __name__ == "__main__":
    main()
