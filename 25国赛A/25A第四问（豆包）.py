import numpy as np
import numba as nb
import random
from multiprocessing import Pool
import time
from tqdm import tqdm
import pandas as pd

# ====================== 1. 读取Excel表格数据 ======================
excel_path = "/mnt/result2.xlsx"  # 表格文件路径
df = pd.read_excel(excel_path)

# 提取无人机参数（按表格列名适配，假设列名与需求一致）
drone_info = {}
for idx, row in df.iterrows():
    drone_id = row["无人机编号"]
    if pd.notna(drone_id):
        drone_info[drone_id] = {
            "direction": row["无人机运动方向"],
            "speed": row["无人机运动速度 (m/s)"],
            "drop_pos": np.array([
                row["烟幕干扰弹投放点的x坐标 (m)"],
                row["烟幕干扰弹投放点的y坐标 (m)"],
                row["烟幕干扰弹投放点的z坐标 (m)"]
            ]),
            "detonate_pos": np.array([
                row["烟幕干扰弹起爆点的x坐标 (m)"],
                row["烟幕干扰弹起爆点的y坐标 (m)"],
                row["烟幕干扰弹起爆点的z坐标 (m)"]
            ]),
            "effective_duration": row["有效干扰时长 (s)"]
        }

# ====================== 2. 核心物理与时间参数 ======================
gravity = 9.8
smoke_sink_speed = 3.0    # 烟幕垂直下沉速度
smoke_radius = 10.0       # 烟幕有效半径
smoke_duration = 20.0     # 烟幕严格生效20s

TIME_START = 0.0
TIME_END = 60.0           # 统计0-60s内的遮蔽
TIME_STEP = 0.1
TIME_STEP_COUNT = int(np.ceil((TIME_END - TIME_START) / TIME_STEP))

# 真目标与导弹参数
TRUE_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TRUE_TARGET_RADIUS = 7.0
TRUE_TARGET_HEIGHT = 10.0
TARGET_SIDE_THETA = 18
TARGET_SIDE_Z = 5
TARGET_TOP_R = 3
TARGET_TOP_THETA = 18

MISSILE_INIT_POS = np.array([20000.0, 0.0, 2000.0])
MISSILE_SPEED = 300.0
FAKE_TARGET_POS = np.array([0.0, 0.0, 0.0])

# ====================== 3. 基础工具函数（Numba加速） ======================
@nb.jit(nopython=True, fastmath=True, cache=True)
def sample_true_target():
    total_points = TARGET_SIDE_THETA * TARGET_SIDE_Z + TARGET_TOP_R * TARGET_TOP_THETA
    points = np.zeros((total_points, 3), dtype=np.float64)
    idx = 0

    thetas_side = np.linspace(0, 2*np.pi, TARGET_SIDE_THETA)[:-1]
    z_side = np.linspace(0, TRUE_TARGET_HEIGHT, TARGET_SIDE_Z)
    for theta in thetas_side:
        x_off = TRUE_TARGET_RADIUS * np.cos(theta)
        y_off = TRUE_TARGET_RADIUS * np.sin(theta)
        for z in z_side:
            points[idx, 0] = TRUE_TARGET_CENTER[0] + x_off
            points[idx, 1] = TRUE_TARGET_CENTER[1] + y_off
            points[idx, 2] = z
            idx += 1

    r_top = np.linspace(0, TRUE_TARGET_RADIUS, TARGET_TOP_R)
    thetas_top = np.linspace(0, 2*np.pi, TARGET_TOP_THETA)[:-1]
    for r in r_top:
        for theta in thetas_top:
            x_off = r * np.cos(theta)
            y_off = r * np.sin(theta)
            points[idx, 0] = TRUE_TARGET_CENTER[0] + x_off
            points[idx, 1] = TRUE_TARGET_CENTER[1] + y_off
            points[idx, 2] = TRUE_TARGET_HEIGHT
            idx += 1
    return points

@nb.jit(nopython=True, fastmath=True, cache=True)
def get_missile_pos(t):
    if t < TIME_START or t > TIME_END:
        return np.array([np.nan, np.nan, np.nan])
    dir_vec = FAKE_TARGET_POS - MISSILE_INIT_POS
    dir_norm = np.linalg.norm(dir_vec)
    return MISSILE_INIT_POS + (dir_vec / dir_norm) * MISSILE_SPEED * t

@nb.jit(nopython=True, fastmath=True, cache=True)
def get_smoke_center(detonate_time, drop_pos, t):
    if t < detonate_time or t > (detonate_time + smoke_duration) or t > TIME_END:
        return np.array([np.nan, np.nan, np.nan])
    return np.array([
        drop_pos[0],
        drop_pos[1],
        drop_pos[2] - smoke_sink_speed * (t - detonate_time)
    ])

@nb.jit(nopython=True, fastmath=True, cache=True)
def is_sight_blocked(missile_pos, target_point, smoke_center):
    if np.any(np.isnan(missile_pos)) or np.any(np.isnan(smoke_center)):
        return False
    line_vec = target_point - missile_pos
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return False
    proj_coeff = np.dot(smoke_center - missile_pos, line_vec) / (line_len ** 2)
    if proj_coeff < 0.0 or proj_coeff > 1.0:
        dist_missile = np.linalg.norm(smoke_center - missile_pos)
        dist_target = np.linalg.norm(smoke_center - target_point)
        min_dist = min(dist_missile, dist_target)
    else:
        proj_point = missile_pos + proj_coeff * line_vec
        min_dist = np.linalg.norm(smoke_center - proj_point)
    return min_dist <= smoke_radius + 1e-6

# ====================== 4. 遮蔽时间计算（含表格参数） ======================
@nb.jit(nopython=True, fastmath=True, cache=True)
def calc_total_block_time(target_points, fy1_detonate, fy1_drop, fy2_detonate, fy2_drop, fy3_detonate, fy3_drop):
    block_count = 0
    for i in range(TIME_STEP_COUNT):
        t = TIME_START + i * TIME_STEP

        missile_pos = get_missile_pos(t)

        f1_smoke = get_smoke_center(fy1_detonate, fy1_drop, t)
        f2_smoke = get_smoke_center(fy2_detonate, fy2_drop, t)
        f3_smoke = get_smoke_center(fy3_detonate, fy3_drop, t)

        all_blocked = True
        for j in range(target_points.shape[0]):
            p = target_points[j]
            blocked = (is_sight_blocked(missile_pos, p, f1_smoke) or
                       is_sight_blocked(missile_pos, p, f2_smoke) or
                       is_sight_blocked(missile_pos, p, f3_smoke))
            if not blocked:
                all_blocked = False
                break
        if all_blocked:
            block_count += 1
    return block_count * TIME_STEP

# ====================== 5. 主函数：整合流程并打印结果 ======================
def main():
    total_start = time.time()
    print("=== 无人机烟幕遮蔽计算（烟雾生效20s，实时找点）===")

    # 1. 采样真目标
    print("\n步骤1：真目标采样...")
    target_points = sample_true_target()
    print(f"真目标采样完成，共{target_points.shape[0]}个点")

    # 2. 提取表格中无人机参数（FY1、FY2、FY3）
    fy1 = drone_info.get("FY1", {})
    fy2 = drone_info.get("FY2", {})
    fy3 = drone_info.get("FY3", {})

    fy1_detonate = fy1.get("effective_duration", 0.0)  # 假设有效干扰时长为起爆延迟相关
    fy1_drop = fy1.get("drop_pos", np.array([17800.0, 0.0, 1800.0]))  # 若表格无数据，用默认值

    fy2_detonate = fy2.get("effective_duration", 25.0)
    fy2_drop = fy2.get("drop_pos", np.array([12000.0, 1400.0, 1400.0]))

    fy3_detonate = fy3.get("effective_duration", 30.0)
    fy3_drop = fy3.get("drop_pos", np.array([6000.0, -3000.0, 700.0]))

    # 3. 计算总遮蔽时间
    print("\n步骤2：计算总遮蔽时间...")
    total_block = calc_total_block_time(
        target_points, fy1_detonate, fy1_drop, fy2_detonate, fy2_drop, fy3_detonate, fy3_drop
    )
    print(f"总遮蔽时间：{total_block:.2f}s（0-60s内，烟雾仅生效20s）")

    # 4. 打印表格中各无人机参量
    print("\n=== 表格中无人机参量 ===")
    for drone_id, info in drone_info.items():
        print(f"\n无人机 {drone_id}：")
        print(f"  运动方向：{info.get('direction', 'N/A')} 度")
        print(f"  运动速度：{info.get('speed', 'N/A')} m/s")
        print(f"  投放点坐标：{info.get('drop_pos', 'N/A')} m")
        print(f"  起爆点坐标：{info.get('detonate_pos', 'N/A')} m")
        print(f"  有效干扰时长：{info.get('effective_duration', 'N/A')} s")

    print(f"\n总耗时：{time.time()-total_start:.2f}s")

if __name__ == "__main__":
    main()