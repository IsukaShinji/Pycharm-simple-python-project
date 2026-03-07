# 烟幕遮蔽时长计算（修正遮蔽逻辑：所有采样点全遮挡才有效）
import numpy as np

# -------------------------- 【手动修改区】：直接填写参数 --------------------------
# 1. 每个高度层的圆柱面均匀采样点数量（可填12、24、36等整数）
num_cyl_points_per_layer = 36  # 示例：每层24个点，可改12/36等

# 2. 时间采样间隔（单位：秒，可填0.01、0.001、0.0001等）
dt = 0.0001  # 示例：0.0001秒精度，可改0.001/0.01等


# -------------------------- 【固定参数区】：严格匹配A题.pdf --------------------------
# 假目标（原点，）
O = np.array([0, 0, 0])

# 导弹M1参数（初始位置(20000,0,2000)，速度300m/s，）
missile_init = np.array([20000, 0, 2000])
v_missile = 300

# 无人机FY1参数（初始(17800,0,1800)，速度120m/s，投放延迟1.5s，起爆延迟3.6s，）
drone_init = np.array([17800, 0, 1800])
v_drone = 120
t_flight_drone = 1.5  # 受领任务后飞行1.5s投放
t_det_delay = 3.6      # 投放后3.6s爆炸

# 烟雾参数（半径10m，下沉3m/s，有效20s，）
smoke_radius = 10
smoke_speed_down = 3
smoke_valid_time = 20

# 真目标圆柱参数（下底面圆心(0,200,0)，半径7m，高10m，）
cyl_center = np.array([0, 200, 0])  # 下底面圆心（z=0）
cyl_radius = 7
cyl_height = 10
# 采样高度层（覆盖圆柱全高度：底部=下底面z=0，中部=z=5，顶部=上底面z=10）
cyl_z_layers = [0,2,4,6,8,10]

# 物理常量
g = 9.8  # 重力加速度


# -------------------------- 【核心计算区】：逻辑固定 --------------------------
# 1. 无人机投放点（向假目标x=0方向飞行，等高度，）
drone_dir = np.array([-1, 0, 0])  # 单位方向：x轴负方向（指向假目标）
drop_point = drone_init + v_drone * t_flight_drone * drone_dir
print(f"无人机投放点坐标：{drop_point.round(2)} (x, y, z)")

# 2. 干扰弹爆炸点（平抛运动，重力作用，）
det_point = np.array([
    drop_point[0] + v_drone * t_det_delay * drone_dir[0],  # x水平位移
    drop_point[1] + v_drone * t_det_delay * drone_dir[1],  # y无位移
    drop_point[2] - 0.5 * g * t_det_delay**2               # z自由下落（z轴向上）
])
t_det = t_flight_drone + t_det_delay  # 爆炸时刻（1.5+3.6=5.1s）
print(f"干扰弹爆炸点坐标：{det_point.round(2)} (x, y, z)")
print(f"干扰弹爆炸时刻：{t_det:.4f} s")

# 3. 圆柱面采样点生成（覆盖底部/中部/顶部，全高度层）
cyl_points = []
for z in cyl_z_layers:  # 遍历每个高度层（0、5、10m）
    theta_list = np.linspace(0, 2*np.pi, num_cyl_points_per_layer, endpoint=False)
    for theta in theta_list:
        x = cyl_center[0] + cyl_radius * np.cos(theta)  # 水平圆周x坐标
        y = cyl_center[1] + cyl_radius * np.sin(theta)  # 水平圆周y坐标
        cyl_points.append(np.array([x, y, z]))  # 每个高度层生成指定数量的点
cyl_points = np.array(cyl_points)
total_cyl_points = len(cyl_points)
print(f"圆柱采样点总数：{total_cyl_points}（{len(cyl_z_layers)}个高度层，每层{num_cyl_points_per_layer}个）")

# 4. 导弹实时位置（指向假目标直线飞行，）
def calc_missile_pos(t):
    L_total = np.linalg.norm(missile_init - O)  # 导弹到假目标初始距离
    T_missile_total = L_total / v_missile       # 导弹总飞行时间（≈66.9992s）
    if t >= T_missile_total:
        return None  # 导弹已命中假目标，无效
    dir_vec = (O - missile_init) / L_total     # 指向假目标的单位方向向量
    return missile_init + v_missile * t * dir_vec

# 5. 烟雾球心实时位置（起爆后匀速下沉，）
def calc_smoke_center(t):
    if t < t_det or t > t_det + smoke_valid_time:
        return None  # 烟雾未起爆或已失效（有效20s）
    z_center = det_point[2] - smoke_speed_down * (t - t_det)  # 竖直下沉
    return np.array([det_point[0], det_point[1], z_center])  # 水平位置不变

# 6. 线段-球体相交判断（遮蔽逻辑：导弹→圆柱点连线是否碰到烟雾球，）
def segment_intersects_sphere(A, B, O_sphere, r):
    AB = B - A  # 线段向量（导弹→圆柱点）
    AO = A - O_sphere  # 向量（导弹→烟雾球心）
    a = np.sum(AB**2)  # 线段长度平方（避免开方，提升效率）
    if a == 0:
        return np.sum((A - O_sphere)**2) <= r**2  # 特殊情况：点在球内
    b = 2 * np.sum(AO * AB)  # 二次方程一次项系数
    c = np.sum(AO**2) - r**2  # 二次方程常数项
    delta = b**2 - 4 * a * c  # 判别式
    if delta < 0:
        return False  # 无实根，线段与球不相交
    sqrt_delta = np.sqrt(delta)
    s1 = (-b - sqrt_delta) / (2 * a)  # 二次方程根1
    s2 = (-b + sqrt_delta) / (2 * a)  # 二次方程根2
    return (s1 <= 1) and (s2 >= 0)  # 根在[0,1]内→线段上有交点


# -------------------------- 【结果统计区】：修正遮蔽逻辑（全遮挡才有效） --------------------------
# 有效时间范围（烟雾有效 + 导弹未命中，取较早结束时刻）
t_start = t_det  # 烟雾起爆时刻（5.1s）
t_end = t_det + smoke_valid_time  # 烟雾理论失效时刻（5.1+20=25.1s）
T_missile_total = np.linalg.norm(missile_init - O) / v_missile
t_end = min(t_end, T_missile_total)  # 避免导弹已命中仍计算

# 遍历时间采样点，统计有效遮蔽时长（修正：所有圆柱点都相交才有效）
effective_duration = 0.0
t_samples = np.arange(t_start, t_end + dt, dt)  # 生成时间采样序列

for t in t_samples:
    # 1. 获取当前导弹位置（无效则跳过）
    missile_pos = calc_missile_pos(t)
    if missile_pos is None:
        continue
    # 2. 获取当前烟雾球心（无效则跳过）
    smoke_center = calc_smoke_center(t)
    if smoke_center is None:
        continue
    # 3. 修正：判断所有圆柱点是否都被遮挡（只要一个不相交，就无效）
    is_all_blocked = True  # 先假设全遮挡
    for cyl_point in cyl_points:
        if not segment_intersects_sphere(missile_pos, cyl_point, smoke_center, smoke_radius):
            is_all_blocked = False  # 找到一个未遮挡的点，标记无效
            break  # 无需判断其他点，直接跳出
    # 4. 只有全遮挡时，才累计有效时长
    if is_all_blocked:
        effective_duration += dt


# -------------------------- 【结果输出区】：明确标注修正后的逻辑 --------------------------
print("=" * 70)
print(f"《A题.pdf 问题1》计算结果（修正：所有采样点全遮挡才有效）")
print(f"参数：{len(cyl_z_layers)}个高度层（z=0/5/10m），每层{num_cyl_points_per_layer}个点，时间间隔{dt}s")
print("=" * 70)
print(f"1. 遮蔽判断标准：导弹到所有圆柱采样点的连线均与烟雾球相交，才算有效")
print(f"2. 圆柱采样点分布：底部(z=0m)、中部(z=5m)、顶部(z=10m)，共{total_cyl_points}个点")
print(f"3. 烟雾有效时间范围：[{t_start:.4f} s, {t_end:.4f} s]")
print(f"4. 导弹总飞行时间：{T_missile_total:.4f} s")
print(f"5. 时间采样总次数：{len(t_samples)}，采样间隔：{dt} s")
print(f"6. 烟幕对M1的有效遮蔽时长：{effective_duration:.6f} 秒")
print("=" * 70)