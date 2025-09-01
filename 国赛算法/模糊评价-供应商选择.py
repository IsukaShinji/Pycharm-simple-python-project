import numpy as np

# 1. 定义基础数据（因素集、评语集、权重向量）
# ------------------------------
# 因素集：评价供应商的4个核心指标
factors = ['产品质量', '价格竞争力', '交货准时率', '服务水平']
n_factors = len(factors)  # 指标数量：4

# 评语集：评价结果的4个等级（优秀>良好>一般>较差）
comments = ['优秀', '良好', '一般', '较差']
n_comments = len(comments)  # 等级数量：4

# 权重向量：通过AHP法计算（已通过一致性检验，总和为1）
weights = np.array([0.631, 0.297, 0.059, 0.013])

# 2. 定义隶属度函数（矩形隶属度函数：区间内隶属度=1，否则=0）
# ------------------------------
def calculate_membership(x, intervals):
    """
    计算单个指标值对各评语的隶属度向量
    :param x: 指标具体数值（如产品质量92分）
    :param intervals: 该指标对应的评语区间（顺序：优秀、良好、一般、较差）
    :return: 隶属度向量（长度=4，元素为0或1）
    """
    membership_vector = [0] * n_comments
    for i, (low, high) in enumerate(intervals):
        # 处理无穷大边界（如“优秀”对应≥90分，用np.inf表示上限）
        if (low <= x <= high) or (low <= x and high == np.inf) or (x <= high and low == -np.inf):
            membership_vector[i] = 1
            break  # 每个指标值仅属于一个评语区间（互斥划分）
    return membership_vector

# 3. 定义各指标的隶属度区间（根据业务规则设定）
# ------------------------------
membership_intervals = {
    '产品质量': [(90, np.inf), (80, 89), (70, 79), (0, 69)],  # 优秀≥90分，良好80-89分...
    '价格竞争力': [(0, 0.9), (0.91, 1.0), (1.01, 1.1), (1.11, np.inf)],  # 优秀≤0.9（低于市场价10%）...
    '交货准时率': [(95, np.inf), (90, 94), (85, 89), (0, 84)],  # 优秀≥95%，良好90-94%...
    '服务水平': [(9, np.inf), (8, 8.9), (7, 7.9), (0, 6.9)]  # 优秀≥9分，良好8-8.9分...
}

# 4. 模拟3个候选供应商的原始数据（指标值顺序与factors一致）
# ------------------------------
suppliers_data = {
    '供应商A': [92, 0.88, 96, 8.5],  # 产品质量92(优秀)、价格0.88(优秀)、交货96%(优秀)、服务8.5(良好)
    '供应商B': [87, 0.95, 92, 9.2],  # 产品质量87(良好)、价格0.95(良好)、交货92%(良好)、服务9.2(优秀)
    '供应商C': [78, 1.08, 88, 7.3]   # 产品质量78(一般)、价格1.08(一般)、交货88%(一般)、服务7.3(一般)
}

# 5. 计算每个供应商的隶属度矩阵（4行：指标；4列：评语）
# ------------------------------
suppliers_membership = {}
for supplier_name, data in suppliers_data.items():
    membership_matrix = []
    for i, factor in enumerate(factors):
        indicator_value = data[i]  # 当前指标值（如供应商A的产品质量=92）
        intervals = membership_intervals[factor]  # 当前指标的评语区间
        # 计算该指标的隶属度向量
        mv = calculate_membership(indicator_value, intervals)
        membership_matrix.append(mv)
    # 转换为numpy数组（方便后续矩阵运算）
    suppliers_membership[supplier_name] = np.array(membership_matrix)

# 6. 模糊综合计算（加权平均法：权重×隶属度矩阵）
# ------------------------------
suppliers_evaluation = {}
for supplier_name, R in suppliers_membership.items():
    # 矩阵乘法：权重向量(1×4) × 隶属度矩阵(4×4) = 综合评价向量(1×4)
    B = weights @ R
    suppliers_evaluation[supplier_name] = B

# 7. 结果分析：确定每个供应商的最终评价等级
# ------------------------------
results = {}
for supplier_name, B in suppliers_evaluation.items():
    # 找到综合评价向量中最大值的索引（对应评语集的等级）
    max_idx = np.argmax(B)
    final_grade = comments[max_idx]
    results[supplier_name] = (B, final_grade)

# 8. 输出结果并推荐最优供应商
# ------------------------------
print("=" * 60)
print("模糊综合评价结果（企业选择最优供应商）")
print("=" * 60)
for supplier_name, (B, grade) in results.items():
    print(f"供应商：{supplier_name}")
    print(f"  综合评价向量：{B.round(3)}（对应等级：{comments}）")
    print(f"  最终评价等级：{grade}")
    print("-" * 60)

# 定义等级优先级（数值越大等级越高），用于推荐最优供应商
grade_priority = {'优秀': 3, '良好': 2, '一般': 1, '较差': 0}
# 排序逻辑：先按等级优先级，再按对应等级的得分（得分越高越优）
best_supplier = max(
    results.items(),
    key=lambda x: (grade_priority[x[1][1]], x[1][0][grade_priority[x[1][1]]])
)[0]

print(f"\n✨ 基于模糊综合评价的最优供应商推荐：{best_supplier} ✨")