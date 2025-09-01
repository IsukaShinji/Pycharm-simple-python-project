import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ---------------------- 1. 基础配置（固定附件路径） ----------------------
BASE_PATH = r"C:\Users\21165\Desktop\23数模C题"
FILE_PATHS = {
    "附件1": f"{BASE_PATH}/附件1.xlsx",
    "附件2": f"{BASE_PATH}/附件2.xlsx",
    "附件3": f"{BASE_PATH}/附件3.xlsx",
    "附件4": f"{BASE_PATH}/附件4.xlsx"
}
TARGET_DATE = pd.to_datetime("2023-07-01").date()  # 目标日期：7月1日（周六）
CANDIDATE_PERIOD = ["2023-06-24", "2023-06-30"]  # 候选单品筛选周期：6月24-30日
DEMAND_SATISFY_RATIO =0.1# 需求满足度要求：品类实际销量≥历史平均？%

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


# ---------------------- 2. 数据加载与预处理（第三问专属逻辑） ----------------------
def load_and_preprocess_data():
    """
    功能：加载4个附件，筛选候选单品（6.24-6.30有销售），计算关键参数
    返回：
        - candidate_items：候选单品列表（单品编码+名称+品类）
        - item_cost：单品7月1日预测成本（6.24-6.30平均批发价）
        - item_loss：单品损耗率（来自附件4“附件4”工作表）
        - cat_demand_base：品类需求基准（6.24-6.30日均销量）
        - item_demand_models：单品需求模型（售价→需求量）
    """
    # 2.1 加载附件1（单品-品类映射），统一单品编码为str
    df1 = pd.read_excel(FILE_PATHS["附件1"], usecols=["单品编码", "单品名称", "分类名称"])
    df1["单品编码"] = df1["单品编码"].astype(str)  # 统一为字符串类型
    df1 = df1.drop_duplicates(subset=["单品编码"])
    print(f"附件1：{len(df1)}个单品，{df1['分类名称'].nunique()}个品类")

    # 2.2 加载附件2（销售流水），修复候选单品筛选逻辑
    df2 = pd.read_excel(
        FILE_PATHS["附件2"],
        usecols=["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型"]
    )
    # 关键修复1：统一单品编码为str，避免筛选时匹配失效
    df2["单品编码"] = df2["单品编码"].astype(str)
    # 关键修复2：处理销售日期（确保格式正确）+ 去除销售类型空格（避免筛选不到数据）
    df2["销售日期"] = pd.to_datetime(df2["销售日期"], errors="coerce").dt.date
    df2["销售类型"] = df2["销售类型"].str.strip()  # 去除字段值前后空格（如“销售 ”→“销售”）
    # 筛选逻辑优化：保留“销售”类型（排除退货/赠品等），允许少量异常值
    df2 = df2[df2["销售类型"].isin(["销售", "正常销售"])]  # 放宽类型筛选，适配可能的字段值差异
    # 筛选候选周期（6.24-6.30），增加数据检查
    start_candidate = pd.to_datetime(CANDIDATE_PERIOD[0]).date()
    end_candidate = pd.to_datetime(CANDIDATE_PERIOD[1]).date()
    df2_candidate = df2[(df2["销售日期"] >= start_candidate) & (df2["销售日期"] <= end_candidate)]

    # 检查候选周期销售数据是否为空（避免后续候选单品为0）
    if len(df2_candidate) == 0:
        print(f"警告：6.24-6.30周期内无销售数据！请检查销售流水日期范围或销售类型筛选条件")
        # 若为空，扩大日期范围（如6.20-6.30）作为备选，确保后续流程能运行
        start_backup = pd.to_datetime("2023-06-20").date()
        df2_candidate = df2[(df2["销售日期"] >= start_backup) & (df2["销售日期"] <= end_candidate)]
        print(f"已自动扩大日期范围为6.20-6.30，当前候选周期数据量：{len(df2_candidate)}条")

    # 候选单品：候选周期内有销售的单品（此时单品编码类型已统一，可正确匹配）
    candidate_item_codes = df2_candidate["单品编码"].unique()
    candidate_items = df1[df1["单品编码"].isin(candidate_item_codes)].copy()
    print(f"候选单品：{len(candidate_items)}个（{start_candidate}-{end_candidate}有销售）")

    # 2.3 加载附件3（批发价），统一单品编码类型，避免merge类型错误
    df3 = pd.read_excel(FILE_PATHS["附件3"], usecols=["日期", "单品编码", "批发价格(元/千克)"])
    df3["日期"] = pd.to_datetime(df3["日期"], errors="coerce").dt.date
    df3["单品编码"] = df3["单品编码"].astype(str)  # 关键：统一为str，解决merge类型不匹配
    # 筛选候选周期批发价数据
    df3_candidate = df3[(df3["日期"] >= start_candidate) & (df3["日期"] <= end_candidate)]
    # 计算每个单品的平均批发价（作为7月1日成本）
    item_cost = df3_candidate.groupby("单品编码")["批发价格(元/千克)"].agg(
        lambda x: x.mean() if len(x) >= 3 else df3[df3["单品编码"] == x.name]["批发价格(元/千克)"].mean()
    ).reset_index()
    item_cost.columns = ["单品编码", "预测成本(元/千克)"]
    item_cost["单品编码"] = item_cost["单品编码"].astype(str)
    # 补充无批发价的单品（用品类平均成本）
    cat_cost = df3_candidate.merge(df1, on="单品编码", how="left").groupby("分类名称")["批发价格(元/千克)"].mean()
    for _, row in candidate_items.iterrows():
        if row["单品编码"] not in item_cost["单品编码"].values:
            cat_avg = cat_cost.get(row["分类名称"], 5.0)  # 默认5元/千克
            item_cost = pd.concat([item_cost, pd.DataFrame({
                "单品编码": [row["单品编码"]],
                "预测成本(元/千克)": [cat_avg]
            })], ignore_index=True)

    # 2.4 加载附件4（单品损耗率），统一单品编码类型
    df4 = pd.read_excel(FILE_PATHS["附件4"], sheet_name="附件4", usecols=["单品编码", "损耗率(%)"])
    df4["单品编码"] = df4["单品编码"].astype(str)  # 统一为str，避免后续匹配错误
    df4["损耗率(小数)"] = df4["损耗率(%)"] / 100
    # 补充无损耗率的单品（用品类平均损耗率）
    cat_loss = df4.merge(df1, on="单品编码", how="left").groupby("分类名称")["损耗率(小数)"].mean()
    item_loss = candidate_items.merge(df4, on="单品编码", how="left")
    for idx, row in item_loss.iterrows():
        if pd.isna(row["损耗率(小数)"]):
            item_loss.loc[idx, "损耗率(小数)"] = cat_loss.get(row["分类名称"], 0.1)  # 默认10%
    item_loss = item_loss[["单品编码", "损耗率(小数)"]]

    # 2.5 计算品类需求基准（候选周期日均销量）
    df2_cat_daily = df2_candidate.merge(df1, on="单品编码", how="left").groupby(
        ["分类名称", "销售日期"]
    )["销量(千克)"].sum().reset_index()
    cat_demand_base = df2_cat_daily.groupby("分类名称")["销量(千克)"].mean().reset_index()
    cat_demand_base.columns = ["分类名称", "品类日均需求(千克)"]

    # 2.6 拟合单品需求模型（逻辑不变，仅确保单品编码匹配）
    item_demand_models = {}
    for _, item in candidate_items.iterrows():
        item_code = item["单品编码"]
        # 提取该单品候选周期的销售数据（单品编码已统一，可正确匹配）
        item_sales = df2_candidate[df2_candidate["单品编码"] == item_code][
            ["销售单价(元/千克)", "销量(千克)"]
        ].dropna()
        if len(item_sales) < 5:  # 数据不足，用品类平均模型
            cat = item["分类名称"]
            cat_sales = df2_candidate.merge(df1, on="单品编码", how="left")
            cat_sales = cat_sales[cat_sales["分类名称"] == cat][["销售单价(元/千克)", "销量(千克)"]].dropna()
            if len(cat_sales) < 10:  # 品类数据也不足，用线性默认模型
                item_demand_models[item_code] = {"type": "linear", "a": 100, "b": -5, "r2": 0.5}
                continue
            else:
                x, y = cat_sales["销售单价(元/千克)"].values, cat_sales["销量(千克)"].values
        else:
            x, y = item_sales["销售单价(元/千克)"].values, item_sales["销量(千克)"].values

        # 拟合多种模型，选R²最高的（逻辑不变）
        models = []
        # 线性模型
        try:
            lin_model = LinearRegression()
            lin_model.fit(x.reshape(-1, 1), y)
            y_pred = lin_model.predict(x.reshape(-1, 1))
            r2_lin = r2_score(y, y_pred)
            models.append(("linear", lin_model.intercept_, lin_model.coef_[0], r2_lin))
        except:
            models.append(("linear", 0, 0, -1))

        # Logistic模型（需求饱和）
        def logistic_model(p, K, a, b):
            return K / (1 + np.exp(-a * (p - b)))

        try:
            init_K = y.max() * 1.2
            init_b = np.median(x)
            init_a = 1 / (x.std() + 1e-6)
            popt, _ = curve_fit(
                logistic_model, x, y, p0=[init_K, init_a, init_b],
                bounds=([10, 0.1, x.min()], [1e5, 10, x.max()])
            )
            y_pred = logistic_model(x, *popt)
            r2_log = r2_score(y, y_pred)
            models.append(("logistic", *popt, r2_log))
        except:
            models.append(("logistic", 0, 0, 0, -1))

        # 选择最优模型（逻辑不变）
        models = [m for m in models if m[-1] > 0]
        if not models:
            best_model = ("linear", 100, -5, 0.5)
        else:
            best_model = max(models, key=lambda x: x[-1])

        if best_model[0] == "linear":
            item_demand_models[item_code] = {
                "type": "linear", "a": best_model[1], "b": best_model[2], "r2": best_model[3]
            }
        else:
            item_demand_models[item_code] = {
                "type": "logistic", "K": best_model[1], "a": best_model[2], "b": best_model[3], "r2": best_model[4]
            }

    # 整合返回数据（逻辑不变）
    candidate_items = candidate_items.merge(item_cost, on="单品编码", how="left")
    candidate_items = candidate_items.merge(item_loss, on="单品编码", how="left")
    return candidate_items, item_demand_models, cat_demand_base


# ---------------------- 3. 遗传算法核心实现（第三问优化逻辑） ----------------------
class GA_Optimizer:
    def __init__(self, candidate_items, item_demand_models, cat_demand_base,
                 pop_size=50, max_iter=100, cross_rate=0.8, mutate_rate=0.1):
        """
        遗传算法优化器初始化
        参数：
            - candidate_items：候选单品数据框
            - item_demand_models：单品需求模型字典
            - cat_demand_base：品类需求基准
            - pop_size：种群规模
            - max_iter：最大迭代次数
            - cross_rate：交叉概率
            - mutate_rate：变异概率
        """
        self.candidate = candidate_items  # 候选单品（index对应染色体位置）
        self.item_models = item_demand_models  # 需求模型
        self.cat_demand = cat_demand_base  # 品类需求基准
        self.n_items = len(candidate_items)  # 候选单品总数
        self.pop_size = pop_size  # 种群规模
        self.max_iter = max_iter  # 最大迭代次数
        self.cross_rate = cross_rate  # 交叉概率
        self.mutate_rate = mutate_rate  # 变异概率

        # 决策变量约束
        self.min_selected = 27  # 最小选中单品数
        self.max_selected = 33  # 最大选中单品数
        self.min_R = 2.5  # 单品最小补货量（kg）
        self.min_m = 0.0  # 最小加成率（0%）
        self.max_m = 1.0  # 最大加成率（100%）
        self.weekend_coeff = 1.2  # 周末需求系数（7.1是周六）

    def _predict_demand(self, item_code, price):
        """预测单品需求量：需求模型×周末系数"""
        model = self.item_models[item_code]
        if model["type"] == "linear":
            demand = model["a"] + model["b"] * price
        else:
            demand = model["K"] / (1 + np.exp(-model["a"] * (price - model["b"])))
        return max(demand * self.weekend_coeff, 0.1)  # 最低0.1kg

    def _calculate_profit(self, chromosome):
        """
        计算染色体（个体）的总收益（适应度）
        染色体结构：每个单品对应3个基因 [select(0/1), R(补货量), m(加成率)]
        """
        # 解析染色体
        selected_mask = chromosome[:, 0].astype(int)  # 选中标记
        R = chromosome[:, 1]  # 补货量
        m = chromosome[:, 2]  # 加成率
        selected_idx = np.where(selected_mask == 1)[0]

        # 约束1：选中单品数检查
        if len(selected_idx) < self.min_selected or len(selected_idx) > self.max_selected:
            return -1e9  # 不满足则收益极小

        total_profit = 0.0
        cat_actual_sales = {}  # 品类实际销量统计

        # 逐单品计算收益
        for idx in selected_idx:
            item = self.candidate.iloc[idx]
            item_code = item["单品编码"]
            cost = item["预测成本(元/千克)"]
            loss_rate = item["损耗率(小数)"]

            # 约束2：补货量和加成率检查
            if R[idx] < self.min_R or m[idx] < self.min_m or m[idx] > self.max_m:
                return -1e9

            # 计算售价、需求量、实际销量
            price = cost * (1 + m[idx])
            demand = self._predict_demand(item_code, price)
            available = R[idx] * (1 - loss_rate)  # 可售量（扣除损耗）
            actual_sales = min(available, demand)  # 实际销量

            # 累加收益
            profit = actual_sales * price - R[idx] * cost
            if profit < -1000:  # 单品亏损过大（排除异常）
                return -1e9
            total_profit += profit

            # 累加品类实际销量
            cat = item["分类名称"]
            cat_actual_sales[cat] = cat_actual_sales.get(cat, 0) + actual_sales

        # 约束3：品类需求满足度检查
        for _, row in self.cat_demand.iterrows():
            cat = row["分类名称"]
            base_demand = row["品类日均需求(千克)"]
            actual = cat_actual_sales.get(cat, 0)
            if actual < base_demand * DEMAND_SATISFY_RATIO:
                return -1e9 * (base_demand * DEMAND_SATISFY_RATIO - actual + 1)  # 需求缺口越大，收益越低

        return total_profit if total_profit > 0 else -1e9  # 总收益非负

    def _init_population(self):
        """初始化种群：每个个体是(n_items, 3)的染色体矩阵"""
        population = []
        for _ in range(self.pop_size):
            # 随机生成选中标记（确保数量在27-33）
            selected_count = np.random.randint(self.min_selected, self.max_selected + 1)
            selected_mask = np.zeros(self.n_items)
            selected_idx = np.random.choice(self.n_items, selected_count, replace=False)
            selected_mask[selected_idx] = 1

            # 随机生成补货量（选中单品：2.5-200kg；未选中：0）
            R = np.zeros(self.n_items)
            R[selected_idx] = np.random.uniform(self.min_R, 200, selected_count)  # 最大200kg（仓储限制）

            # 随机生成加成率（选中单品：0-1；未选中：0）
            m = np.zeros(self.n_items)
            m[selected_idx] = np.random.uniform(self.min_m, self.max_m, selected_count)

            # 组合染色体
            chromosome = np.column_stack([selected_mask, R, m])
            population.append(chromosome)
        return np.array(population)

    def _selection(self, population, fitness):
        """锦标赛选择：从种群中选择优秀个体"""
        new_pop = []
        for _ in range(self.pop_size):
            # 随机选3个个体竞争
            competitors = np.random.choice(len(population), 3, replace=False)
            comp_fitness = [fitness[i] for i in competitors]
            best_idx = competitors[np.argmax(comp_fitness)]
            new_pop.append(population[best_idx])
        return np.array(new_pop)

    def _crossover(self, parent1, parent2):
        """单点交叉：交换两个亲本的染色体片段"""
        if np.random.random() > self.cross_rate:
            return parent1, parent2  # 不交叉

        # 随机选择交叉点（按单品分组，避免拆分单品基因）
        cross_point = np.random.randint(1, self.n_items)
        child1 = np.vstack([parent1[:cross_point], parent2[cross_point:]])
        child2 = np.vstack([parent2[:cross_point], parent1[cross_point:]])

        # 交叉后修复选中数（确保在27-33）
        for child in [child1, child2]:
            selected_count = int(child[:, 0].sum())
            if selected_count < self.min_selected:
                # 补充选中未选中的单品
                unselected = np.where(child[:, 0] == 0)[0]
                add_idx = np.random.choice(unselected, self.min_selected - selected_count, replace=False)
                child[add_idx, 0] = 1
                child[add_idx, 1] = np.random.uniform(self.min_R, 200, len(add_idx))
                child[add_idx, 2] = np.random.uniform(self.min_m, self.max_m, len(add_idx))
            elif selected_count > self.max_selected:
                # 取消部分选中单品
                selected = np.where(child[:, 0] == 1)[0]
                remove_idx = np.random.choice(selected, selected_count - self.max_selected, replace=False)
                child[remove_idx, 0] = 0
                child[remove_idx, 1] = 0
                child[remove_idx, 2] = 0

        return child1, child2

    def _mutation(self, chromosome):
        """变异：随机调整选中标记、补货量或加成率"""
        for idx in range(self.n_items):
            if np.random.random() < self.mutate_rate:
                # 变异类型1：调整选中标记
                if np.random.random() < 0.3:
                    current_selected = int(chromosome[idx, 0])
                    new_selected = 1 - current_selected
                    selected_count = int(chromosome[:, 0].sum())

                    # 确保变异后选中数仍在范围内
                    if (new_selected == 1 and selected_count < self.max_selected) or \
                            (new_selected == 0 and selected_count > self.min_selected):
                        chromosome[idx, 0] = new_selected
                        # 同步调整补货量和加成率
                        if new_selected == 1:
                            chromosome[idx, 1] = np.random.uniform(self.min_R, 200)
                            chromosome[idx, 2] = np.random.uniform(self.min_m, self.max_m)
                        else:
                            chromosome[idx, 1] = 0
                            chromosome[idx, 2] = 0
                # 变异类型2：调整补货量（仅选中单品）
                elif chromosome[idx, 0] == 1 and np.random.random() < 0.5:
                    chromosome[idx, 1] = max(np.random.normal(chromosome[idx, 1], 10), self.min_R)
                # 变异类型3：调整加成率（仅选中单品）
                else:
                    if chromosome[idx, 0] == 1:
                        chromosome[idx, 2] = np.clip(np.random.normal(chromosome[idx, 2], 0.1), self.min_m, self.max_m)
        return chromosome

    def optimize(self):
        """主优化流程：初始化→迭代（选择→交叉→变异）→输出最优解"""
        # 1. 初始化种群
        population = self._init_population()
        best_fitness = -1e9
        best_chromosome = None
        fitness_history = []

        # 2. 迭代优化
        for iter in range(self.max_iter):
            # 计算种群适应度
            fitness = [self._calculate_profit(chrom) for chrom in population]
            current_best_idx = np.argmax(fitness)
            current_best_fit = fitness[current_best_idx]
            current_best_chrom = population[current_best_idx]

            # 更新全局最优
            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                best_chromosome = current_best_chrom.copy()

            # 记录适应度历史
            fitness_history.append(np.mean(fitness))
            if (iter + 1) % 10 == 0:
                print(
                    f"迭代{iter + 1:3d}/{self.max_iter} | 平均收益：{np.mean(fitness):8.2f}元 | 最优收益：{best_fitness:8.2f}元")

            # 终止条件：适应度收敛（连续5代变化<1%）
            if iter > 5 and abs(fitness_history[-1] - fitness_history[-6]) / fitness_history[-6] < 0.01:
                print(f"迭代{iter + 1}代：适应度收敛，提前终止")
                break

            # 选择操作
            population = self._selection(population, fitness)

            # 交叉操作
            new_pop = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    child1, child2 = self._crossover(population[i], population[i + 1])
                    new_pop.append(child1)
                    new_pop.append(child2)
                else:
                    new_pop.append(population[i])
            population = np.array(new_pop)

            # 变异操作
            population = np.array([self._mutation(chrom) for chrom in population])

        # 3. 输出优化结果
        print(f"\n优化完成！最优总收益：{best_fitness:.2f}元")
        return best_chromosome, fitness_history


# ---------------------- 4. 结果解析与输出 ----------------------
def parse_best_solution(best_chromosome, candidate_items, item_demand_models):
    """解析最优染色体，生成补货定价决策表"""
    selected_mask = best_chromosome[:, 0].astype(int)
    R = best_chromosome[:, 1]
    m = best_chromosome[:, 2]
    selected_idx = np.where(selected_mask == 1)[0]

    # 生成决策表
    decision_result = []
    ga_optimizer = GA_Optimizer(candidate_items, item_demand_models, pd.DataFrame())  # 临时实例用于调用方法
    for idx in selected_idx:
        item = candidate_items.iloc[idx]
        cost = item["预测成本(元/千克)"]
        loss_rate = item["损耗率(小数)"]
        price = cost * (1 + m[idx])
        demand = ga_optimizer._predict_demand(item["单品编码"], price)
        available = R[idx] * (1 - loss_rate)
        actual_sales = min(available, demand)
        profit = actual_sales * price - R[idx] * cost

        decision_result.append({
            "单品编码": item["单品编码"],
            "单品名称": item["单品名称"],
            "分类名称": item["分类名称"],
            "预测成本(元/千克)": round(cost, 2),
            "成本加成率": round(m[idx], 3),
            "定价(元/千克)": round(price, 2),
            "补货量(千克)": round(R[idx], 2),
            "损耗率(%)": round(loss_rate * 100, 2),
            "预计需求量(千克)": round(demand, 2),
            "预计实际销量(千克)": round(actual_sales, 2),
            "单品收益(元)": round(profit, 2)
        })

    # 转换为DataFrame并补充品类汇总
    decision_df = pd.DataFrame(decision_result)
    cat_summary = decision_df.groupby("分类名称").agg({
        "单品编码": "count",
        "补货量(千克)": "sum",
        "预计实际销量(千克)": "sum",
        "单品收益(元)": "sum"
    }).reset_index()
    cat_summary.columns = ["分类名称", "选中单品数", "品类总补货量(千克)", "品类总销量(千克)", "品类总收益(元)"]

    return decision_df, cat_summary


def save_and_visualize(decision_df, cat_summary, fitness_history):
    """保存结果到Excel并可视化"""
    # 保存Excel
    with pd.ExcelWriter(f"{BASE_PATH}/20230701_单品补货定价决策表.xlsx", engine="openpyxl") as writer:
        decision_df.to_excel(writer, sheet_name="单品决策明细", index=False)
        cat_summary.to_excel(writer, sheet_name="品类汇总", index=False)
    print(f"\n决策表已保存至：{BASE_PATH}/20230701_单品补货定价决策表.xlsx")

    # 可视化1：适应度变化曲线
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_history, color="#1f77b4", linewidth=2, marker="o", markersize=4)
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("种群平均收益（元）", fontsize=12)
    plt.title("遗传算法优化过程：种群平均收益变化", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.savefig(f"{BASE_PATH}/GA_适应度变化曲线.png", bbox_inches="tight")
    plt.close()

    # 可视化2：品类收益分布
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_summary)))
    bars = plt.bar(cat_summary["分类名称"], cat_summary["品类总收益(元)"], color=colors)
    plt.xlabel("蔬菜品类", fontsize=12)
    plt.ylabel("品类总收益（元）", fontsize=12)
    plt.title("2023年7月1日各品类预计收益分布", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    # 标注收益值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f"{height:.0f}元", ha="center", va="bottom", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{BASE_PATH}/品类收益分布图.png", bbox_inches="tight")
    plt.close()

    # 打印关键信息
    print(f"\n=== 2023年7月1日单品补货定价决策摘要 ===")
    print(f"1. 选中单品总数：{len(decision_df)}个（满足27-33个约束）")
    print(f"2. 总预计收益：{decision_df['单品收益(元)'].sum():.2f}元")
    print(f"3. 各品类选中单品数：")
    for _, row in cat_summary.iterrows():
        print(f"   - {row['分类名称']:8s}：{row['选中单品数']}个，总收益{row['品类总收益(元)']:.2f}元")


# ---------------------- 5. 主程序（串联所有流程） ----------------------
if __name__ == "__main__":
    print("=" * 80)
    print("2023国赛C题问题3：单品补货定价遗传算法优化")
    print("=" * 80)

    # 步骤1：数据加载与预处理
    print("\n【步骤1：数据加载与预处理】")
    candidate_items, item_demand_models, cat_demand_base = load_and_preprocess_data()

    # 步骤2：初始化遗传算法优化器
    print("\n【步骤2：初始化遗传算法】")
    ga_optimizer = GA_Optimizer(
        candidate_items=candidate_items,
        item_demand_models=item_demand_models,
        cat_demand_base=cat_demand_base,
        pop_size=50,
        max_iter=100,
        cross_rate=0.8,
        mutate_rate=0.1
    )

    # 步骤3：执行优化
    print("\n【步骤3：遗传算法迭代优化】")
    best_chromosome, fitness_history = ga_optimizer.optimize()

    # 步骤4：结果解析与输出
    # 在主程序步骤4中修改：
    print("\n【步骤4：结果解析与输出】")
    if best_chromosome is None:
        print("Error：未找到满足约束条件的可行解，请调整约束参数后重试！")
    else:
        decision_df, cat_summary = parse_best_solution(best_chromosome, candidate_items, item_demand_models)
        save_and_visualize(decision_df, cat_summary, fitness_history)

    print(f"\n✅ 所有流程执行完成！结果文件已保存至：{BASE_PATH}")
    print(f"📊 生成文件清单：")
    print(f"1. 单品决策表：{BASE_PATH}/20230701_单品补货定价决策表.xlsx")
    print(f"2. 适应度曲线：{BASE_PATH}/GA_适应度变化曲线.png")
    print(f"3. 品类收益图：{BASE_PATH}/品类收益分布图.png")
    print("=" * 80)