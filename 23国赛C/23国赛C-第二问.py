import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit, minimize
import warnings

# 在文件开头导入依赖（若未导入）
import numpy as np


# 定义Logistic模型（需求存在饱和上限，符合生鲜市场容量）
def logistic_model(x, K, a, b):
    """
    D = K / (1 + exp(-a*(P - b)))
    K: 市场最大需求（饱和值）
    a: 价格对需求的影响强度
    b: 半饱和价格（需求为K/2时的售价）
    """
    return K / (1 + np.exp(-a * (x - b)))


warnings.filterwarnings('ignore')

# ---------------------- 1. 基础配置（你的C盘路径！） ----------------------
BASE_PATH = r"C:\Users\21165\Desktop\23数模C题"  # 确保是你的实际路径
FILE_PATHS = {
    "附件1": f"{BASE_PATH}/附件1.xlsx",
    "附件2": f"{BASE_PATH}/附件2.xlsx",
    "附件3": f"{BASE_PATH}/附件3.xlsx",
    "附件4": f"{BASE_PATH}/附件4.xlsx"
}
FUTURE_DATES = pd.date_range("2023-07-01", "2023-07-07").date.tolist()

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 6)


# ---------------------- 2. 数据加载与清洗（核心修改：附件4指定工作表“附件4”） ----------------------
def load_data():
    # 附件1：品类-单品映射（列名匹配实际数据）
    df1 = pd.read_excel(FILE_PATHS["附件1"], usecols=["单品编码", "分类名称"]).drop_duplicates()
    df1 = df1.dropna(subset=["单品编码", "分类名称"])
    df1["分类名称"] = df1["分类名称"].str.strip()
    print(f"附件1：{df1['分类名称'].nunique()}个品类，{len(df1)}条单品映射")

    # 附件2：销售数据（列名匹配实际数据）
    df2 = pd.read_excel(FILE_PATHS["附件2"],
                        usecols=["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型"])
    df2 = df2[df2["销售类型"] == "销售"]  # 仅保留实际销售记录
    df2["销售日期"] = pd.to_datetime(df2["销售日期"], errors="coerce").dt.date  # 转为日期格式
    df2 = df2.dropna(subset=["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)"])
    # 时间范围：题目指定2020-07-01至2023-06-30
    start_date = pd.to_datetime("2020-07-01").date()
    end_date = pd.to_datetime("2023-06-30").date()
    df2 = df2[(df2["销售日期"] >= start_date) & (df2["销售日期"] <= end_date)]
    print(f"附件2：{len(df2)}条有效销售记录（{start_date}至{end_date}）")

    # 附件3：成本数据（批发价，列名匹配实际数据）
    df3 = pd.read_excel(FILE_PATHS["附件3"], usecols=["日期", "单品编码", "批发价格(元/千克)"])
    df3["日期"] = pd.to_datetime(df3["日期"], errors="coerce").dt.date
    df3 = df3.dropna(subset=["日期", "单品编码", "批发价格(元/千克)"])
    df3 = df3[(df3["日期"] >= start_date) & (df3["日期"] <= end_date)]
    print(f"附件3：{len(df3)}条有效批发价记录")

    # 附件4：核心修改——指定工作表为“附件4”（第二个工作表）
    # 步骤1：查看附件4中“附件4”工作表的所有列名（调试用，确认列名匹配）
    df4_test = pd.read_excel(
        FILE_PATHS["附件4"],
        sheet_name="附件4"  # 关键修改：读取第二个工作表“附件4”
    )
    print(f"\n附件4（工作表“附件4”）的所有列名：{df4_test.columns.tolist()}")  # 打印列名确认（应为['单品编码','单品名称','损耗率(%)']）

    # 步骤2：读取附件4的“附件4”工作表，仅保留需要的列（单品编码、损耗率(%)）
    df4 = pd.read_excel(
        FILE_PATHS["附件4"],
        sheet_name="附件4",  # 强制读取第二个工作表“附件4”
        usecols=["单品编码", "损耗率(%)"]  # 列名与附件4“附件4”工作表完全匹配
    )
    # 去重：同一单品可能有多条记录，保留唯一值
    df4 = df4.drop_duplicates(subset=["单品编码"])
    # 损耗率转换为小数（便于后续计算）
    df4["损耗率(小数)"] = df4["损耗率(%)"] / 100

    # 关联附件1的品类信息，计算每个品类的平均损耗率
    df4_with_cat = df4.merge(
        df1[["单品编码", "分类名称"]],
        on="单品编码",
        how="inner"  # 仅保留有品类映射的单品（避免无意义数据）
    )
    # 按品类分组计算平均损耗率
    cat_loss = df4_with_cat.groupby("分类名称")["损耗率(小数)"].mean().reset_index()
    print(f"附件4（工作表“附件4”）：{cat_loss['分类名称'].nunique()}个品类的平均损耗率计算完成")

    return df1, df2, df3, cat_loss


# ---------------------- 3. 生成每日品类数据（逻辑不变，列名已匹配） ----------------------
def generate_cat_daily_data(df1, df2, df3):
    # 销售数据关联品类信息
    sales_with_cat = df2.merge(
        df1[["单品编码", "分类名称"]],
        on="单品编码",
        how="inner"
    )

    # 按“品类+日期”聚合销售数据：每日总销量、日均售价
    daily_sales = sales_with_cat.groupby(["分类名称", "销售日期"]).agg(
        销售总量_千克=pd.NamedAgg(column="销量(千克)", aggfunc="sum"),
        日均价_元每千克=pd.NamedAgg(column="销售单价(元/千克)", aggfunc="mean")
    ).reset_index()

    # 成本数据关联品类信息，并统一日期列名
    cost_with_cat = df3.merge(
        df1[["单品编码", "分类名称"]],
        on="单品编码",
        how="inner"
    ).rename(columns={"日期": "销售日期"})  # 与销售数据的日期列名统一

    # 按“品类+日期”聚合成本数据：每日平均成本
    daily_cost = cost_with_cat.groupby(["分类名称", "销售日期"]).agg(
        日成本_元每千克=pd.NamedAgg(column="批发价格(元/千克)", aggfunc="mean")
    ).reset_index()

    # 合并销售与成本数据，生成每日品类核心数据
    cat_daily_data = daily_sales.merge(
        daily_cost,
        on=["分类名称", "销售日期"],
        how="inner"  # 仅保留同时有销量和成本的记录
    )

    # 计算成本加成率（题目核心关联指标）
    cat_daily_data["成本加成率"] = (
            (cat_daily_data["日均价_元每千克"] - cat_daily_data["日成本_元每千克"])
            / cat_daily_data["日成本_元每千克"]
    )

    # 数据清洗：剔除异常值（确保后续模型拟合可靠）
    cat_daily_data = cat_daily_data[
        (cat_daily_data["销售总量_千克"] > 0) &  # 销量为正
        (cat_daily_data["日均价_元每千克"] > 0) &  # 售价为正
        (cat_daily_data["日成本_元每千克"] > 0) &  # 成本为正
        (cat_daily_data["成本加成率"] >= 0) &  # 加成率非负（不亏本）
        (cat_daily_data["成本加成率"] <= 1)  # 加成率≤100%（符合商超实际）
        ].reset_index(drop=True)

    print(f"生成每日品类数据：{len(cat_daily_data)}条记录，覆盖{cat_daily_data['分类名称'].nunique()}个品类")
    return cat_daily_data


# ---------------------- 4. 分析销售总量与成本加成定价的非线性关系 ----------------------
def analyze_sales_margin_relation(cat_daily_data, cat_loss):
    categories = cat_daily_data["分类名称"].unique()
    relation_results = []
    best_model_params = {}

    for cat in categories:
        cat_data = cat_daily_data[cat_daily_data["分类名称"] == cat].copy()
        # 获取当前品类的平均损耗率
        loss_rate = cat_loss[cat_loss["分类名称"] == cat]["损耗率(小数)"].values[0]

        if len(cat_data) < 20:
            # 定义x和y（修复未定义问题）
            x = cat_data["日均价_元每千克"].values
            y = cat_data["销售总量_千克"].values
            # 线性模型拟合（强制使用线性，避免复杂模型过拟合）
            x_lin = x.reshape(-1, 1)
            lin_model = LinearRegression()
            lin_model.fit(x_lin, y)
            y_pred = lin_model.predict(x_lin)
            r2 = r2_score(y, y_pred)
            a, b = lin_model.intercept_, lin_model.coef_[0]
            best_model_params[cat] = {"type": "linear", "a": a, "b": b, "r2": r2, "loss_rate": loss_rate}
            # 记录结果
            relation_results.append({
                "品类名称": cat,
                "样本量": len(cat_data),
                "最优模型类型": "线性模型",
                "最优模型R²": round(r2, 3),
                "模型方程": f"D={a:.2f}{b:+.2f}P",
                "品类平均损耗率": round(loss_rate, 4)
            })
            continue

        x = cat_data["日均价_元每千克"].values  # 售价（自变量）
        y = cat_data["销售总量_千克"].values  # 销量（因变量）
        models = []  # 存储模型：(名称, R², 参数1, 参数2, ..., 方程, AIC, 参数数量)

        # ====================== 1. 线性模型（基准，鲁棒性强） ======================
        try:
            x_lin = x.reshape(-1, 1)
            lin_model = LinearRegression()
            lin_model.fit(x_lin, y)
            y_pred = lin_model.predict(x_lin)
            r2 = r2_score(y, y_pred)
            a, b = lin_model.intercept_, lin_model.coef_[0]
            rss = ((y - y_pred) ** 2).sum()
            aic = len(x) * np.log(rss / len(x)) + 2 * 2  # 参数数量k=2（a, b）
            models.append(
                ("线性模型", r2, a, b, f"D={a:.2f}{b:+.2f}P", aic, 2)
            )
        except:
            models.append(("线性模型", -1, 0, 0, "拟合失败", np.inf, 0))

        # ====================== 2. 幂函数模型（优化约束：b≥0.5） ======================
        try:
            popt, _ = curve_fit(
                lambda x, a, b: a * (x ** (-b)),
                x[x > 0], y[x > 0],
                bounds=([1, 0.5], [1e4, 10]),  # 约束：a∈[1,1e4], b∈[0.5,10]（避免b过小）
                maxfev=2000
            )
            a, b = popt
            y_pred = a * (x ** (-b))
            r2 = r2_score(y, y_pred)
            rss = ((y - y_pred) ** 2).sum()
            aic = len(x) * np.log(rss / len(x)) + 2 * 2  # k=2
            models.append(
                ("幂函数模型", r2, a, b, f"D={a:.2f}×P^(-{b:.2f})", aic, 2)
            )
        except:
            models.append(("幂函数模型", -1, 0, 0, "拟合失败", np.inf, 0))

        # ====================== 3. 指数函数模型（优化约束：b≥0.01） ======================
        try:
            # 改进后（拟合后强制限制x>0）：
            valid_x = x[x > 1e-3]  # 售价≥0.001元（避免0附近波动）
            valid_y = y[x > 1e-3]
            if len(valid_x) < 5:  # 有效样本不足，跳过幂函数
                models.append(("指数函数模型", -1, 0, 0, "有效样本不足", np.inf, 0))
            else:
                popt, _ = curve_fit(
                    lambda x, a, b: a * np.exp(-b * x),
                    valid_x, valid_y,
                    bounds=([1, 0.01], [1e4, 10]),  # 约束a>0, b≥0.01
                    maxfev=2000
                )

            a, b = popt
            y_pred = a * np.exp(-b * x)
            r2 = r2_score(y, y_pred)
            rss = ((y - y_pred) ** 2).sum()
            aic = len(x) * np.log(rss / len(x)) + 2 * 2  # k=2
            models.append(
                ("指数函数模型", r2, a, b, f"D={a:.2f}×e^(-{b:.2f}P)", aic, 2)
            )
        except:
            models.append(("指数函数模型", -1, 0, 0, "拟合失败", np.inf, 0))

        # ====================== 4. Logistic模型（新增，需求有饱和上限） ======================
        try:
            # 改进后（动态估计）：
            init_K = y.max() * 1.2 if y.max() > 0 else 100  # 无销量时设默认值
            init_b = np.median(x)  # 用价格中位数替代均值，更抗异常值
            init_a = 1 / (x.std() + 1e-6)  # 价格波动越大，a越小（需求对价格越不敏感）

            popt, _ = curve_fit(
                logistic_model,
                x, y,
                p0=[init_K, init_a, init_b],
                bounds=([10, 0.1, x.min()], [1e5, 10, x.max()])  # K∈[10,1e5], a∈[0.1,10], b∈[minP, maxP]
            )
            K, a, b = popt
            y_pred = logistic_model(x, K, a, b)
            r2 = r2_score(y, y_pred)
            rss = ((y - y_pred) ** 2).sum()
            aic = len(x) * np.log(rss / len(x)) + 2 * 3  # k=3（K, a, b）
            models.append(
                ("Logistic模型", r2, K, a, b, f"D={K:.0f}/(1+e^(-{a:.2f}(P-{b:.2f})))", aic, 3)
            )
        except:
            models.append(("Logistic模型", -1, 0, 0, 0, "拟合失败", np.inf, 0))

        # ====================== 模型选择：R²优先 + AIC辅助 ======================
        valid_models = [m for m in models if 0 < m[1] <= 1]  # 仅保留有效模型（R²合理）

        # 确保best_model在所有情况下都有值
        if valid_models:
            # 排序规则：① R²降序 ② AIC升序（复杂度低更优）
            valid_models.sort(key=lambda m: (-m[1], m[6]))
            best_model = valid_models[0]
        else:
            # 关键修复：如果所有模型都无效，强制从models中找线性模型，找不到则手动创建默认值
            linear_candidates = [m for m in models if m[0] == "线性模型"]
            if linear_candidates:
                best_model = linear_candidates[0]
            else:
                # 完全找不到线性模型时，手动创建一个默认线性模型
                best_model = ("线性模型", -1, 0, 0, "默认线性模型（所有模型拟合失败）", np.inf, 2)

        # 提取最优模型参数（将这部分代码移出else块，确保在所有分支都能执行）
        best_name = best_model[0]
        best_r2 = best_model[1]
        # 根据模型参数数量提取（最后一个元素是参数数量）
        param_count = best_model[-1]
        best_params = best_model[2: 2 + param_count]  # 正确提取参数部分
        best_eq = best_model[4]
        best_aic = best_model[5]

        # 存储模型参数（关键修复：根据模型类型动态处理参数数量）
        if best_name == "线性模型":
            a, b = best_params  # 线性模型：2个参数
            best_model_params[cat] = {"type": "linear", "a": a, "b": b, "r2": best_r2, "loss_rate": loss_rate}
        elif best_name == "幂函数模型":
            a, b = best_params  # 幂函数模型：2个参数
            best_model_params[cat] = {"type": "power", "a": a, "b": b, "r2": best_r2, "loss_rate": loss_rate}
        elif best_name == "指数函数模型":
            a, b = best_params  # 指数函数模型：2个参数
            best_model_params[cat] = {"type": "exponential", "a": a, "b": b, "r2": best_r2, "loss_rate": loss_rate}
        elif best_name == "Logistic模型":
            # 关键修复：Logistic模型有3个参数(K, a, b)
            if len(best_params) == 3:
                K, a, b = best_params
                best_model_params[cat] = {"type": "logistic", "K": K, "a": a, "b": b, "r2": best_r2,
                                          "loss_rate": loss_rate}
            else:
                # 参数异常时的降级处理
                best_model_params[cat] = {"type": "linear", "a": 0, "b": 0, "r2": -1, "loss_rate": loss_rate}
        else:
            best_model_params[cat] = {"type": "none", "a": 0, "b": 0, "r2": 0, "loss_rate": loss_rate}

        # 绘制“销售总量-售价”关系图（保存到指定路径）
        fig, ax = plt.subplots(1, 1)
        # 散点图：原始数据
        ax.scatter(x, y, color="#1f77b4", alpha=0.6, s=50, label=f"样本点（n={len(cat_data)}）")
        # 最优模型拟合曲线
        x_fit = np.linspace(x.min(), x.max(), 100)  # 生成拟合曲线的x范围
        if best_name == "线性模型":
            y_fit = best_model_params[cat]["a"] + best_model_params[cat]["b"] * x_fit
        elif best_name == "幂函数模型":
            y_fit = best_model_params[cat]["a"] * (x_fit ** (-best_model_params[cat]["b"]))
        elif best_name == "Logistic模型":
            y_fit = logistic_model(x_fit, best_model_params[cat]["K"], best_model_params[cat]["a"],
                                   best_model_params[cat]["b"])
        else:  # 指数函数模型
            y_fit = best_model_params[cat]["a"] * np.exp(-best_model_params[cat]["b"] * x_fit)
        ax.plot(x_fit, y_fit, color="#ff7f0e", linewidth=3, label=f"最优模型：{best_eq}")
        # 标注关键信息（R²、模型对比）
        model_info = "\n".join([f"{m[0]}: R²={m[1]:.3f}" for m in models if m[1] >= 0])
        ax.text(0.05, 0.95, f"最优模型R²：{best_r2:.3f}\n{model_info}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        # 图表格式（中文黑体，清晰易读）
        ax.set_xlabel("售价（元/千克）", fontsize=12)
        ax.set_ylabel("销售总量（千克）", fontsize=12)
        ax.set_title(f"{cat} - 销售总量与售价的非线性关系", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        # 保存图片到C盘指定路径
        save_path = f"{BASE_PATH}/{cat}_销售总量-售价关系图.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ {cat}关系图已保存：{save_path}")

        # 记录当前品类的分析结果
        relation_results.append({
            "品类名称": cat,
            "样本量": len(cat_data),
            "最优模型类型": best_name,
            "最优模型R²": round(best_r2, 3),
            "模型方程": best_eq,
            "品类平均损耗率": round(best_model_params[cat]["loss_rate"], 4)
        })

    # 保存所有品类的关系分析结果到Excel
    relation_df = pd.DataFrame(relation_results)
    relation_df.to_excel(f"{BASE_PATH}/各品类销售总量-售价关系分析表.xlsx", index=False)
    print(f"✅ 关系分析汇总表已保存：{BASE_PATH}/各品类销售总量-售价关系分析表.xlsx")

    return best_model_params, relation_df


# ---------------------- 5. 收益计算（严格贴合题干净收益公式） ----------------------
def calculate_profit(m, R, cost, loss_rate, cat, best_model_params, date):
    """
    题干净收益公式：
    净收益 = min(补货量×(1-损耗率), 市场需求量) × 售价 - 补货量×成本
    其中：
    - m：成本加成率 → 售价 = 成本×(1+m)
    - R：补货总量（千克）
    - 市场需求量：由最优非线性模型+时间特征（周末/7月）共同决定
    """
    # 1. 计算售价（严格按成本加成定价）
    price = cost * (1 + m)
    if price <= 0:  # 售价为负时，收益设为极小值（优化时自动排除）
        return -np.inf, price, 0, 0

    # 2. 基于最优模型计算基础需求量（仅由售价决定）
    model = best_model_params.get(cat, {})
    if model.get("type") == "none":  # 模型无效时用默认需求
        base_demand = 10
    else:
        x = price  # 自变量为售价
        if model["type"] == "linear":
            base_demand = model["a"] + model["b"] * x
        elif model["type"] == "power":
            base_demand = model["a"] * (x ** (-model["b"]))
        elif model["type"] == "logistic":
            base_demand = logistic_model(x, model["K"], model["a"], model["b"])
        elif model["type"] == "exponential":  # 原"指数函数模型"改为显式判断
            base_demand = model["a"] * np.exp(-model["b"] * x)
        else:  # 指数函数模型
            base_demand = model["a"] * np.exp(-model["b"] * x)
        base_demand = max(base_demand, 1)  # 避免需求量为0或负数

    # 3. 融入时间特征（题干强调“需求量与时间相关”）
    # 特征1：是否周末（周末需求更高，系数1.2；工作日1.0）
    is_weekend = 1 if date.weekday() in [5, 6] else 0  # 5=周六，6=周日
    weekend_coeff = 1.2 if is_weekend else 1.0
    # 特征2：是否7月（预测周期为7月，需求旺季，系数1.1）
    july_coeff = 1.1 if date.month == 7 else 1.0
    # 最终市场需求量 = 基础需求量 × 时间特征系数
    market_demand = base_demand * weekend_coeff * july_coeff

    # 4. 计算实际可售量（补货量扣除损耗）
    available_sales = R * (1 - loss_rate)
    # 5. 计算实际销量（取“可售量”和“需求量”的最小值，避免滞销/缺货）
    actual_sales = min(available_sales, market_demand)
    # 6. 计算净收益（严格贴合题干公式）
    net_profit = actual_sales * price - R * cost

    return net_profit, price, actual_sales, market_demand


# ---------------------- 6. 单日补货与定价优化（收益最大化） ----------------------
def optimize_daily_decision(cat, date, pred_cost, best_model_params):
    """优化单个品类单日的“成本加成率(m)+补货总量(R)”，最大化净收益"""
    # 提取当前品类的平均损耗率
    loss_rate = best_model_params[cat]["loss_rate"]

    # 目标函数：最小化负收益（等同于最大化收益，因为优化器默认最小化）
    def objective(x):
        m, R = x  # x[0]=加成率，x[1]=补货量
        profit, _, _, _ = calculate_profit(m, R, pred_cost, loss_rate, cat, best_model_params, date)
        return -profit

    # 约束条件（贴合商超实际运营）
    constraints = [
        {"type": "ineq", "fun": lambda x: x[0]},  # 加成率m ≥ 0（不亏本）
        {"type": "ineq", "fun": lambda x: 1 - x[0]},  # 加成率m ≤ 1（不超过100%，符合商超常规）
        {"type": "ineq", "fun": lambda x: x[1] - 2.5},  # 补货量R ≥ 2.5kg（题干问题3提到的最小陈列量，提前适配）
        {"type": "ineq", "fun": lambda x: 500 - x[1]}  # 补货量R ≤ 500kg（仓储空间限制）
    ]

    # 初始值：基于历史数据的合理值（减少优化迭代次数）
    init_m = 0.3  # 默认初始加成率30%（商超常见加成范围）
    # 初始补货量=该品类历史平均销量（从最优模型反推合理值）
    init_price = pred_cost * (1 + init_m)
    if best_model_params[cat]["type"] == "linear":
        init_R = best_model_params[cat]["a"] + best_model_params[cat]["b"] * init_price
    elif best_model_params[cat]["type"] == "power":
        init_R = best_model_params[cat]["a"] * (init_price ** (-best_model_params[cat]["b"]))
    elif best_model_params[cat]["type"] == "logistic":
        init_R = logistic_model(init_price, best_model_params[cat]["K"],
                                best_model_params[cat]["a"], best_model_params[cat]["b"])
    else:  # 指数函数模型
        init_R = best_model_params[cat]["a"] * np.exp(-best_model_params[cat]["b"] * init_price)

    init_R = max(init_R, 5)  # 初始补货量不低于5kg

    # 优化求解（使用SLSQP方法，适合带约束的非线性优化）
    result = minimize(
        objective,
        x0=[init_m, init_R],
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "disp": False}
    )

    # 提取优化结果（成功则用最优值，失败则用默认值）
    if result.success:
        opt_m, opt_R = result.x
        opt_profit, opt_price, opt_sales, opt_demand = calculate_profit(
            opt_m, opt_R, pred_cost, loss_rate, cat, best_model_params, date
        )
        return {
            "日期": date,
            "品类名称": cat,
            "预测日成本(元/千克)": round(pred_cost, 2),
            "成本加成率": round(opt_m, 3),
            "定价(元/千克)": round(opt_price, 2),
            "补货总量(千克)": round(opt_R, 2),
            "预计市场需求量(千克)": round(opt_demand, 2),
            "预计实际销量(千克)": round(opt_sales, 2),
            "品类平均损耗率": round(loss_rate, 4),
            "预计净收益(元)": round(opt_profit, 2),
            "优化状态": "成功",
            "最优需求模型": best_model_params[cat]["type"]
        }
    else:
        # 优化失败时用默认值（确保程序鲁棒性）
        default_m = 0.3
        default_R = 50
        default_price = pred_cost * (1 + default_m)
        default_profit, _, default_sales, default_demand = calculate_profit(
            default_m, default_R, pred_cost, loss_rate, cat, best_model_params, date
        )
        return {
            "日期": date,
            "品类名称": cat,
            "预测日成本(元/千克)": round(pred_cost, 2),
            "成本加成率": default_m,
            "定价(元/千克)": round(default_price, 2),
            "补货总量(千克)": default_R,
            "预计市场需求量(千克)": round(default_demand, 2),
            "预计实际销量(千克)": round(default_sales, 2),
            "品类平均损耗率": round(loss_rate, 4),
            "预计净收益(元)": round(default_profit, 2),
            "优化状态": f"失败（{result.message[:20]}）",
            "最优需求模型": best_model_params[cat]["type"]
        }


# ---------------------- 7. 生成未来7天（2023-07-01至07）的决策表 ----------------------
def generate_future_decisions(cat_daily_data, best_model_params, future_dates):
    """预测未来7天各品类的日成本，并生成每日补货+定价决策"""
    # 步骤1：预测未来7天各品类的日成本（用最近14天移动平均，贴合题干“凌晨进货未知价”场景）
    cost_pred_dict = {}
    for cat in cat_daily_data["分类名称"].unique():
        # 提取当前品类的历史成本数据
        cat_cost_history = cat_daily_data[cat_daily_data["分类名称"] == cat][
            ["销售日期", "日成本_元每千克"]
        ].sort_values("销售日期")  # 按日期排序
        # 改进后（线性趋势预测）：
        from sklearn.linear_model import LinearRegression
        def trend_prediction(series):
            series = series.dropna().reset_index(drop=True)
            if len(series) < 7:
                return series.mean()
            x = np.arange(len(series)).reshape(-1, 1)
            model = LinearRegression().fit(x, series)
            # 预测未来1天（即当前要预测的日期）
            pred = model.predict([[len(series)]])[0]
            return max(pred, series.min())  # 避免预测值低于历史最低成本

        avg_cost = trend_prediction(cat_cost_history["日成本_元每千克"])
        cost_pred_dict[cat] = round(avg_cost, 2)  # 修复变量名错误（pred_cost→avg_cost）

    # 步骤2：逐日期、逐品类优化决策
    decision_list = []
    for date in future_dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n正在优化{date_str}的补货定价决策...")
        for cat in cost_pred_dict:
            pred_cost = cost_pred_dict[cat]
            # 调用优化函数，获取当日决策
            daily_decision = optimize_daily_decision(cat, date, pred_cost, best_model_params)
            decision_list.append(daily_decision)

    # 步骤3：整理决策结果为DataFrame，并调整列顺序（便于阅读）
    decision_df = pd.DataFrame(decision_list)
    col_order = [
        "日期", "品类名称", "预测日成本(元/千克)", "成本加成率", "定价(元/千克)",
        "补货总量(千克)", "品类平均损耗率", "预计市场需求量(千克)", "预计实际销量(千克)",
        "预计净收益(元)", "最优需求模型", "优化状态"
    ]
    decision_df = decision_df[col_order]

    # 步骤4：保存决策表到C盘指定路径（带日期，便于识别）
    excel_path = f"{BASE_PATH}/20230701-07_蔬菜品类补货定价决策表.xlsx"
    decision_df.to_excel(excel_path, index=False)
    print(f"\n✅ 未来7天决策表已保存：{excel_path}")

    # 步骤5：输出前3天决策摘要（控制台快速查看）
    print("\n" + "=" * 80)
    print("2023年7月1-3日补货定价决策摘要（完整7天见Excel）")
    print("=" * 80)
    for date in future_dates[:3]:
        date_str = date.strftime("%Y-%m-%d")
        date_decision = decision_df[decision_df["日期"] == date]
        print(f"\n【{date_str}】")
        for _, row in date_decision.iterrows():
            print(
                f"品类：{row['品类名称']:8s} | 定价：{row['定价(元/千克)']:5.2f}元 | "
                f"补货量：{row['补货总量(千克)']:6.2f}kg | 预计收益：{row['预计净收益(元)']:6.2f}元 | "
                f"模型：{row['最优需求模型']:6s}"
            )

    return decision_df


# ---------------------- 8. 主程序（串联所有流程） ----------------------
if __name__ == "__main__":
    print("=" * 80)
    print("2023国赛C题问题2：蔬菜品类自动定价与补货决策")
    print("=" * 80)

    # 步骤1：加载4个附件数据（核心：附件4读取工作表“附件4”）
    df1, df2, df3, cat_loss = load_data()

    # 步骤2：生成每日品类数据（销量、成本、加成率）
    cat_daily_data = generate_cat_daily_data(df1, df2, df3)

    # 步骤3：分析销售总量与售价的非线性关系，获取最优模型参数
    best_model_params, relation_df = analyze_sales_margin_relation(cat_daily_data, cat_loss)

    # 步骤4：生成未来7天（2023-07-01至07）的补货+定价决策
    decision_df = generate_future_decisions(cat_daily_data, best_model_params, FUTURE_DATES)

    print(f"\n✅ 所有流程执行完成！所有结果文件已保存至：{BASE_PATH}")
    print(f"📊 生成文件清单：")
    print(f"1. 各品类关系图：{BASE_PATH}/*_销售总量-售价关系图.png")
    print(f"2. 关系分析表：{BASE_PATH}/各品类销售总量-售价关系分析表.xlsx")
    print(f"3. 未来7天决策表：{BASE_PATH}/20230701-07_蔬菜品类补货定价决策表.xlsx")
    print("=" * 80)