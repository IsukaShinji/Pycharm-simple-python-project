import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import Counter

# ---------------------- 1. 配置与全局设置 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ---------------------- 2. 数据加载（基于C题.pdf描述的附件结构） ----------------------
def load_data():
    """根据C题.pdf提及的附件结构加载数据：附件1含品类信息，附件2含销售流水"""
    # 加载附件1：6个蔬菜品类的商品信息（含单品编码、单品名称、分类名称）
    category_df = pd.read_excel(r"C:\Users\21165\Desktop\23数模C题\附件1.xlsx")
    category_df['单品编码'] = category_df['单品编码'].astype(str)  # 统一编码为字符串

    # 加载附件2：2020.7.1-2023.6.30的销售流水明细
    sales_df = pd.read_excel(
        r"C:\Users\21165\Desktop\23数模C题\附件2.xlsx"
        ,
        parse_dates=[['销售日期', '扫码销售时间']],  # 合并日期时间
        dtype={'单品编码': 'str', '销量(千克)': 'float32'}
    )
    sales_df.rename(columns={'销售日期_扫码销售时间': 'datetime'}, inplace=True)

    return category_df, sales_df


category_df, sales_df = load_data()

# 建立映射关系
item_name_map = category_df.set_index('单品编码')['单品名称'].to_dict()
item_category_map = category_df.set_index('单品编码')['分类名称'].to_dict()
all_categories = category_df['分类名称'].unique().tolist()  # 提取6个蔬菜品类


# ---------------------- 3. 季度划分（按C题分析需求自定义） ----------------------
def get_quarter_code(month):
    """季度划分：3-5月（春）、6-8月（夏）、9-11月（秋）、12-2月（冬）"""
    if 3 <= month <= 5:
        return 1
    elif 6 <= month <= 8:
        return 2
    elif 9 <= month <= 11:
        return 3
    else:
        return 4


sales_df['年季'] = sales_df['datetime'].apply(
    lambda dt: f"{dt.year}-{get_quarter_code(dt.month)}"
)

# ---------------------- 4. 数据聚合（单品×季度销量） ----------------------
item_quarter_sales = sales_df.groupby(['单品编码', '年季'])['销量(千克)'].sum().reset_index()
pivot_df = item_quarter_sales.pivot(
    index='单品编码',
    columns='年季',
    values='销量(千克)'
).fillna(0)

# 过滤无销售记录的单品
valid_pivot = pivot_df.loc[pivot_df.sum(axis=1) != 0]

# ---------------------- 5. 绘制趋势图 ----------------------
n_items = len(valid_pivot)
color_palette = sns.color_palette("husl", n_items)
cmap = ListedColormap(color_palette)

plt.figure(figsize=(18, 12), dpi=120)
for idx, code in enumerate(valid_pivot.index):
    name = item_name_map.get(code, f"单品_{code}")
    cat = item_category_map.get(code, "未知品类")
    plt.plot(
        valid_pivot.columns,
        valid_pivot.loc[code],
        label=f"{name}（{cat}）",
        color=cmap(idx),
        alpha=0.7,
        linewidth=1.2,
        marker='o',
        markersize=4
    )

plt.title("蔬菜单品季度销量趋势（基于C题数据）", fontsize=16, pad=30)
plt.xlabel("季度（年-季度码）", fontsize=14)
plt.ylabel("销量（千克）", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(0.5, -0.4), loc='upper center', ncol=6, fontsize=9, frameon=False)
plt.subplots_adjust(bottom=0.3)
plt.savefig(r"C:\Users\21165\Desktop\23数模C题\单品季度销量趋势.png", bbox_inches='tight')
plt.show()


# ---------------------- 6. 季节性分析（含大类内占比，基于C题分析需求） ----------------------
def analyze_seasonality(pivot_df, name_map, cat_map, categories, cv_threshold=0.7):
    """分析季节性强/弱单品及其在所属大类中的占比（符合C题对品类关系的分析要求）"""
    # 计算变异系数（CV）
    cv_series = pivot_df.apply(lambda row: row.std() / row.mean(), axis=1)
    strong_codes = cv_series[cv_series > cv_threshold].index
    weak_codes = cv_series[cv_series <= cv_threshold].index

    # 统计各品类的单品总数、强/弱季节性数量
    cat_stats = {cat: {'total': 0, 'strong': 0, 'weak': 0} for cat in categories}
    for code in pivot_df.index:
        cat = cat_map.get(code, None)
        if cat in cat_stats:
            cat_stats[cat]['total'] += 1
    for code in strong_codes:
        cat = cat_map.get(code, None)
        if cat in cat_stats:
            cat_stats[cat]['strong'] += 1
    for code in weak_codes:
        cat = cat_map.get(code, None)
        if cat in cat_stats:
            cat_stats[cat]['weak'] += 1

    # 生成品类占比总结
    cat_summary = ""
    for cat in categories:
        total = cat_stats[cat]['total']
        if total == 0:
            continue
        strong_pct = (cat_stats[cat]['strong'] / total) * 100
        weak_pct = (cat_stats[cat]['weak'] / total) * 100
        cat_summary += (
            f"  • {cat}：\n"
            f"    - 总单品数：{total}个\n"
            f"    - 强季节性占比：{strong_pct:.2f}%（{cat_stats[cat]['strong']}个）\n"
            f"    - 弱季节性占比：{weak_pct:.2f}%（{cat_stats[cat]['weak']}个）\n"
        )

    # 全量输出弱季节性单品
    weak_names = [name_map.get(code, f"单品_{code}") for code in weak_codes]
    weak_summary = "\n".join(weak_names)

    # 输出结果
    print(f"""
=====================================
C题蔬菜单品季节性分析报告（基于2020-2023年销售数据）：

1. 各品类季节性占比分布：
{cat_summary}

2. 强季节性单品（共{len(strong_codes)}个，CV > {cv_threshold}）：
   季度销量波动显著，符合季节规律。

3. 弱季节性单品（共{len(weak_codes)}个，CV ≤ {cv_threshold}）：
   年度销量平稳，季节差异小。
   完整清单：
{weak_summary}
=====================================
    """)

# 执行分析（严格高阈值，符合高门槛要求）
analyze_seasonality(valid_pivot, item_name_map, item_category_map, all_categories, cv_threshold=1)