import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

# ---------------------- 1. 配置与路径设置（基于C题.pdf） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 强制中文黑体显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
SAVE_DIR = r"C:\Users\21165\Desktop\23数模C题"
os.makedirs(SAVE_DIR, exist_ok=True)  # 确保保存目录存在

# 附件路径（严格对应C题.pdf提及的附件1和附件2）
ATTACH1_PATH = os.path.join(SAVE_DIR, "附件1.xlsx")
ATTACH2_PATH = os.path.join(SAVE_DIR, "附件2.xlsx")


# ---------------------- 2. 数据加载（仅使用C题.pdf指定附件） ----------------------
def load_data():
    """加载附件1（品类信息）和附件2（销售流水），符合C题数据范围"""
    # 附件1：6个蔬菜品类的商品信息（含单品编码、分类名称）
    category_df = pd.read_excel(ATTACH1_PATH)
    category_df['单品编码'] = category_df['单品编码'].astype(str)  # 统一编码格式

    # 附件2：2020.7.1-2023.6.30销售流水明细
    sales_df = pd.read_excel(
        ATTACH2_PATH,
        parse_dates=[['销售日期', '扫码销售时间']],  # 合并日期时间字段
        dtype={'单品编码': 'str', '销量(千克)': 'float32'}
    )
    sales_df.rename(columns={'销售日期_扫码销售时间': 'datetime'}, inplace=True)
    return category_df, sales_df


# 加载数据并建立映射关系
category_df, sales_df = load_data()
item_to_name = dict(zip(category_df['单品编码'], category_df['单品名称']))  # 单品→名称
item_to_category = dict(zip(category_df['单品编码'], category_df['分类名称']))  # 单品→品类
all_categories = category_df['分类名称'].unique().tolist()  # 提取所有品类

# ---------------------- 3. 周末特征提取（时间关联分析） ----------------------
# 标记周末（周六、日）（销售量与时间关联）
sales_df['is_weekend'] = sales_df['datetime'].dt.weekday.isin([5, 6])

# ---------------------- 4. 周末影响计算（CV阈值0.3，降低判定门槛） ----------------------
# 品类层面分析
sales_df['分类名称'] = sales_df['单品编码'].map(item_to_category)
cat_sales = sales_df.groupby(['分类名称', 'is_weekend'])['销量(千克)'].sum().unstack(fill_value=0)
cat_sales.columns = ['工作日销量', '周末销量']
cat_sales['CV'] = cat_sales.apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0, axis=1)
cat_sales['周末影响'] = ['影响大' if cv > 0.3 else '影响小' for cv in cat_sales['CV']]

# 单品层面分析
item_sales = sales_df.groupby(['单品编码', 'is_weekend'])['销量(千克)'].sum().unstack(fill_value=0)
item_sales.columns = ['工作日销量', '周末销量']
item_sales['CV'] = item_sales.apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0, axis=1)
item_sales['周末影响'] = ['影响大' if cv > 0.3 else '影响小' for cv in item_sales['CV']]
item_sales['所属品类'] = [item_to_category.get(code, '未知') for code in item_sales.index]
item_sales['单品名称'] = [item_to_name.get(code, f"未知_{code}") for code in item_sales.index]

# ---------------------- 5. 占比计算 ----------------------
# 品类层面占比
total_cats = len(all_categories)
cat_impact_count = cat_sales['周末影响'].value_counts()
cat_impact_pct = (cat_impact_count / total_cats * 100).round(1)

# 单品层面占比
total_items = len(item_sales)
item_impact_count = item_sales['周末影响'].value_counts()
item_impact_pct = (item_impact_count / total_items * 100).round(1)

# 分品类单品占比
item_cat_pct = item_sales.groupby(['所属品类', '周末影响']).size().unstack(fill_value=0)
item_cat_pct = (item_cat_pct.div(item_cat_pct.sum(axis=1), axis=0) * 100).round(1)

# ---------------------- 6. 可视化展示（两张图片保存至指定路径） ----------------------
# 图1：品类周末销量对比（图例含CV值）
plt.figure(figsize=(12, 8), dpi=120)
cat_colors = sns.color_palette("husl", len(all_categories))
for i, cat in enumerate(cat_sales.index):
    cv = round(cat_sales.loc[cat, 'CV'], 2)
    plt.plot(
        ['工作日', '周末'],
        cat_sales.loc[cat, ['工作日销量', '周末销量']],
        label=f"{cat}（CV={cv}，{cat_sales.loc[cat, '周末影响']}）",
        color=cat_colors[i],
        linewidth=2, marker='s', markersize=8
    )
plt.title("各品类工作日与周末销量对比", fontsize=16, pad=20)
plt.xlabel("时间类型", fontsize=14)
plt.ylabel("总销量（千克）", fontsize=14)
plt.grid(linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fontsize=10)
plt.subplots_adjust(bottom=0.2)
# 保存品类图
cat_img_path = os.path.join(SAVE_DIR, "品类周末性分析图.png")
plt.savefig(cat_img_path, bbox_inches='tight')
print(f"品类分析图已保存至：{cat_img_path}")
plt.show()

# 图2：所有单品周末性分析图（替换原占比图）
plt.figure(figsize=(18, 12), dpi=120)
n_items = len(item_sales)
item_colors = sns.color_palette("husl", n_items)  # 百种颜色区分单品
for i, (code, row) in enumerate(item_sales.iterrows()):
    plt.plot(
        ['工作日', '周末'],
        [row['工作日销量'], row['周末销量']],
        label=f"{row['单品名称']}（{row['所属品类']}）",
        color=item_colors[i],
        alpha=0.7, linewidth=1.2, marker='o', markersize=4
    )
plt.title("所有单品工作日与周末销量对比", fontsize=16, pad=30)
plt.xlabel("时间类型", fontsize=14)
plt.ylabel("销量（千克）", fontsize=14)
plt.grid(linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=5, fontsize=8, frameon=False)
plt.subplots_adjust(bottom=0.3)
# 保存单品图
item_img_path = os.path.join(SAVE_DIR, "所有单品周末性分析图.png")
plt.savefig(item_img_path, bbox_inches='tight')
print(f"单品分析图已保存至：{item_img_path}")
plt.show()

# ---------------------- 7. 分析总结（基于C题第一问要求） ----------------------
print("\n=====================================")
print("C题蔬菜周末影响分析总结（销售量与时间关联）")  #
print("=====================================")
print(f"1. 品类层面（共{total_cats}个品类）：")
print(f"   - 周末影响大：{cat_impact_count.get('影响大', 0)}个，占比{cat_impact_pct.get('影响大', 0)}%")
print(f"   - 周末影响小：{cat_impact_count.get('影响小', 0)}个，占比{cat_impact_pct.get('影响小', 0)}%")

print(f"\n2. 单品层面（共{total_items}个单品）：")
print(f"   - 整体周末影响大：{item_impact_count.get('影响大', 0)}个，占比{item_impact_pct.get('影响大', 0)}%")
print(f"   - 整体周末影响小：{item_impact_count.get('影响小', 0)}个，占比{item_impact_pct.get('影响小', 0)}%")

print("\n3. 各品类内单品影响占比：")
for cat in all_categories:
    if cat in item_cat_pct.index:
        print(f"   - {cat}：影响大{item_cat_pct.loc[cat, '影响大']}%，影响小{item_cat_pct.loc[cat, '影响小']}%")
print("=====================================")