import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 1. 强制中文支持（核心配置） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 全局设置中文为黑体
plt.rcParams['axes.unicode_minus'] = False    # 修复负号显示异常

# ---------------------- 2. 文件路径配置 ----------------------
CATEGORY_FILE = r"C:\Users\21165\Desktop\23数模C题\附件1.xlsx"
SALES_FILE = r"C:\Users\21165\Desktop\23数模C题\附件2.xlsx"
OUTPUT_IMAGE = r"C:\Users\21165\Desktop\23数模C题\蔬菜大类相关性热力图.png"

# ---------------------- 3. 数据加载与清洗 ----------------------
# 加载品类数据（单品→分类映射）
category_df = pd.read_excel(CATEGORY_FILE)
category_df['单品编码'] = category_df['单品编码'].astype(str)  # 统一编码格式

# 加载销售数据（合并日期时间）
sales_df = pd.read_excel(
    SALES_FILE,
    parse_dates=[['销售日期', '扫码销售时间']],
    dtype={'单品编码': 'str'}
)
sales_df.rename(columns={'销售日期_扫码销售时间': 'datetime'}, inplace=True)

# ---------------------- 4. 季度编码处理 ----------------------
def get_quarter_code(month):
    return 1 if 3<=month<=5 else 2 if 6<=month<=8 else 3 if 9<=month<=11 else 4

sales_df['年季'] = sales_df['datetime'].apply(lambda dt: f"{dt.year}-{get_quarter_code(dt.month)}")

# ---------------------- 5. 品类-季度销量聚合 ----------------------
merged_df = sales_df.merge(
    category_df[['单品编码', '分类名称']],
    on='单品编码',
    how='left'
)

category_quarter_sales = merged_df.groupby(
    ['分类名称', '年季']
)['销量(千克)'].sum().reset_index()

# ---------------------- 6. 宽表转换 ----------------------
pivot_df = category_quarter_sales.pivot(
    index='年季',
    columns='分类名称',
    values='销量(千克)'
).fillna(0)

# ---------------------- 7. 相关性计算 ----------------------
corr_matrix = pivot_df.corr()

# ---------------------- 8. 可视化（强制中文黑体） ----------------------
plt.figure(figsize=(12, 10), dpi=120)
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True
)

# 以下文本元素已通过全局配置强制使用黑体
plt.title("蔬菜大类季度销量相关性热力图", fontsize=16, pad=20)
plt.xlabel("蔬菜大类", fontsize=14)              # X轴标签
plt.ylabel("蔬菜大类", fontsize=14)              # Y轴标签
plt.xticks(rotation=45, ha='right', fontsize=12) # X轴类别名称
plt.yticks(rotation=0, fontsize=12)              # Y轴类别名称

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
plt.show()