# 一、环境准备与数据模拟（导入内置库，无需额外安装）
from collections import defaultdict

# 模拟训练数据：(关键词列表, 类别标签)，1=垃圾邮件，0=正常邮件
train_data = [
    # 垃圾邮件（label=1）
    (["免费领取", "中奖通知", "点击链接"], 1),
    (["中奖号码", "限时优惠", "汇款验证"], 1),
    (["免费奖品", "点击链接", "限时活动"], 1),
    (["中奖通知", "汇款账号", "免费参与"], 1),
    (["限时折扣", "免费会员", "点击链接"], 1),
    (["中奖名单", "汇款确认", "免费领取"], 1),
    (["免费试用", "点击链接", "中奖机会"], 1),
    (["限时抢购", "中奖信息", "汇款要求"], 1),
    (["免费礼品", "中奖通知", "点击链接"], 1),
    (["中奖通知", "限时活动", "汇款地址"], 1),
    # 正常邮件（label=0）
    (["会议通知", "项目进度", "明天见面"], 0),
    (["快递提醒", "取件码", "门口驿站"], 0),
    (["工作汇报", "截止日期", "需要修改"], 0),
    (["朋友聚会", "时间地点", "不见不散"], 0),
    (["账单提醒", "本月消费", "支付链接"], 0),
    (["面试通知", "时间地点", "携带简历"], 0),
    (["家庭聚餐", "周末时间", "准备食材"], 0),
    (["项目文档", "需要审核", "请查收"], 0),
    (["快递延误", "抱歉通知", "重新派送"], 0),
    (["生日祝福", "明天晚上", "一起吃饭"], 0),
]

# 打印训练数据规模
print(f"训练数据包含{len(train_data)} 封邮件（{sum(1 for _, label in train_data if label == 1)} 封垃圾邮件，{sum(1 for _, label in train_data if label == 0)} 封正常邮件）")


# 二、核心函数定义
def build_vocab(train_data):
    """
    构建词汇表（所有邮件中的唯一关键词）及词到索引的映射（提升特征转换效率）
    参数：train_data - 训练数据列表
    返回：vocab - 词汇表（列表）；word_to_idx - 词到索引的映射（字典）
    """
    vocab = set()  # 用集合自动去重
    for keywords, _ in train_data:
        vocab.update(keywords)
    vocab = list(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab)}  # O(1)查找效率
    return vocab, word_to_idx


def convert_to_feature(keywords, word_to_idx):
    """
    将关键词列表转换为二进制特征向量（1=关键词存在，0=不存在）
    参数：keywords - 邮件关键词列表；word_to_idx - 词到索引的映射
    返回：feature_vector - 二进制特征向量（列表）
    """
    feature_vector = [0] * len(word_to_idx)  # 初始化全0向量
    for word in keywords:
        if word in word_to_idx:
            feature_vector[word_to_idx[word]] = 1
    return feature_vector


def train_naive_bayes(train_data, word_to_idx):
    """
    训练朴素贝叶斯模型（计算先验概率和带拉普拉斯平滑的似然概率）
    参数：train_data - 训练数据列表；word_to_idx - 词到索引的映射
    返回：prior - 先验概率字典；likelihood - 似然概率字典
    """
    # 1. 计算先验概率 P(类别) = 类别样本数 / 总样本数
    label_counts = defaultdict(int)
    for _, label in train_data:
        label_counts[label] += 1
    total_samples = len(train_data)
    prior = {label: count / total_samples for label, count in label_counts.items()}

    # 2. 计算似然概率 P(关键词存在|类别)（伯努利模型：仅关心关键词是否出现）
    likelihood = defaultdict(dict)
    # 统计每个类别中，包含该关键词的邮件数量
    word_doc_counts = defaultdict(lambda: defaultdict(int))  # word_doc_counts[label][word] = 出现次数

    for keywords, label in train_data:
        unique_words = set(keywords)  # 同一邮件中关键词仅算1次
        for word in unique_words:
            word_doc_counts[label][word] += 1

    K = 2  # 拉普拉斯平滑参数（二元特征默认K=2）
    vocab = list(word_to_idx.keys())  # 从映射中获取词汇表

    for label in label_counts:
        total_doc = label_counts[label]  # 该类别的邮件总数
        for word in vocab:
            count = word_doc_counts[label].get(word, 0)  # 关键词在该类别中的出现次数（默认0）
            # 平滑公式：(count + 1) / (该类别总数 + K)
            likelihood[label][word] = (count + 1) / (total_doc + K)

    return prior, likelihood


def predict_email(new_keywords, word_to_idx, prior, likelihood):
    """
    预测新邮件类别（0=正常，1=垃圾）
    参数：new_keywords - 新邮件关键词列表；word_to_idx - 词到索引映射；prior - 先验概率；likelihood - 似然概率
    返回：predicted_label - 预测类别；scores - 各类别分数（似然乘积×先验）
    """
    # 转换新邮件为特征向量
    feature_vector = convert_to_feature(new_keywords, word_to_idx)
    scores = {}  # 存储各类别分数
    vocab = list(word_to_idx.keys())

    for label in prior:
        score = prior[label]  # 初始分数 = 先验概率
        for i in range(len(vocab)):
            word = vocab[i]
            x = feature_vector[i]  # 特征值（1=存在，0=不存在）
            if x == 1:
                score *= likelihood[label][word]  # 存在则乘 P(关键词|类别)
            else:
                score *= (1 - likelihood[label][word])  # 不存在则乘 P(非关键词|类别)
        scores[label] = score

    # 选择分数最高的类别作为预测结果
    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores


# 三、模型训练与预测
# 步骤1：构建词汇表及词到索引映射
vocab, word_to_idx = build_vocab(train_data)
print(f"\n词汇表大小：{len(vocab)}（包含所有唯一关键词）")
print(f"词汇表示例：{vocab[:5]}（前5个关键词）")
print(f"词到索引映射示例：{list(word_to_idx.items())[:5]}（前5个映射）")

# 步骤2：训练朴素贝叶斯模型
prior, likelihood = train_naive_bayes(train_data, word_to_idx)
print("\n先验概率（类别占比）：", prior)
print(f"似然概率示例：垃圾邮件中'中奖通知'的概率 = {likelihood[1]['中奖通知']:.4f}")
print(f"似然概率示例：正常邮件中'中奖通知'的概率 = {likelihood[0]['中奖通知']:.4f}")

# 步骤3：预测新邮件
new_email = ["中奖通知", "汇款账号", "免费领取"]  # 待预测的新邮件关键词
predicted_label, scores = predict_email(new_email, word_to_idx, prior, likelihood)

# 打印预测结果
print("\n=== 新邮件预测结果 ===")
print(f"新邮件关键词：{new_email}")
print(f"预测类别：{'垃圾邮件' if predicted_label == 1 else '正常邮件'}")
print(f"类别分数（似然乘积×先验）：{scores}")