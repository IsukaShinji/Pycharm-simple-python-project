import json
from collections import defaultdict
import jieba

INDEX_DIR = "./index"
PROCESSED_DIR = "./processed_data"
STOPWORDS = {"的", "是", "在", "和", "及", "与", "等", "为", "了", "有", "这", "那", "个", "件"}

# 加载预构建的索引和文档
with open(f"{INDEX_DIR}/inverted_index.json", "r", encoding="UTF-8") as f:
    inverted_index = json.load(f)
with open(f"{PROCESSED_DIR}/processed_docs.json", "r", encoding="UTF-8") as f:
    processed_docs = json.load(f)


def tokenize_query(query):
    """查询分词（与文档预处理一致，保证检索准确性）"""
    return [token.lower() for token in jieba.lcut(query) if token not in STOPWORDS and len(token) > 1]


def basic_retrieval(query):
    """基本检索：返回含所有关键词的文档（按词频排序，对应PDF ）"""
    query_tokens = tokenize_query(query)
    if not query_tokens:
        return []

    # 1. 获取每个关键词的文档集合
    token_doc_sets = []
    for token in query_tokens:
        if token not in inverted_index:
            return []  # 关键词无匹配，返回空
        token_doc_sets.append({item["doc_id"] for item in inverted_index[token]})

    # 2. 求文档交集（含所有关键词）
    matched_doc_ids = set.intersection(*token_doc_sets)
    if not matched_doc_ids:
        return []

    # 3. 按词频总和排序（相关性优先）
    doc_score = defaultdict(int)
    for doc_id in matched_doc_ids:
        total_tf = 0
        for token in query_tokens:
            for item in inverted_index[token]:
                if item["doc_id"] == doc_id:
                    total_tf += item["tf"]
                    break
        doc_score[doc_id] = total_tf

    # 4. 返回排序后的文档
    sorted_docs = sorted(matched_doc_ids, key=lambda x: doc_score[x], reverse=True)
    return [processed_docs[str(doc_id)] for doc_id in sorted_docs]