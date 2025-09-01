import json
from collections import defaultdict, Counter
import os

PROCESSED_DIR = "./processed_data"
INDEX_DIR = "./index"
os.makedirs(INDEX_DIR, exist_ok=True)


def build_inverted_index():
    """基于预处理数据构建倒排索引（对应PDF ）"""
    # 加载预处理文档
    with open(f"{PROCESSED_DIR}/processed_docs.json", "r", encoding="UTF-8") as f:
        processed_docs = json.load(f)

    inverted_index = defaultdict(list)  # 结构：{词项: [{doc_id: 123, tf: 5, positions: [0,2,4]}]}
    for doc_id_str, doc_info in processed_docs.items():
        doc_id = int(doc_id_str)
        tokens = doc_info["tokens"]

        # 计算词频（TF）和位置
        token_counter = Counter(tokens)
        token_positions = defaultdict(list)
        for idx, token in enumerate(tokens):
            token_positions[token].append(idx)

        # 填充倒排表
        for token, tf in token_counter.items():
            inverted_index[token].append({
                "doc_id": doc_id,
                "tf": tf,
                "positions": token_positions[token]
            })

    # 保存索引和词典
    with open(f"{INDEX_DIR}/inverted_index.json", "w", encoding="UTF-8") as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=2)
    with open(f"{INDEX_DIR}/vocabulary.json", "w", encoding="UTF-8") as f:
        json.dump(list(inverted_index.keys()), f, ensure_ascii=False, indent=2)

    print(f"倒排索引构建完成！词典含 {len(inverted_index)} 个词项")
    return inverted_index


if __name__ == "__main__":
    build_inverted_index()