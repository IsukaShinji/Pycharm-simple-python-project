import os
import re
import jieba
import numpy as np
import msgpack
from collections import defaultdict, Counter
from urllib.parse import unquote


def load_stopwords():
    sw = {"the", "a", "an", "in", "on", "at", "of", "to"}
    try:
        with open("stopwords.txt", 'r', encoding='utf-8') as f:
            sw.update({line.strip() for line in f if line.strip()})
    except:
        pass
    return sw


def restore_url(fn):
    if fn.endswith('.html'):
        parts = fn[:-5].rsplit('_', 1)
        if len(parts) == 2:
            return unquote(parts[0]), parts[1]
    return None, None


def process_html(fp, sw):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            cnt = re.sub(r'<[^>]+>|\s+', ' ', f.read()).strip()
        words = [w for w in jieba.cut(cnt) if w.strip() and w not in sw]
        return words, len(cnt)  # 返回词列表和文本长度
    except:
        return [], 0


def build_index(html_dir, sw):
    idx = defaultdict(list)
    url_map = {}
    doc_lengths = {}  # 存储文档长度
    files = [f for f in os.listdir(html_dir) if f.endswith('.html')]

    for f in files:
        url, docid = restore_url(f)
        if not url or not docid:
            continue
        url_map[docid] = url

        words, length = process_html(os.path.join(html_dir, f), sw)
        doc_lengths[docid] = length  # 保存文档长度

        word_counts = Counter(words)
        for w, tf in word_counts.items():
            if idx[w] and idx[w][-1][0] == docid:
                idx[w][-1] = (docid, idx[w][-1][1] + tf)
            else:
                idx[w].append((docid, tf))

    # 保存文档长度
    length_path = os.path.join(r"C:\Users\21165\Desktop\编程集训", "document_lengths.txt")
    with open(length_path, 'w', encoding='utf-8') as f:
        for docid, length in doc_lengths.items():
            f.write(f"{docid}\t{length}\n")

    return idx, url_map, len(files), doc_lengths


def load_day3_data():
    idx_txt = r"C:\Users\21165\Desktop\编程集训\inverted_index_dict.txt"
    url_txt = r"C:\Users\21165\Desktop\编程集训\docid_to_url_map.txt"
    length_txt = r"C:\Users\21165\Desktop\编程集训\document_lengths.txt"
    idx_mp = idx_txt.replace(".txt", ".mp")

    if os.path.exists(idx_mp) and os.path.getmtime(idx_mp) >= os.path.getmtime(idx_txt):
        try:
            with open(idx_mp, 'rb') as f:
                return msgpack.load(f, raw=False)
        except:
            pass

    if not (os.path.exists(idx_txt) and os.path.exists(url_txt) and os.path.exists(length_txt)):
        return None

    try:
        url_map = {}
        with open(url_txt, 'r', encoding='utf-8') as f:
            url_map = dict(line.strip().split('\t', 1) for line in f if line.strip() and '\t' in line)

        doc_lengths = {}
        with open(length_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    docid, length = line.split('\t', 1)
                    try:
                        doc_lengths[docid] = int(length)
                    except ValueError:
                        continue

        N = len(url_map)
        idx = defaultdict(list)
        with open(idx_txt, 'r', encoding='utf-8') as f:
            content = f.read().split('\n')

        split_re = re.compile(r'\||:')
        for line in content:
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = split_re.split(line)
            if len(parts) < 3:
                continue
            word = parts[0]
            idx[word].extend(
                (parts[i], int(parts[i + 1]))
                for i in range(1, len(parts), 2)
                if i + 1 < len(parts)
            )

        with open(idx_mp, 'wb') as f:
            msgpack.dump((idx, url_map, N, doc_lengths), f)
        return idx, url_map, N, doc_lengths
    except:
        return None


def bm25_score(tf, doc_length, avg_length, df, N, k1=1.2, b=0.75):
    """BM25评分公式实现"""
    if df == 0:
        idf = 0
    else:
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
    denominator = tf + k1 * (1 - b + b * (doc_length / avg_length))
    return idf * (tf * (k1 + 1)) / denominator if denominator != 0 else 0


def search(idx, url_map, N, doc_lengths, query, k=20, sw=None):
    """使用BM25算法进行检索"""
    if N == 0:
        return []

    # 计算平均文档长度
    total_length = sum(doc_lengths.values()) if doc_lengths else 0
    avg_length = total_length / N if N > 0 else 0

    # 处理查询
    q_terms = [w for w in jieba.cut(query) if w not in (sw or set())]
    if not q_terms:
        return []

    # 计算每个词的文档频率
    df = {term: len(idx.get(term, [])) for term in q_terms}

    # 收集相关文档并计算分数
    scores = defaultdict(float)
    term_counts = defaultdict(lambda: defaultdict(int))  # 记录词频

    for term in q_terms:
        for docid, tf in idx.get(term, []):
            term_counts[docid][term] = tf
            doc_len = doc_lengths.get(docid, avg_length)
            doc_df = df.get(term, 0)
            scores[docid] += bm25_score(tf, doc_len, avg_length, doc_df, N)

    # 获取前20个结果
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:k]

    # 补充词频信息
    final_results = []
    for docid, score in sorted_results:
        if docid not in url_map:
            continue
        total_tf = sum(term_counts[docid].values())
        target_tf = sum(term_counts[docid].get(term, 0) for term in q_terms)
        final_results.append((docid, score, total_tf, target_tf))

    return final_results


def main():
    html_dir = r"C:\Users\21165\Desktop\编程集训\crawled_htm_files"
    sw = load_stopwords()
    data = load_day3_data()

    if data:
        idx, url_map, N, doc_lengths = data
        print(f"加载完成：{len(idx)}个词条，{len(url_map)}篇文档")
    else:
        print("重建索引...")
        idx, url_map, N, doc_lengths = build_index(html_dir, sw)

    print("\n查询系统（输入'quit'退出）")
    while True:
        q = input("查询: ").strip()
        if q.lower() == 'quit':
            break
        if not q:
            continue

        results = search(idx, url_map, N, doc_lengths, q, sw=sw)
        print(f"\n查询结果（共{len(results)}条）：")

        for i, (docid, score, total_tf, target_tf) in enumerate(results, 1):
            print(f"[{i}] BM25分数: {score:.4f} | 总词频: {total_tf} | 目标词频: {target_tf}")
            print(f"   {url_map[docid]}（{docid}）\n")


if __name__ == "__main__":
    main()