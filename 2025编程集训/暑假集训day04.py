import os
import re
import math
import msgpack
import jieba
from collections import defaultdict, Counter
from urllib.parse import unquote


# 1. 对齐day03的停用词加载逻辑
def load_stopwords(stopwords_path="stopwords.txt"):
    stopwords = set()
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        english_stopwords = {"the", "a", "an", "in", "on", "at", "of", "to"}
        stopwords.update(english_stopwords)
    except FileNotFoundError:
        print("⚠️ 停用词文件不存在 - 仅用默认英文停用词")
        stopwords = {"the", "a", "an", "in", "on", "at", "of", "to"}
    return stopwords


# 2. 对齐day03的URL恢复逻辑
def restore_url_from_filename(filename):
    if not filename.endswith('.html'):
        return None, None
    name_part = filename[:-5]
    parts = name_part.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    encoded_url, docid = parts[0], parts[1]
    try:
        original_url = unquote(encoded_url)
        return original_url, docid
    except Exception as e:
        print(f"URL解码失败：{e}（文件名：{filename}）")
        return None, None


# 3. 对齐day03的HTML处理逻辑（增加标签过滤）
def process_html(html_path, stopwords):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            original_html = f.read()

        filename = os.path.basename(html_path)
        docid = filename.rsplit('_', 1)[-1].replace('.html', '') if '_' in filename else 'unknown'
        problematic_docids = {'2516', '909'}  # 保留day03的问题文档处理

        # 调试文档保存（同day03）
        if docid in problematic_docids:
            debug_dir = r"C:\Users\21165\Desktop\编程集训\debug_texts"
            os.makedirs(debug_dir, exist_ok=True)
            original_html_path = os.path.join(debug_dir, f"{docid}_original.html")
            with open(original_html_path, 'w', encoding='utf-8') as f:
                f.write(original_html)
            print(f"📋 已保存原始HTML: {original_html_path}")

        # 增强文本提取（同day03的标签过滤）
        content = original_html
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # 过滤导航/页脚等冗余标签
        special_patterns = [
            r'<nav.*?</nav>', r'<header.*?</header>', r'<footer.*?</footer>',
            r'<aside.*?</aside>', r'<div class="nav.*?</div>',
            r'<div class="menu.*?</div>', r'<div class="ad.*?</div>',
            r'<div class="footer.*?</div>', r'<div class="sidebar.*?</div>'
        ]
        for pattern in special_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

        # 提取纯文本
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        text_length = len(text)

        # 调试文本保存
        if docid in problematic_docids:
            debug_text_path = os.path.join(debug_dir, f"{docid}_extracted.txt")
            with open(debug_text_path, 'w', encoding='utf-8') as f:
                f.write(f"文档编号: {docid}\n")
                f.write(f"原始文件名: {filename}\n")
                f.write(f"提取文本长度: {text_length} 字符\n\n")
                f.write(text)
            print(f"📋 已保存提取文本: {debug_text_path}")

        # 分词与过滤（同day03的过滤逻辑）
        words = jieba.cut(text, cut_all=False)
        filtered_words = [
            word for word in words
            if word.strip()
               and word.strip() not in stopwords
               and not word.strip().isdigit()
               and not (word.strip() in ',.!?;:"\'()[]{}、，。！？；：“”‘’（）【】{}')
        ]

        return filtered_words, text_length
    except Exception as e:
        print(f"\n⚠️ 处理{os.path.basename(html_path)}异常：{e} - 跳过")
        return [], 0


# 4. 保留day04的索引构建优化，对齐day03的核心流程
def build_index(html_dir, sw):
    idx = defaultdict(list)
    url_map = {}
    doc_lengths = {}
    files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    total_files = len(files)
    processed = 0
    progress_records = []
    print(f"📂 批量处理：总文件数={total_files}，每100个打印进度")

    for f in files:
        url, docid = restore_url_from_filename(f)  # 使用day03的URL恢复函数
        if not url or not docid:
            processed += 1
            continue
        url_map[docid] = url

        words, length = process_html(os.path.join(html_dir, f), sw)  # 使用增强的HTML处理
        doc_lengths[docid] = length

        word_counts = Counter(words)
        for w, tf in word_counts.items():
            if idx[w] and idx[w][-1][0] == docid:
                idx[w][-1] = (docid, idx[w][-1][1] + tf)
            else:
                idx[w].append((docid, tf))

        # 进度显示（同day03）
        processed += 1
        if processed % 100 == 0 or processed == total_files:
            progress_text = (
                f"[{processed}/{total_files}] 处理中（完成度：{processed / total_files * 100:.1f}%）"
                if processed < total_files
                else f"[{processed}/{total_files}] 处理完成（100%）"
            )
            progress_records.append(progress_text)
            if len(progress_records) > 15:
                progress_records.pop(0)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("📊 最近进度：")
            print("\n".join(progress_records))

    # 保存文档长度
    length_path = os.path.join(r"C:\Users\21165\Desktop\编程集训", "document_lengths.txt")
    with open(length_path, 'w', encoding='utf-8') as f:
        for docid, length in doc_lengths.items():
            f.write(f"{docid}\t{length}\n")
    print(f"✅ 文档长度已保存到：{length_path}")

    return idx, url_map, len(files), doc_lengths


# 5. 保留day04的数据加载优化
def load_day3_data():
    idx_txt = r"C:\Users\21165\Desktop\编程集训\inverted_index_dict.txt"
    url_txt = r"C:\Users\21165\Desktop\编程集训\docid_to_url_map.txt"
    length_txt = r"C:\Users\21165\Desktop\编程集训\document_lengths.txt"
    idx_mp = idx_txt.replace(".txt", ".mp")

    global _CACHED_DATA
    try:
        return _CACHED_DATA
    except NameError:
        _CACHED_DATA = None

    if _CACHED_DATA is None:
        if os.path.exists(idx_mp) and os.path.getmtime(idx_mp) >= os.path.getmtime(idx_txt):
            try:
                with open(idx_mp, 'rb') as f:
                    data = msgpack.load(f, raw=False)
                    if len(data) == 5:
                        _CACHED_DATA = data
                        return _CACHED_DATA
            except:
                pass

        if not (os.path.exists(idx_txt) and os.path.exists(url_txt) and os.path.exists(length_txt)):
            _CACHED_DATA = (defaultdict(list), {}, 0, {}, 0.0)
            return _CACHED_DATA

        try:
            url_map = {}
            with open(url_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        docid, url = line.split('\t', 1)
                        url_map[docid] = url
            N = len(url_map)

            doc_lengths = {}
            total_length = 0
            with open(length_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        docid, length_str = line.split('\t', 1)
                        try:
                            length = int(length_str)
                            doc_lengths[docid] = length
                            total_length += length
                        except:
                            continue
            avg_length = total_length / N if N > 0 else 0.0

            idx = defaultdict(list)
            split_re = re.compile(r'\||:')
            with open(idx_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '|' not in line:
                        continue
                    parts = split_re.split(line)
                    if len(parts) < 3:
                        continue
                    word = parts[0]
                    entries = []
                    for i in range(1, len(parts), 2):
                        if i + 1 >= len(parts):
                            break
                        entries.append((parts[i], int(parts[i+1])))
                    idx[word].extend(entries)

            _CACHED_DATA = (idx, url_map, N, doc_lengths, avg_length)
            try:
                with open(idx_mp, 'wb') as f:
                    msgpack.dump(_CACHED_DATA, f)
            except:
                pass
        except:
            _CACHED_DATA = (defaultdict(list), {}, 0, {}, 0.0)
    return _CACHED_DATA


# 6. 对齐day03的BM25参数（b=0.75）
def bm25_score(tf, doc_length, avg_length, df, N, k1=1.2, b=0.75):
    if df == 0:
        return 0.0
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    denominator = tf + k1 * (1 - b + b * (doc_length / avg_length))
    return idf * (tf * (k1 + 1)) / denominator if denominator != 0 else 0.0


# 7. 优化搜索逻辑，返回day03格式的结果
def search(idx, url_map, N, doc_lengths, avg_length, query, k=20, sw=None):
    if N == 0:
        return []

    q_terms = [w for w in jieba.lcut(query) if w.strip() and w not in (sw or set())]
    if not q_terms:
        return []

    df = {term: len(idx.get(term, [])) for term in q_terms}
    scores = defaultdict(float)
    term_counts = defaultdict(dict)  # 记录每个文档的词频信息

    for term in q_terms:
        doc_list = idx.get(term, [])
        current_df = df[term]
        if current_df == 0:
            continue
        idf = math.log((N - current_df + 0.5) / (current_df + 0.5) + 1) if N > 0 else 0
        for docid, tf in doc_list:
            term_counts[docid][term] = tf  # 记录词频
            doc_len = doc_lengths.get(docid, avg_length)
            denominator = tf + 1.2 * (1 - 0.75 + 0.75 * (doc_len / avg_length)) if avg_length != 0 else tf + 1.2
            if denominator == 0:
                continue
            score_add = idf * (tf * 2.2) / denominator
            scores[docid] += score_add

    # 整理结果格式（同day03）
    sorted_results = []
    for docid, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]:
        total_tf = sum(term_counts[docid].values())
        target_tf = sum(term_counts[docid].get(term, 0) for term in q_terms)
        sorted_results.append((docid, score, total_tf, target_tf))
    return sorted_results


# 预初始化jieba
jieba.initialize()


# 8. 对齐day03的主函数逻辑
def main():
    print("============================================================")
    print("📅 暑假集训day04 - 基于day03核心逻辑优化版")
    print("============================================================")

    html_dir = r"C:\Users\21165\Desktop\编程集训\crawled_htm_files"
    sw = load_stopwords()
    print("✅ 停用词表加载完成")

    print("\n【2/3】检查已有数据...")
    idx, url_map, N, doc_lengths, avg_length = load_day3_data()  # 补充avg_length
    if len(url_map) > 0:
        print(f"✅ 检测到已有数据：{len(idx)}个词条，{len(url_map)}篇文档")
    else:
        print("\n【3/3】无已有数据，开始重建索引...")
        idx, url_map, N, doc_lengths = build_index(html_dir, sw)
        # 重新计算平均长度（因为build_index不返回avg_length）
        total_length = sum(doc_lengths.values())
        avg_length = total_length / N if N > 0 else 0.0

    print("\n查询系统（输入'quit'退出）")
    while True:
        q = input("查询: ").strip()
        if q.lower() == 'quit':
            break
        if not q:
            continue

        # 调用搜索函数时传入avg_length
        results = search(idx, url_map, N, doc_lengths, avg_length, q, sw=sw)
        print(f"\n查询结果（共{len(results)}条）：")

        # 按day03格式打印结果
        for i, (docid, score, total_tf, target_tf) in enumerate(results, 1):
            print(f"[{i}] BM25分数: {score:.4f} | 总词频: {total_tf} | 目标词频: {target_tf}")
            print(f"   {url_map[docid]}（{docid}）\n")


if __name__ == "__main__":
    main()