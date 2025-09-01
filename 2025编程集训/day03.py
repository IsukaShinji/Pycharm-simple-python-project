import os
import re
import jieba
from collections import defaultdict, Counter
from urllib.parse import unquote


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

def bigram_tokenize(text: str) -> list:
    """生成文本的二元分词结果（相邻字符组合）"""
    text = text.strip()
    if len(text) < 2:
        return [text] if text else []
    return [text[i] + text[i+1] for i in range(len(text) - 1)]


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


def process_single_html(html_path, stopwords):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            original_html = f.read()

        filename = os.path.basename(html_path)
        docid = filename.rsplit('_', 1)[-1].replace('.html', '') if '_' in filename else 'unknown'
        problematic_docids = {'2516', '909'}

        if docid in problematic_docids:
            debug_dir = r"C:\Users\21165\Desktop\编程集训\debug_texts"
            os.makedirs(debug_dir, exist_ok=True)
            original_html_path = os.path.join(debug_dir, f"{docid}_original.html")
            with open(original_html_path, 'w', encoding='utf-8') as f:
                f.write(original_html)
            print(f"📋 已保存原始HTML: {original_html_path}")

        content = original_html
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        special_patterns = [
            r'<nav.*?</nav>', r'<header.*?</header>', r'<footer.*?</footer>',
            r'<aside.*?</aside>', r'<div class="nav.*?</div>',
            r'<div class="menu.*?</div>', r'<div class="ad.*?</div>',
            r'<div class="footer.*?</div>', r'<div class="sidebar.*?</div>'
        ]
        for pattern in special_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        text_length = len(text)  # 计算文本长度

        if docid in problematic_docids:
            debug_text_path = os.path.join(debug_dir, f"{docid}_extracted.txt")
            with open(debug_text_path, 'w', encoding='utf-8') as f:
                f.write(f"文档编号: {docid}\n")
                f.write(f"原始文件名: {filename}\n")
                f.write(f"提取文本长度: {text_length} 字符\n\n")
                f.write(text)
            print(f"📋 已保存提取文本: {debug_text_path}")

            # jieba分词
            jieba_words = [w for w in jieba.cut(text, cut_all=False) if w.strip() and w not in stopwords]
            # 二元分词
            bigram_words = [w for w in bigram_tokenize(text) if w.strip() and w not in stopwords]
            # 合并去重
            filtered_words = list(set(jieba_words + bigram_words))

        target_count = filtered_words.count("人工智能")
        if target_count > 20:
            print(f"🔍 高次数文档 {filename} (ID: {docid}):")
            print(f"   目标词出现次数: {target_count}")
            print(f"   提取文本长度: {text_length} 字符\n")

        return filtered_words, text_length  # 返回词列表和文本长度
    except Exception as e:
        print(f"\n⚠️ 处理{os.path.basename(html_path)}异常：{e} - 跳过")
        return [], 0


def batch_process_html(html_dir, stopwords):
    inverted_index = defaultdict(lambda: defaultdict(int))
    docid_to_url = {}
    doc_lengths = {}  # 存储每个文档的长度
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    total_files = len(html_files)
    processed = 0
    progress_records = []
    print(f"📂 批量处理：总文件数={total_files}，每100个打印进度")

    for filename in html_files:
        html_path = os.path.join(html_dir, filename)
        url, docid = restore_url_from_filename(filename)
        if not url or not docid:
            processed += 1
            continue
        docid_to_url[docid] = url

        try:
            words, text_length = process_single_html(html_path, stopwords)
            doc_lengths[docid] = text_length  # 保存文档长度
            for word in words:
                inverted_index[word][docid] += 1
        except Exception as e:
            print(f"\n⚠️ 批量处理{filename}异常：{e} - 跳过")

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

    # 保存文档长度到txt文件
    length_save_path = os.path.join(r"C:\Users\21165\Desktop\编程集训", "document_lengths.txt")
    with open(length_save_path, 'w', encoding='utf-8') as f:
        for docid, length in doc_lengths.items():
            f.write(f"{docid}\t{length}\n")
    print(f"✅ 文档长度已保存到：{length_save_path}")

    return inverted_index, docid_to_url, doc_lengths


def save_dict(inverted_index, docid_to_url, doc_lengths):
    save_dir = r"C:\Users\21165\Desktop\编程集训"
    os.makedirs(save_dir, exist_ok=True)

    dict_path = os.path.join(save_dir, "inverted_index_dict.txt")
    url_map_path = os.path.join(save_dir, "docid_to_url_map.txt")
    length_path = os.path.join(save_dir, "document_lengths.txt")

    with open(dict_path, 'w', encoding='utf-8') as f:
        for word, doc_counts in inverted_index.items():
            parts = [word] + [f"{docid}:{count}" for docid, count in doc_counts.items()]
            f.write('|'.join(parts) + '\n')
    print(f"\n✅ 词典已保存到：{dict_path}")

    with open(url_map_path, 'w', encoding='utf-8') as f:
        for docid, url in docid_to_url.items():
            f.write(f"{docid}\t{url}\n")
    print(f"✅ URL映射已保存到：{url_map_path}")

    with open(length_path, 'w', encoding='utf-8') as f:
        for docid, length in doc_lengths.items():
            f.write(f"{docid}\t{length}\n")
    print(f"✅ 文档长度已保存到：{length_path}")


def load_existing_data():
    save_dir = r"C:\Users\21165\Desktop\编程集训"
    dict_path = os.path.join(save_dir, "inverted_index_dict.txt")
    url_map_path = os.path.join(save_dir, "docid_to_url_map.txt")
    length_path = os.path.join(save_dir, "document_lengths.txt")

    if os.path.exists(dict_path) and os.path.exists(url_map_path) and os.path.exists(length_path):
        docid_to_url = {}
        with open(url_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '\t' not in line:
                    print(f"⚠️ 跳过无效URL映射行：{line}")
                    continue
                docid, url = line.split('\t', 1)
                docid_to_url[docid] = url

        doc_lengths = {}
        with open(length_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                docid, length = line.split('\t', 1)
                try:
                    doc_lengths[docid] = int(length)
                except ValueError:
                    continue

        inverted_index = defaultdict(lambda: defaultdict(int))
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 1:
                    print(f"⚠️ 跳过无效词典行（行号{line_num}）：{line}")
                    continue
                word = parts[0]
                for doc_count in parts[1:]:
                    if ':' not in doc_count:
                        print(f"⚠️ 跳过格式错误的条目（行号{line_num}）：{doc_count}")
                        continue
                    try:
                        docid, count = doc_count.split(':', 1)
                        inverted_index[word][docid] = int(count)
                    except ValueError as e:
                        print(f"⚠️ 解析失败（行号{line_num}）：{doc_count}，错误：{e}")
                    except Exception as e:
                        print(f"⚠️ 未知错误（行号{line_num}）：{e}")

        return inverted_index, docid_to_url, doc_lengths, True
    return defaultdict(lambda: defaultdict(int)), {}, {}, False


def parse_query(query):
    """查询词同时进行jieba分词和二元分词，合并去重"""
    jieba_terms = [q.strip() for q in jieba.cut(query) if q.strip()]
    bigram_terms = bigram_tokenize(query)
    return list(set(jieba_terms + bigram_terms))  # 去重合并


def calculate_bm25_score(term, docid, tf, doc_length, avg_doc_length, N, df, k1=1.2, b=0.75):
    """计算BM25分数"""
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)  # IDF计算
    term1 = tf * (k1 + 1)
    term2 = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
    return idf * (term1 / term2)


def execute_query(query, inverted_index, docid_to_url, doc_lengths):
    terms = parse_query(query)
    if not terms:
        return []

    # 准备BM25计算所需参数
    N = len(docid_to_url)  # 总文档数
    if N == 0:
        return []

    # 计算平均文档长度
    total_length = sum(doc_lengths.values())
    avg_doc_length = total_length / N if N > 0 else 0

    # 计算每个词的文档频率
    df = {term: len(inverted_index.get(term, {})) for term in terms}

    # 收集集所有相关文档
    result_docids = set()
    for term in terms:
        term_docs = {docid for docid in inverted_index.get(term, {}).keys()}
        result_docids.update(term_docs)
    if not result_docids:
        return []

    # 计算每个文档的BM25分数
    doc_scores = defaultdict(float)
    term_counts = defaultdict(dict)  # 记录每个文档中各词的词频

    for term in terms:
        for docid, count in inverted_index.get(term, {}).items():
            term_counts[docid][term] = count
            doc_length = doc_lengths.get(docid, avg_doc_length)
            df_term = df.get(term, 0)
            score = calculate_bm25_score(term, docid, count, doc_length, avg_doc_length, N, df_term)
            doc_scores[docid] += score

    # 排序并获取前20个结果
    sorted_docids = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])[:20]

    results = []
    for docid in sorted_docids:
        total_tf = sum(term_counts.get(docid, {}).values())
        target_tf = sum(term_counts.get(docid, {}).get(term, 0) for term in terms)
        results.append((docid, docid_to_url[docid], doc_scores[docid], total_tf, target_tf))

    return results


def query_interaction(inverted_index, docid_to_url, doc_lengths):
    print("\n==================== 查询系统 ====================")
    print("规则：单字直接查 / 多词空格分隔（OR逻辑） / 输入'quit'退出")
    while True:
        query = input("\n请输入查询词：").strip()
        if query.lower() == 'quit':
            print("👋 退出查询")
            break
        if not query:
            print("⚠️ 请输入有效查询词")
            continue

        results = execute_query(query, inverted_index, docid_to_url, doc_lengths)
        total = len(results)
        print(f"\n🔍 查询结果总数：{total}（按BM25分数降序排列）")

        show_count = min(20, total)
        for i in range(show_count):
            docid, url, score, total_tf, target_tf = results[i]
            print(f"[{i + 1}] 文档编号：{docid}")
            print(f"   原始URL：{url}")
            print(f"   BM25分数：{score:.4f}，总词频：{total_tf}，目标词出现：{target_tf}次\n")
        if total > 20:
            print(f"💡 还有{total - 20}条未显示，共{total}条")


def main():
    print("============================================================")
    print("📅 暑假集训day03 - BM25增强版：支持文档长度保存与BM25排序")
    print("============================================================")

    print("\n【1/3】加载停用词表...")
    stopwords = load_stopwords()
    print("✅ 停用词表加载完成")

    print("\n【2/3】检查已有词典/URL映射...")
    inverted_index, docid_to_url, doc_lengths, data_exists = load_existing_data()
    if data_exists:
        print(f"✅ 检测到已有数据：{len(inverted_index)}个词条，{len(docid_to_url)}条URL")
        query_interaction(inverted_index, docid_to_url, doc_lengths)
        return

    print("\n【3/3】无已有数据，开始批量处理HTML...")
    html_dir = r"C:\Users\21165\Desktop\编程集训\crawled_htm_files"
    if not os.path.exists(html_dir):
        print(f"❌ 错误：HTML文件目录不存在 - {html_dir}")
        return
    inverted_index, docid_to_url, doc_lengths = batch_process_html(html_dir, stopwords)

    print("\n【4/4】保存数据到指定路径...")
    save_dict(inverted_index, docid_to_url, doc_lengths)

    query_interaction(inverted_index, docid_to_url, doc_lengths)


if __name__ == "__main__":
    import numpy as np  # 延迟导入，仅在运行时需要

    main()
