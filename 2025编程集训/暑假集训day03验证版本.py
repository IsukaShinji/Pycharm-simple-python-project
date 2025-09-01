import os
import re
import jieba
from collections import defaultdict
from urllib.parse import unquote

# 停用词加载函数，与 day3 保持一致
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

# 从文件名还原原始 URL
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

# 单 HTML 处理函数，与 day3 保持一致
def process_single_html(html_path, stopwords):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            original_html = f.read()  # 保存原始 HTML 内容

        filename = os.path.basename(html_path)
        docid = filename.rsplit('_', 1)[-1].replace('.html', '') if '_' in filename else 'unknown'

        # 基础过滤：移除脚本、样式、注释
        content = original_html
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # 增强过滤：针对常见非正文标签
        special_patterns = [
            r'<nav.*?</nav>', r'<header.*?</header>', r'<footer.*?</footer>',
            r'<aside.*?</aside>', r'<div class="nav.*?</div>',
            r'<div class="menu.*?</div>', r'<div class="ad.*?</div>',
            r'<div class="footer.*?</div>', r'<div class="sidebar.*?</div>'
        ]
        for pattern in special_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

        # 提取纯文本
        text = re.sub(r'<[^>]+>', ' ', content)  # 替换标签为空格
        text = re.sub(r'\s+', ' ', text).strip()  # 合并空白

        # 分词与过滤
        words = jieba.cut(text, cut_all=False)  # 精确分词
        filtered_words = [
            word for word in words
            if word.strip()
               and word.strip() not in stopwords
               and not word.strip().isdigit()
               and word.strip() not in ',.!?;:"\'()[]{}、，。！？；：“”‘’（）【】'
        ]

        # 返回分词结果
        return filtered_words

    except Exception as e:
        print(f"\n⚠️ 处理 {os.path.basename(html_path)} 异常：{e} - 跳过")
        return []

# 主函数：统计每个HTML文件中关键词的出现次数
def count_keywords_per_html(debug_dir):
    stopwords = load_stopwords()
    total_counts = defaultdict(int)  # 总计统计
    file_counts = []  # 每个文件的统计结果

    # 遍历指定目录中的所有文件
    html_files = [f for f in os.listdir(debug_dir) if f.endswith('.html')]
    total_files = len(html_files)
    print(f"找到 {total_files} 个 HTML 文件，开始处理...\n")

    for idx, html_file in enumerate(html_files, 1):
        html_path = os.path.join(debug_dir, html_file)
        print(f"处理 [{idx}/{total_files}] 文件：{html_file}")

        filtered_words = process_single_html(html_path, stopwords)
        counts = defaultdict(int)

        for word in filtered_words:
            if word in {"人", "人工", "人工智能"}:
                counts[word] += 1
                total_counts[word] += 1

        # 记录当前文件的统计结果
        file_counts.append({
            "filename": html_file,
            "人": counts.get("人", 0),
            "人工": counts.get("人工", 0),
            "人工智能": counts.get("人工智能", 0)
        })

    # 输出每个文件的统计结果
    print("\n【每个文件的统计结果】")
    for result in file_counts:
        print(f"\n文件名：{result['filename']}")
        print(f"‘人’ 出现次数：{result['人']}")
        print(f"‘人工’ 出现次数：{result['人工']}")
        print(f"‘人工智能’ 出现次数：{result['人工智能']}")

    # 输出总体统计结果
    print("\n【总体统计结果】")
    print(f"‘人’ 总计出现次数：{total_counts.get('人', 0)}")
    print(f"‘人工’ 总计出现次数：{total_counts.get('人工', 0)}")
    print(f"‘人工智能’ 总计出现次数：{total_counts.get('人工智能', 0)}")

# 指定文件夹路径
debug_dir = r"C:\Users\21165\Desktop\编程集训\debug_texts"

# 执行统计
count_keywords_per_html(debug_dir)