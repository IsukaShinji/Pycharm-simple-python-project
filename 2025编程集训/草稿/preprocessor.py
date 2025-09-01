from bs4 import BeautifulSoup
import jieba
import os
import json
from collections import defaultdict

DATA_DIR = "./data"
PROCESSED_DIR = "./processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 停用词表（过滤无意义词汇，对应PDF分词优化思路）
STOPWORDS = {"的", "是", "在", "和", "及", "与", "等", "为", "了", "有", "这", "那", "个", "件"}


def extract_content(html_path):
    """提取标题和正文（基于PDF HTML结构：<title>、<h1>-<h6>、<p>，）"""
    with open(html_path, "r", encoding="UTF-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 提取标题：优先<h1>（新闻主标题），其次<title>
    title = ""
    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.string:
        title = h1_tag.string.strip()
    else:
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

    # 提取正文：所有<p>标签内容合并
    content = ""
    for p_tag in soup.find_all("p"):
        if p_tag.string:
            content += p_tag.string.strip() + "\n"
    return title, content.strip()


def tokenize(text):
    """中英文分词（中文用jieba.lcut，英文用split，对应PDF ）"""
    tokens = []
    # 中文分词（精确模式，PDF推荐）
    chinese_tokens = jieba.lcut(text)
    for token in chinese_tokens:
        if token not in STOPWORDS and len(token) > 1 and token.isalnum():
            tokens.append(token.lower())
    # 英文分词
    for token in text.split():
        clean_token = "".join([c.lower() for c in token if c.isalnum()])
        if clean_token not in STOPWORDS and len(clean_token) > 1:
            tokens.append(clean_token)
    return tokens


def process_all_docs():
    """批量预处理所有爬取的文档"""
    processed_docs = defaultdict(dict)
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".html"):
            doc_id = int(filename.split(".")[0])
            html_path = os.path.join(DATA_DIR, filename)

            title, content = extract_content(html_path)
            tokens = tokenize(title + " " + content)  # 标题+正文合并分词

            # 存储预处理结果（含模拟URL，实际可从爬虫记录获取）
            processed_docs[doc_id] = {
                "title": title,
                "content": content,
                "tokens": tokens,
                "url": f"https://news.ruc.edu.cn/?doc_id={doc_id}"  # 模拟原文链接
            }

    # 保存到JSON文件（便于后续索引构建）
    with open(f"{PROCESSED_DIR}/processed_docs.json", "w", encoding="UTF-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)
    print(f"预处理完成！共处理 {len(processed_docs)} 个文档")
    return processed_docs


if __name__ == "__main__":
    # 添加人大专属词汇（提升分词准确性，如“通州校区”，对应PDF分词优化思路）
    jieba.add_word("通州校区")
    jieba.add_word("陕公精神")
    process_all_docs()