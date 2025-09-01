import requests
from bs4 import BeautifulSoup
from url_normalize import url_normalize
import time
import os

# 严格遵循PDF指定的目标域名（）
BASE_URL = "https://news.ruc.edu.cn/"
START_URL = "https://news.ruc.edu.cn/zonghexinwen.html"  # 综合新闻首页
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
SLEEP_TIME = 1  # 爬取间隔，避免给服务器压力（PDF未提但合规必要）
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 链接去重（用set存储，对应PDF“避免重复爬取”思路）
crawled_urls = set()
to_crawl_urls = [START_URL]


def normalize_url(cur_url, href):
    """URL规范化（处理相对路径、多余字符，对应PDF ）"""
    if not href or href.startswith("mailto:") or href.endswith((".png", ".jpg", ".gif")):
        return None  # 过滤无效链接（PDF未提但需优化）
    if not href.startswith(("http://", "https://")):
        normalized = url_normalize(f"{cur_url.rstrip('/')}/{href.lstrip('/')}")
    else:
        normalized = url_normalize(href)
    # 仅保留PDF指定域名的链接（）
    if BASE_URL in normalized:
        return normalized
    return None


def crawl_page(url):
    """爬取单个页面并保存HTML（对应PDF ）"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # 处理HTTP状态码（如404、500，对应PDF ）
        response.encoding = "UTF-8"  # 匹配PDF中Content-Type编码（）

        # 用URL哈希值作为文档ID，避免重复存储
        doc_id = hash(url)
        with open(f"{DATA_DIR}/{doc_id}.html", "w", encoding="UTF-8") as f:
            f.write(response.text)
        print(f"爬取成功：{url} → 文档ID：{doc_id}")

        # 提取页面中的链接（用BeautifulSoup，对应PDF ）
        soup = BeautifulSoup(response.text, "html.parser")  # PDF推荐解析器
        for a_tag in soup.find_all("a"):
            href = a_tag.get("href")
            normalized_url = normalize_url(url, href)
            if normalized_url and normalized_url not in crawled_urls:
                to_crawl_urls.append(normalized_url)
                crawled_urls.add(normalized_url)

    except requests.exceptions.RequestException as e:
        print(f"爬取失败 {url}：{e}")


# 启动爬虫（广度优先，对应PDF“爬取给定域名下所有网页”要求）
if __name__ == "__main__":
    crawled_urls.add(START_URL)
    while to_crawl_urls and len(crawled_urls) <= 50:  # 限制爬取50个页面，避免耗时过久
        current_url = to_crawl_urls.pop(0)
        crawl_page(current_url)
        time.sleep(SLEEP_TIME)
    print(f"爬取结束！共获取 {len(crawled_urls)} 个页面")