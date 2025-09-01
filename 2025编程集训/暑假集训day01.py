from bs4 import BeautifulSoup
import jieba
from url_normalize import url_normalize
from urllib.parse import urlparse

#====================1.读取文件
def load_html(file_path):     #直接输入文件地址
    with open(file_path,'r',encoding='utf-8') as f:
        return f.read()

#======================2.提取并规范链接
def extract_and_normalize_links(soup,current_html_url):
    all_links=set()
    for anchor in soup.find_all('a'):
        href=anchor.attrs.get("href")
        if href is None or href.strip()=="":
            continue

        parsed= urlparse(href)  #解析URL

        if parsed.scheme and parsed.scheme not in{'http','https'}:
            all_links.add(href)
            continue

        if not href.startswith('http'):
            normalized_base_url=url_normalize(current_html_url.rstrip('/')+'/')
            href=url_normalize(normalized_base_url+href.lstrip('/'))
        else:
            href=url_normalize(href)

        href=href.replace('/#','#')

        all_links.add(href)
    return all_links

#=========================3.抓取标题并分词
def extract_and_cut_titles(soup):
    title_result={}
    if soup.title and soup.title.string:
        main_title=soup.title.string.strip()
        main_title_words=jieba.lcut(main_title)
        title_result["main_title"]=(main_title,main_title_words)

    multi_level_titles=[]
    for t in soup.find_all({"h1","h2","h3","h4","h5","h6"}):
        if t.string is not None:
            title_text=t.string.strip()
            title_words=jieba.lcut(title_text)
            multi_level_titles.append((title_text,title_words))
    title_result["multi_level_titles"] = multi_level_titles
    return title_result

def extract_and_cut_body(soup):
    body_result={}
    p_texts=[]
    for p in soup.find_all('p'):
        if p.string is not None:
            p_text=p.string.strip()
            if p_text:
                p_texts.append(p_text)
    raw_body="\n".join(p_texts)
    body_result["raw_body"]=raw_body

    body_words=jieba.lcut(raw_body)
    body_result["body_words"]=body_words
    return body_result

def main():
    # -------------------------- 配置参数（需用户根据实际情况修改）--------------------------
    HTML_FILE_PATH = r"C:\Users\21165\Desktop\人大新闻网.html"       # 备用：r"C:\Users\21165\Desktop\day01测试.html"    # 给定的HTML文档路径（本地文件）
    CURRENT_HTML_URL = "view-source:https://news.ruc.edu.cn/zonghexinwen.html"     #备用："https://kimi-demo.moonshot.cn/day01-test/index.html"   # 该HTML对应的在线URL（用于相对路径规范化）
    # -------------------------------------------------------------------------------------
    # 步骤1：加载HTML文档
    html_content = load_html(HTML_FILE_PATH)
    if not html_content:
        return  # 文件加载失败则退出

    # 步骤2：初始化BeautifulSoup解析器
    soup = BeautifulSoup(html_content, "html.parser")

    # 步骤3：抓取并规范化超链接
    normalized_links = extract_and_normalize_links(soup, CURRENT_HTML_URL)
    print("=" * 50)
    print("1. 抓取并规范化后的超链接（共{}个）：".format(len(normalized_links)))
    for idx, link in enumerate(sorted(normalized_links), 1):
        print(f"   {idx}. {link}")

    # 步骤4：抓取并分词标题
    title_result = extract_and_cut_titles(soup)
    print("\n" + "=" * 50)
    print("2. 标题及分词结果：")
    main_title_info = title_result.get("main_title")
    if main_title_info:
        main_title, main_title_words = main_title_info
    else:
        main_title, main_title_words = "无总标题", []
    print(f"   总标题：{main_title}")
    print(f"   总标题分词：{main_title_words}")
    print(f"   多级标题（h1-h6）：")
    for level_idx, (title_text, title_words) in enumerate(title_result["multi_level_titles"], 1):
        print(f"     {level_idx}. 标题：{title_text} | 分词：{title_words}")

    # 步骤5：抓取并分词正文
    body_result = extract_and_cut_body(soup)
    print("\n" + "=" * 50)
    print("3. 正文及分词结果：")
    print(f"   原始正文（前500字符）：{body_result['raw_body'][:500]}..." if len(
        body_result['raw_body']) > 500 else f"   原始正文：{body_result['raw_body']}")
    print(f"   正文分词（前20个词）：{body_result['body_words'][:20]}..." if len(
        body_result['body_words']) > 20 else f"   正文分词：{body_result['body_words']}")


if __name__ == "__main__":
    main()


