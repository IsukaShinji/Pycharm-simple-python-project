from urllib.parse import urlparse, urljoin
import os

def save_html(html_doc, url, htmlpath, count):
    """保存HTML文件，保留原始URL的关键信息，确保文件名合法且可高效还原"""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc  # 获取域名
        path = parsed_url.path  # 获取路径部分

        # 将路径中的"/"替换为"_"
        path_part = path.replace("/", "_")

        # 如果路径为空，设置为"root"
        if not path_part:
            path_part = "root"

        # 将域名和处理后的路径组合成文件名
        filename = f"{domain}_{path_part}_{count}.html"

        # 获取完整的文件路径
        full_path = os.path.join(htmlpath, filename)

        # 确保目录存在
        os.makedirs(htmlpath, exist_ok=True)

        # 将内容强制转换为UTF-8编码
        try:
            content = html_doc.encode('utf-8', errors='replace').decode('utf-8')
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已保存HTML：{full_path}")
        except Exception as e:
            print(f"保存失败：{e}（URL：{url}）")
    except Exception as e:
        print(f"保存失败：{e}（URL：{url}）")