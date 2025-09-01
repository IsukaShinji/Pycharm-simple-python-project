import os
import tempfile
from urllib.parse import quote, unquote  # 使用修改后函数依赖的编码工具


# ----------------------
# 1. 导入修改后的两个核心函数
# ----------------------
def modified_save_html(url, htmlpath, count):
    """仅用于生成文件名（简化版，专注URL转存逻辑）"""
    try:
        encoded_url = quote(url, safe='')  # 完整编码URL
        filename = f"{encoded_url}_{count}.html"
        full_path = os.path.join(htmlpath, filename)
        return filename  # 仅返回文件名用于测试
    except Exception as e:
        print(f"保存失败：{e}（URL：{url}）")
        return None


def modified_restore_url_from_filename(filename):
    """修改后的URL还原函数"""
    if not filename.endswith('.html'):
        return None
    name_part = filename[:-5]
    parts = name_part.rsplit('_', 1)  # 分割编码URL和计数
    if len(parts) != 2:
        return None
    encoded_url, _ = parts
    try:
        return unquote(encoded_url)  # 解码还原
    except Exception as e:
        print(f"解码失败：{e}（文件名：{filename}）")
        return None


# ----------------------
# 2. 测试数据集（覆盖各类URL场景）
# ----------------------
test_urls = [
    # 基础HTTP/HTTPS
    "http://keyan.ruc.edu.cn/",
    "https://xsc.ruc.edu.cn/about",
    # 带路径和多级目录
    "http://keyan.ruc.edu.cn/news/research/2023",
    "https://xsc.ruc.edu.cn/activity?type=academic",
    # 带查询参数和锚点
    "http://xsc.ruc.edu.cn/xjgl2/zgrmdxbksxjglgd.htm",
    "https://xsc.ruc.edu.cn/notice?id=123&category=important#top",
    # 含特殊字符（空格、中文、符号）
    "http://keyan.ruc.edu.cn/article?title=机器学习 应用",
    "https://xsc.ruc.edu.cn/event?name=暑期集训_2024",
]


# ----------------------
# 3. 执行测试
# ----------------------
def test_url_restore():
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print("测试开始，共{}个URL样本\n".format(len(test_urls)))
        success_count = 0

        for i, original_url in enumerate(test_urls, 1):
            # 生成文件名
            filename = modified_save_html(original_url, temp_dir, i)
            if not filename:
                print(f"样本{i}失败：生成文件名失败")
                continue

            # 还原URL
            restored_url = modified_restore_url_from_filename(filename)

            # 验证结果
            if restored_url == original_url:
                success_count += 1
                print(f"样本{i} [成功]")
                print(f"  原始URL：{original_url}")
                print(f"  还原URL：{restored_url}\n")
            else:
                print(f"样本{i} [失败]")
                print(f"  原始URL：{original_url}")
                print(f"  还原URL：{restored_url}\n")

        # 统计结果
        accuracy = (success_count / len(test_urls)) * 100
        print(f"测试结束：共{len(test_urls)}个样本，成功{success_count}个，准确率：{accuracy:.2f}%")


# 执行测试
if __name__ == "__main__":
    test_url_restore()