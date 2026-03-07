import os
import json
import shutil
import random
import threading
from urllib.parse import urlparse, urljoin, quote
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# 全局唯一锁（保证多线程下编号/队列安全）
global_lock = threading.Lock()

# User-Agent池（反爬用，随机切换）
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0'
]

def get_html_with_timeout(url, timeout=15):
    """跨平台超时请求（替代signal，Windows兼容）"""
    result = [None]  # 用列表存结果，实现线程内修改

    def fetch():
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            resp = requests.get(url, headers=headers, timeout=(5, 10), allow_redirects=True)
            resp.encoding = resp.apparent_encoding
            result[0] = resp.text
        except Exception as e:
            print(f"⚠️ 请求失败（{url}）：{str(e)}")
            result[0] = None

    thread = threading.Thread(target=fetch)
    thread.start()
    thread.join(timeout)  # 超时等待
    if thread.is_alive():
        print(f"⏰ 请求超时（{url}，{timeout}秒）")
        return None
    return result[0]


def extract_links(html, base_url, allowed_prefixes):
    links = set()
    if not html:
        print(f"❌ 页面内容为空，无法提取链接（base_url: {base_url}）")
        return links

    # 允许的网页后缀（空字符串表示无后缀的网页URL）
    allowed_extensions = {'.html', '.htm', ''}
    # 需排除的非网页后缀（图片、PDF、文档、视频等）
    excluded_extensions = {
        '.jpg', '.jpeg', '.png', '.gif',  # 图片
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # 文档
        '.mp4', '.avi', '.mov',  # 视频
        '.zip', '.rar', '.exe'  # 压缩包/可执行文件
    }

    try:
        soup = BeautifulSoup(html, 'html.parser')
        a_tags = soup.find_all('a', href=True)

        if not a_tags:
            print(f"⚠️ 未找到任何<a>标签（base_url: {base_url}）")
            return links

        for a in a_tags:
            href = a['href'].strip()
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            path = parsed.path
            # 提取路径的后缀（如 "/index.html" → ".html"）
            ext = os.path.splitext(path)[1].lower()

            # 条件：后缀是“允许的网页后缀” 或 “不属于排除的非网页后缀”
            if ext in allowed_extensions or ext not in excluded_extensions:
                norm_url = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip('/')
                # 仅保留指定前缀的链接
                if any(norm_url.startswith(prefix) for prefix in allowed_prefixes):
                    links.add(norm_url)

        print(f"✅ 从 {base_url} 提取到 {len(links)} 个有效网页链接")
        return links
    except Exception as e:
        print(f"❌ 提取链接失败（base_url: {base_url}）：{str(e)}")
        return links

def get_global_max_number(save_dir):
    """获取目录中已存文件的最大编号（保证编号唯一）"""
    max_num = -1
    if not os.path.exists(save_dir):
        return max_num
    for f in os.listdir(save_dir):
        if f.endswith('.html'):
            try:
                parts = f.rsplit('_', 1)
                if len(parts) == 2 and parts[1].replace('.html', '').isdigit():
                    max_num = max(max_num, int(parts[1].replace('.html', '')))
            except:
                continue
    return max_num

def save_html(html, url, save_dir, global_counter):
    """保存HTML文件，生成全局唯一编号"""
    with global_lock:  # 加锁保证编号唯一
        try:
            encoded_url = quote(url, safe='')
            file_number = global_counter[0]
            global_counter[0] += 1  # 立即递增，保证下一个编号不重复

            filename = f"{encoded_url}_{file_number}.html"
            full_path = os.path.join(save_dir, filename)

            # 极端情况：文件已存在（理论上不会，仅双保险）
            if os.path.exists(full_path):
                new_num = get_global_max_number(save_dir) + 1
                filename = f"{encoded_url}_{new_num}.html"
                full_path = os.path.join(save_dir, filename)
                global_counter[0] = new_num + 1

            os.makedirs(save_dir, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return True, file_number, filename
        except Exception as e:
            print(f"⚠️ 保存文件失败（{url}）：{str(e)}")
            return False, -1, None

def save_progress(progress_path, data):
    """保存爬取进度（临时文件防损坏）"""
    if not data:
        return
    temp = progress_path + '.tmp'
    try:
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        shutil.move(temp, progress_path)
    except Exception as e:
        print(f"⚠️ 保存进度失败：{str(e)}")


def load_progress(progress_path):
    """加载爬取进度（兼容首次运行）"""
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载进度失败：{str(e)}")
        return None


def main():
    # ---------------------- 配置区 ----------------------
    allowed_prefixes = ["http://keyan.ruc.edu.cn/", "http://xsc.ruc.edu.cn/"]  # 允许爬取的网站前缀
    save_dir = r"C:\Users\21165\Desktop\编程集训\crawled_htm_files"         # HTML保存目录
    progress_path = r"C:\Users\21165\Desktop\编程集训\crawl_progress.json"  # 进度保存路径
    max_workers = 15                                                       # 15线程
    max_retries = 5                                                        # 每个URL最大重试次数
    check_interval = 30                                                    # 线程监控间隔（秒）

    # 初始化失败URL日志
    with open("failed_urls.txt", "a", encoding="utf-8") as f:
        f.write(f"\n===== 新会话 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    # ---------------------- 初始化全局计数器 ----------------------
    initial_max = get_global_max_number(save_dir)
    global_counter = [initial_max + 1]  # 从已有最大编号+1开始

    # ---------------------- 加载历史进度 ----------------------
    progress = load_progress(progress_path) or {
        "total_crawled": 0,
        "all_urls": [],
        "used_urls": [],
        "queue": [],
        "retries": {}
    }
    all_urls = set(progress["all_urls"])   # 所有发现的URL（去重）
    used_urls = set(progress["used_urls"]) # 已爬取的URL
    queue = deque(progress["queue"])       # 待爬取队列
    retries = progress["retries"]          # 重试记录
    total_crawled = progress["total_crawled"]  # 总爬取数

    # ---------------------- 初始化队列（首次或队列为空时） ----------------------
    if not queue:
        for prefix in allowed_prefixes:
            parsed = urlparse(prefix)
            norm_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
            if norm_url not in all_urls:
                queue.append(norm_url)
                all_urls.add(norm_url)
        print(f"初始化队列：添加 {len(queue)} 个起始URL")

    # ---------------------- 线程监控（防止卡死） ----------------------
    last_active_time = time.time()
    active_lock = threading.Lock()  # 监控锁

    def monitor():
        while True:
            time.sleep(check_interval)
            with active_lock:
                elapsed = time.time() - last_active_time
                if elapsed > check_interval * 2:  # 超过2倍间隔无活动，视为卡死
                    print(f"⚠️ 检测到可能卡死（{elapsed:.1f}秒无活动），尝试唤醒线程...")
                    with global_lock:
                        queue.append("http://dummy-url-to-wake-threads")  # 加入唤醒URL

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # ---------------------- 工作线程逻辑 ----------------------
    def worker():
        nonlocal total_crawled, last_active_time
        while True:
            # 1. 取待爬URL（加锁防止队列空）
            with global_lock:
                if not queue:
                    break
                url = queue.popleft()

            # 2. 跳过已爬或重试超限的URL
            if url in used_urls or retries.get(url, 0) >= max_retries:
                continue

            # 3. 标记线程活跃时间
            with active_lock:
                last_active_time = time.time()

            try:
                # 4. 标记为“已爬取中”（防止重复爬取）
                used_urls.add(url)
                with global_lock:
                    total_crawled += 1
                    current_crawl = total_crawled  # 当前爬取计数

                # 5. 超时请求页面
                html = get_html_with_timeout(url, timeout=15)
                if not html:  # 请求失败，重试
                    retries[url] = retries.get(url, 0) + 1
                    with global_lock:
                        queue.append(url)  # 放回队列
                    print(
                        f"❌ 已爬取 {current_crawl} | 重试{retries[url]}/{max_retries} | URL: {url} | 队列剩余: {len(queue)}")

                    # 重试超限则记录
                    if retries[url] >= max_retries:
                        with open("failed_urls.txt", "a", encoding="utf-8") as f:
                            f.write(f"{url}（重试{max_retries}次失败）\n")

                    time.sleep(random.uniform(1, 2))  # 重试延迟更长
                    continue

                retries.pop(url, None)  # 成功则清除重试记录

                # 6. 保存HTML文件（全局唯一编号）
                success, file_number, filename = save_html(html, url, save_dir, global_counter)

                # 7. 提取并加入新链接
                links = extract_links(html, url, allowed_prefixes)
                for link in links:
                    if link not in all_urls:
                        with global_lock:  # 加锁保护共享数据
                            all_urls.add(link)
                            queue.append(link)
                        print(f"📌 新增链接：{link} | 队列剩余：{len(queue)}")

                # 8. 成功爬取的输出
                if success:
                    print(f"✅ 已爬取 {current_crawl} | 编号：{file_number} | URL：{url} | 队列剩余：{len(queue)}")

                # 9. 定期保存进度（每20次爬取保存一次）
                if current_crawl % 20 == 0:
                    with global_lock:
                        save_progress(progress_path, {
                            "total_crawled": total_crawled,
                            "all_urls": list(all_urls),
                            "used_urls": list(used_urls),
                            "queue": list(queue),
                            "retries": retries,
                            "last_number": global_counter[0] - 1
                        })

                # 10. 随机延迟（模拟人类行为，反爬）
                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                print(f"⚠️ 线程错误: {str(e)} | URL: {url}")
                with global_lock:
                    queue.append(url)  # 异常URL放回队列
                time.sleep(1)

    # 启动工作线程
    print(f"开始爬取（线程数：{max_workers}）| 起始编号：{global_counter[0]}")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(max_workers):
                executor.submit(worker)
    except Exception as e:
        print(f"主进程错误: {str(e)}")

    #  爬取结束，最终保存进度
    with global_lock:
        save_progress(progress_path, {
            "total_crawled": total_crawled,
            "all_urls": list(all_urls),
            "used_urls": list(used_urls),
            "queue": list(queue),
            "retries": retries,
            "last_number": global_counter[0] - 1
        })
    print(f"\n🎉 爬取完成！共爬取 {total_crawled} 页 | 最后文件编号: {global_counter[0] - 1}")
    print(f"失败URL记录在：{os.path.abspath('failed_urls.txt')}")


if __name__ == "__main__":
    main()