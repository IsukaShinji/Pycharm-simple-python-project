import os
import shutil
from urllib.parse import quote, unquote


def restore_url_from_filename(filename):
    """从文件名解析原始URL和编号（复用原解析函数）"""
    if not filename.endswith('.html'):
        return None, None
    name_part = filename[:-5]
    parts = name_part.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    encoded_url, docid = parts[0], parts[1]
    try:
        original_url = unquote(encoded_url)
        return encoded_url, docid  # 返回编码后的URL和编号（用于查重）
    except:
        return None, None


def fix_duplicate_files(save_dir):
    """检查并修复目录中重复的文件编号"""
    if not os.path.exists(save_dir):
        print(f"错误：目录不存在 - {save_dir}")
        return

    # 1. 收集所有文件的信息
    file_info = []  # 存储 (原文件名, encoded_url, 编号, 完整路径)
    all_numbers = set()  # 记录所有已存在的编号
    invalid_files = []  # 记录无法解析的文件

    for filename in os.listdir(save_dir):
        if not filename.endswith('.html'):
            continue

        full_path = os.path.join(save_dir, filename)
        encoded_url, docid = restore_url_from_filename(filename)

        if not encoded_url or not docid or not docid.isdigit():
            invalid_files.append(filename)
            continue

        file_info.append({
            'original_name': filename,
            'encoded_url': encoded_url,
            'number': int(docid),
            'path': full_path
        })
        all_numbers.add(int(docid))

    # 2. 找出重复的编号（相同数字）
    number_counts = {}
    for info in file_info:
        num = info['number']
        number_counts[num] = number_counts.get(num, 0) + 1

    duplicate_numbers = [num for num, count in number_counts.items() if count > 1]
    if not duplicate_numbers and not invalid_files:
        print("✅ 未发现重复编号或无效文件，所有文件均符合规范")
        return

    # 3. 生成新的唯一编号（从最大编号+1开始）
    max_existing = max(all_numbers) if all_numbers else 0
    new_number = max_existing + 1

    # 4. 修复重复编号的文件
    fixed_count = 0
    for info in file_info:
        num = info['number']
        if num in duplicate_numbers:
            # 生成新文件名：encoded_url_新编号.html
            new_filename = f"{info['encoded_url']}_{new_number}.html"
            new_path = os.path.join(save_dir, new_filename)

            # 确保新文件名不冲突（极端情况）
            while os.path.exists(new_path):
                new_number += 1
                new_filename = f"{info['encoded_url']}_{new_number}.html"
                new_path = os.path.join(save_dir, new_filename)

            # 重命名文件
            shutil.move(info['path'], new_path)
            print(f"🔄 修复重复编号：{info['original_name']} → {new_filename}")

            # 更新状态
            all_numbers.add(new_number)
            new_number += 1
            fixed_count += 1
            # 从重复列表中移除（只保留第一次出现的编号）
            if number_counts[num] == 1:
                duplicate_numbers.remove(num)
            else:
                number_counts[num] -= 1

    # 5. 输出结果
    print(f"\n修复完成：共处理 {fixed_count} 个重复编号文件")
    if invalid_files:
        print(f"⚠️ 发现 {len(invalid_files)} 个无效格式文件（需手动检查）：")
        for f in invalid_files:
            print(f"  - {f}")


if __name__ == "__main__":
    # 目标目录（与爬虫保存目录一致）
    target_dir = r"C:\Users\21165\Desktop\编程集训\crawled_htm_files"
    print(f"开始检查目录：{target_dir}")
    fix_duplicate_files(target_dir)
    print("操作结束")
