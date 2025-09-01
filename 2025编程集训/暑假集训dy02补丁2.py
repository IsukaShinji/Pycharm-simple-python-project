import os
import chardet

def check_file_encoding(file_path):
    """检测单个文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            print(f"{file_path} → 编码：{result['encoding']}，置信度：{result['confidence']}")
    except Exception as e:
        print(f"{file_path} → 错误：{e}")

def batch_check_folder(folder_path):
    """批量检测文件夹内所有文件"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.html'):  # 只检测HTML文件
                check_file_encoding(os.path.join(root, file))

if __name__ == "__main__":
    target = input("请输入文件/文件夹路径：").strip()
    if os.path.isfile(target):
        check_file_encoding(target)
    elif os.path.isdir(target):
        batch_check_folder(target)
    else:
        print("路径无效！")