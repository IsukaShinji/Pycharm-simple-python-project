def restore_url_from_filename(filename):
    if not filename.endswith('.html'):
        return None, None
    name_part = filename[:-5]  # 去除.html后缀
    parts = name_part.split('_')
    if len(parts) < 3:
        return None, None
    domain = parts[0]
    # 还原路径：将拆分后的路径部分用'/'拼接（处理根路径）
    path_parts = parts[1:-1]
    path = '/'.join(path_parts) if path_parts else ''
    # 根路径特殊处理（避免URL结尾多一个'/'）
    url = f"http://{domain}/{path}" if path else f"http://{domain}"
    docid = parts[-1]
    return url, docid