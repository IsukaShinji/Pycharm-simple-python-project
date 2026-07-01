# 检索系统草稿

本目录保存暑假编程集训中小型检索系统的模块化草稿代码。相比上级目录中的按天脚本，这里更接近最终工程结构，把爬虫、预处理、索引、检索和 Web 页面拆成独立模块。

## 文件导览

| 文件 | 说明 |
| --- | --- |
| `spider.py` | 抓取指定域名页面，提取链接并保存 HTML。 |
| `preprocessor.py` | 读取 HTML 文件，抽取标题、正文、链接等字段。 |
| `inverted_index.py` | 构建倒排索引，用于关键词到文档的快速映射。 |
| `retriever.py` | 基于索引执行检索和结果排序。 |
| `app.py` | Flask Web 页面原型，提供搜索入口和结果展示。 |

## 运行提示

可以从 `app.py` 作为入口查看检索页面原型：

```bash
python app.py
```

运行前需要确认 HTML 数据目录、索引文件路径和依赖包是否与当前机器一致。常见依赖包括 `flask`、`requests`、`beautifulsoup4` 和 `url-normalize`。
