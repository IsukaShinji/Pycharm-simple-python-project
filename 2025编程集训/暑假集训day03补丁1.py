import jieba
from collections import defaultdict
from bs4 import BeautifulSoup

# 示例HTML内容
html_content = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>人工智能与机器学习</title>
</head>
<body>
    <h1>人工智能与机器学习</h1>
    <p>人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。</p>
    <p>机器学习是实现人工智能的一个重要方法。</p>
    <p>人是智能的主体。</p>
    <p>人工智能正在改变我们的生活。</p>
    <li>
        <a href="http://v.ruc.edu.cn/">微人大</a>
    </li>
</body>
</html>
"""

# 分词测试
def test_gen_terms():
    # 模拟从HTML提取文本
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)
    full_text = title + " " + text

    # 分词+过滤
    terms = jieba.lcut(full_text)
    filtered_terms = []
    for t in terms:
        t_stripped = t.strip()
        if len(t_stripped) > 0:  # 保留所有非空词
            filtered_terms.append(t_stripped)
    return filtered_terms

# 倒排索引测试
def test_inverted_index(terms):
    index = defaultdict(list)
    docid = 1  # 测试用的docid
    for term in terms:
        index[term].append(docid)
    return index

# 查询测试
def test_query(index, query_term):
    return list(index.get(query_term, []))  # 转换为列表并去重

# 运行测试
if __name__ == "__main__":
    terms = test_gen_terms()
    print("分词结果:", terms)
    index = test_inverted_index(terms)
    print("\n倒排索引:", index)
    query_term = "人工智能"
    results = test_query(index, query_term)
    print(f"\n查询结果 for '{query_term}':", results)
    query_term = "人"
    results = test_query(index, query_term)
    print(f"\n查询结果 for '{query_term}':", results)
    query_term = "微人大"
    results = test_query(index, query_term)
    print(f"\n查询结果 for '{query_term}':", results)