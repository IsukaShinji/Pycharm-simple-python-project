from flask import Flask, request, render_template
import retriever
import os

app = Flask(__name__)

# 创建Web页面模板文件夹（必须命名为templates，Flask默认路径）
if not os.path.exists("./templates"):
    os.makedirs("./templates")

# 写入Web页面模板（无需手动创建，代码自动生成）
template_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>人大新闻检索（基于PDF）</title>
    <style>
        body { max-width: 1000px; margin: 50px auto; font-family: Arial; }
        .search-box { text-align: center; margin-bottom: 30px; }
        #query { width: 500px; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #0066cc; color: white; border: none; cursor: pointer; }
        .result { margin: 20px 0; padding: 15px; border-bottom: 1px solid #eee; }
        .title { font-size: 18px; margin-bottom: 8px; }
        .title a { color: #0066cc; text-decoration: none; }
        .content { color: #666; font-size: 14px; line-height: 1.5; }
        .no-result { color: #ff3333; text-align: center; font-size: 16px; }
    </style>
</head>
<body>
    <div class="search-box">
        <h1>人大新闻检索系统（基于PDF搭建）</h1>
        <form method="POST">
            <input type="text" id="query" name="query" placeholder="输入关键词搜索（如“通州校区”“新生”）" required>
            <button type="submit">搜索</button>
        </form>
    </div>

    <div class="results">
        {% if results %}
            <p>找到 {{ results|length }} 条相关结果：</p>
            {% for doc in results %}
                <div class="result">
                    <div class="title"><a href="{{ doc.url }}" target="_blank">{{ doc.title }}</a></div>
                    <div class="content">{{ doc.content[:150] }}...（点击标题查看全文）</div>
                </div>
            {% endfor %}
        {% elif query %}
            <div class="no-result">未找到“{{ query }}”相关结果，请换关键词重试！</div>
        {% endif %}
    </div>
</body>
</html>
'''
with open("./templates/index.html", "w", encoding="UTF-8") as f:
    f.write(template_content)


@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "GET":
        # Get请求：显示搜索页面
        return render_template("index.html", results=[], query="")
    else:
        # Post请求：处理搜索，返回结果
        query = request.form.get("query", "").strip()
        results = retriever.basic_retrieval(query)
        return render_template("index.html", results=results, query=query)


if __name__ == "__main__":
    # 启动本地Web服务（仅localhost访问，符合PDF范围）
    app.run(debug=True, host="localhost", port=5000)