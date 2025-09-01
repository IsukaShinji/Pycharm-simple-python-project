# -*- coding: utf-8 -*-
# grid_search_bm25.py —— BM25 (k1, b, title_weight) 参数网格搜索（DEBUG 模式）
import json
import time
from itertools import product

# 复用现有登录/提交逻辑
import client
import search_engine

def run_once(idx: str, queries, k1: float, b: float, title_weight: float):
    """设置 BM25 参数和标题权重后跑一遍查询，返回 MRR。"""
    # 修改 search_engine 模块内的常量
    search_engine.K1 = k1
    search_engine.B = b
    search_engine.title_weight = title_weight  # 新增标题权重参数设置

    tot_urls = []
    for q in queries:
        urls = search_engine.evaluate(q)  # 每个查询返回 20 个 URL
        tot_urls.append(urls)

    # DEBUG 模式：passwd 传空字符串
    mode, mrr = client.send_ans(idx, "", tot_urls)
    return mode, mrr

def main():
    print("=== BM25 参数网格搜索（含 title_weight，DEBUG 模式）===")
    print(f"将连接到服务器: {client.base_url}")
    print("提示：本脚本只用于 DEBUG（passwd 为空），请勿用于正式提交！\n")

    idx = input("idx（随便填个标识，便于服务器区分调试记录）: ").strip()
    passwd = ""  # DEBUG 模式固定为空

    # 先登录获取查询（DEBUG 模式下服务器会返回 queries）
    queries = client.login(idx, passwd)

    # 网格参数设置
    k1_list = [1.0, 1.1, 1.2, 1.15, 1.19, 1.21, 1.22, 1.23, 1.25, 1.27]
    b_list = [0.50, 0.65, 0.7, 0.75, 0.85, 0.90]
    # 生成 title_weight 列表（0.1 到 10.0，间隔 0.1）
    title_weight_list = [round(0.1 * i, 1) for i in range(1, 101)]  # 1-100 对应 0.1-10.0

    results = []  # 存 (k1, b, title_weight, mrr)
    best = (-1.0, None, None, None)  # (mrr, k1, b, title_weight)

    # 三重参数网格搜索
    for k1, b, title_weight in product(k1_list, b_list, title_weight_list):
        print(f"\n>>> 试参：k1={k1:.2f}, b={b:.2f}, title_weight={title_weight:.1f}")
        try:
            mode, mrr = run_once(idx, queries, k1, b, title_weight)
            print(f"    返回模式: {mode} | MRR@20 = {mrr:.6f}")
            results.append((k1, b, title_weight, mrr))
            if mrr > best[0]:
                best = (mrr, k1, b, title_weight)
        except Exception as e:
            print(f"    失败: {e}")
        # 增加间隔时间（因参数组合增多，避免请求过于频繁）
        time.sleep(1.0)

    # 排序并打印汇总
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n=== 结果汇总（降序） ===")
    for k1, b, title_weight, mrr in results:
        print(f"k1={k1:.2f}, b={b:.2f}, title_weight={title_weight:.1f} -> MRR={mrr:.6f}")

    # 最优参数
    if best[1] is not None:
        print(f"\n✅ 最优参数：k1={best[1]:.2f}, b={best[2]:.2f}, title_weight={best[3]:.1f}，MRR={best[0]:.6f}")
    else:
        print("\n⚠️ 未获得有效结果，请检查网络/服务器/返回格式。")

    # 保存到本地 json
    with open("bm25_grid_results_with_title_weight.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"k1": k1, "b": b, "title_weight": title_weight, "mrr": mrr}
             for k1, b, title_weight, mrr in results],
            f, ensure_ascii=False, indent=2
        )
    print("已保存：bm25_grid_results_with_title_weight.json")

if __name__ == "__main__":
    main()