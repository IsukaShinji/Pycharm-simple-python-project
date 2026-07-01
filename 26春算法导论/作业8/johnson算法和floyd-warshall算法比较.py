import random
import time
import math
from typing import List, Tuple, Dict


NUM_VERTICES = 200            # 顶点数量
EDGE_PROB = 0.2               # 边生成概率 (稀疏图推荐0.1~0.3，稠密图可设为0.8~1.0)
                                # 注意：Floyd-Warshall 在稠密图耗时急剧上升，可调低顶点数
MAX_BASE_WEIGHT = 10          # 基础正权重上界
POTENTIAL_RANGE = 20          # 顶点势能范围，决定负权边出现的幅度
SEED = 42                     # 随机种子，保证可复现

INF = math.inf


def generate_graph(n: int, edge_prob: float, max_base: int, pot_range: int) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
    """
    生成无负环的有向图。利用势能函数构造权重，保证无负环。
    返回: 邻接矩阵 (用于Floyd), 邻接表边列表 (用于Johnson)
    """
    random.seed(SEED)
    # 为每个顶点随机生成势能值
    potential = [random.randint(-pot_range, pot_range) for _ in range(n)]

    adj_matrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        adj_matrix[i][i] = 0

    edges = []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if random.random() < edge_prob:
                # 权重 = 势能差 + 随机正偏移，保证 w + pot[u] - pot[v] >= 0 → 无负环
                base = random.uniform(0, max_base)
                w = potential[v] - potential[u] + base
                w = round(w, 2)          # 保留两位小数，便于观察
                adj_matrix[u][v] = w
                edges.append((u, v, w))
    return adj_matrix, edges


def floyd_warshall(adj: List[List[float]]) -> List[List[float]]:
    """Floyd-Warshall 全源最短路径"""
    n = len(adj)
    dist = [row[:] for row in adj]       # 拷贝矩阵
    for k in range(n):                    # 中间节点
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                # 松弛操作
                if dk[j] != INF and dik + dk[j] < di[j]:
                    di[j] = dik + dk[j]
    return dist


def johnson(n: int, edges: List[Tuple[int, int, float]]) -> List[List[float]]:
    """Johnson 全源最短路径，使用 Bellman-Ford + Dijkstra"""
    # 1. 添加虚拟节点 n，与所有顶点连接权重0的边
    dist = [0.0] * (n + 1)          # 从虚拟节点出发的距离
    # Bellman-Ford: 由于图保证无负环，只需放松 n 次 (实际 n 条边, 节点数 n+1)
    for _ in range(n):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        # 虚拟节点出发的边：0 权重
        for i in range(n):
            if dist[n] + 0 < dist[i]:
                dist[i] = dist[n]   # = 0
                updated = True
        if not updated:
            break
    h = dist[:n]                    # 各顶点的势能

    # 2. 重新赋权边，确保非负
    reweighted_edges = [[] for _ in range(n)]
    for u, v, w in edges:
        new_w = w + h[u] - h[v]
        reweighted_edges[u].append((v, new_w))

    # 3. 对每个顶点运行 Dijkstra
    all_pairs_dist = [[INF] * n for _ in range(n)]
    import heapq
    for s in range(n):
        dist_s = [INF] * n
        dist_s[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_s[u]:
                continue
            for v, w in reweighted_edges[u]:
                nd = d + w
                if nd < dist_s[v]:
                    dist_s[v] = nd
                    heapq.heappush(pq, (nd, v))
        # 恢复真实距离: d(u,v) = dist'(u,v) - h[u] + h[v]
        for v in range(n):
            if dist_s[v] != INF:
                all_pairs_dist[s][v] = dist_s[v] - h[s] + h[v]
    return all_pairs_dist


def compare_algorithms():
    print("图参数：")
    print(f"  顶点数: {NUM_VERTICES}")
    print(f"  边生成概率: {EDGE_PROB}")
    print(f"  预期边数: 约 {NUM_VERTICES * (NUM_VERTICES-1) * EDGE_PROB:.0f}")
    print(f"  最大基础权重: {MAX_BASE_WEIGHT}  势能范围: ±{POTENTIAL_RANGE}")
    print()

    adj_matrix, edge_list = generate_graph(NUM_VERTICES, EDGE_PROB, MAX_BASE_WEIGHT, POTENTIAL_RANGE)

    # --- Floyd-Warshall ---
    start = time.perf_counter()
    dist_fw = floyd_warshall(adj_matrix)
    t_fw = time.perf_counter() - start

    # --- Johnson ---
    start = time.perf_counter()
    dist_j = johnson(NUM_VERTICES, edge_list)
    t_j = time.perf_counter() - start

    # 正确性验证：随机抽取若干点对比较（忽略无穷）
    diffs = 0
    check_pairs = min(500, NUM_VERTICES * NUM_VERTICES)
    samples = random.sample([(i, j) for i in range(NUM_VERTICES) for j in range(NUM_VERTICES) if i != j], check_pairs)
    for i, j in samples:
        if abs(dist_fw[i][j] - dist_j[i][j]) > 1e-6:
            diffs += 1
    if diffs == 0:
        print("结果验证: 两种算法输出一致。")
    else:
        print(f"结果验证: 存在 {diffs} 处差异 (可能由浮点误差导致)。")

    print()
    print("运行时间比较：")
    print(f"  Floyd-Warshall 耗时: {t_fw:.6f} 秒")
    print(f"  Johnson        耗时: {t_j:.6f} 秒")

    if t_fw < t_j:
        winner = "Floyd-Warshall"
        ratio = t_j / t_fw
    else:
        winner = "Johnson"
        ratio = t_fw / t_j
    print(f"  较快算法: {winner} (速度比约 {ratio:.2f}x)")

    # 简要分析
    print()
    print("简析：")
    if EDGE_PROB > 0.7:
        print("  当前为稠密图，Floyd-Warshall 因实现紧凑往往优于需多次Dijkstra的Johnson。")
    else:
        print("  当前为稀疏图，Johnson通过重赋权和Dijkstra堆优化，复杂度接近 O(VE log V)，")
        print("  明显优于 O(V^3) 的Floyd-Warshall。")
    print("  调节 EDGE_PROB 可观察两者性能交叉。")


if __name__ == "__main__":
    compare_algorithms()