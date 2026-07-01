class TarjanSCC:
    def __init__(self, graph):
        """
        :param graph: 邻接表表示的有向图，例如 {0: [1,2], 1: [2], 2: [0], 3: [4], 4: [3]}
        """
        self.graph = graph
        self.n = len(graph)
        self.index = 0                 # 全局时间戳
        self.dfn = [-1] * self.n       # 节点发现时间（访问序号），-1 表示未访问
        self.low = [0] * self.n        # 能回溯到的最早栈内节点序号
        self.on_stack = [False] * self.n
        self.stack = []
        self.sccs = []                 # 存储所有 SCC，每个 SCC 是一个节点列表

        # 用于打印 DFS 树形结构
        self.depth = 0

    def dfs(self, u):
        """递归 DFS，计算 low 值并找出 SCC"""
        self.dfn[u] = self.low[u] = self.index
        self.index += 1
        self.stack.append(u)
        self.on_stack[u] = True

        # 打印树状图（缩进表示深度）
        print("  " * self.depth + f"访问节点 {u} (dfn={self.dfn[u]}, low={self.low[u]})")
        self.depth += 1

        for v in self.graph.get(u, []):
            if self.dfn[v] == -1:      # v 未被访问，树边
                self.dfs(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.on_stack[v]:     # v 在栈中，回边或横叉边（但只处理栈中的）
                self.low[u] = min(self.low[u], self.dfn[v])

        # 如果是 SCC 的根节点，弹出栈中节点形成一个 SCC
        if self.dfn[u] == self.low[u]:
            scc = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                scc.append(w)
                if w == u:
                    break
            self.sccs.append(scc)
            print("  " * (self.depth - 1) + f"  └── 发现 SCC: {scc}")

        self.depth -= 1

    def find_sccs(self):
        """对每个未访问节点启动 DFS"""
        for node in range(self.n):
            if self.dfn[node] == -1:
                print(f"\n从节点 {node} 开始新的 DFS 树：")
                self.dfs(node)
        return self.sccs


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 构造一个有向图（常见例子）
    # 节点 0,1,2 构成一个强连通分量；节点 3,4 构成另一个；节点 5 单独一个
    graph = {
        0: [1],
        1: [2, 3],
        2: [0],
        3: [4],
        4: [3],
        5: []          # 孤立节点
    }

    print("图结构（邻接表）：")
    for u, neighbors in graph.items():
        print(f"  {u} -> {neighbors}")
    print("\n开始 Tarjan DFS 搜索：")

    tarjan = TarjanSCC(graph)
    sccs = tarjan.find_sccs()

    print("\n" + "="*40)
    print("强连通分量结果：")
    for idx, comp in enumerate(sccs):
        print(f"  SCC {idx+1}: {comp}")