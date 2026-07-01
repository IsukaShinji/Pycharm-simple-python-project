import sys
sys.setrecursionlimit(1 << 25)

class TwoSAT:
    def __init__(self, n):
        """
        n: 变量个数，变量编号从 1 到 n
        内部节点索引：
            2*i     : 表示变量 i 为 False（即 ¬x_i 为 True）
            2*i + 1 : 表示变量 i 为 True （即 x_i 为 True）
        """
        self.n = n
        self.N = 2 * n
        self.g = [[] for _ in range(self.N)]
        self.rg = [[] for _ in range(self.N)]

    def _lit_node(self, lit):
        """将文字（整数）转换为节点索引"""
        var = abs(lit) - 1
        if lit > 0:          # x_var 为 True
            return 2 * var + 1
        else:                # ¬x_var 为 True，即 x_var 为 False
            return 2 * var

    def add_clause(self, a, b):
        """
        添加子句 (a ∨ b)
        a, b 为非零整数，正数表示 x_i，负数表示 ¬x_i
        """
        na = self._lit_node(-a)   # ¬a 对应的节点
        nb = self._lit_node(-b)   # ¬b 对应的节点
        pa = self._lit_node(a)    # a 对应的节点
        pb = self._lit_node(b)    # b 对应的节点

        # (¬a -> b) 和 (¬b -> a)
        self.g[na].append(pb)
        self.rg[pb].append(na)

        self.g[nb].append(pa)
        self.rg[pa].append(nb)

    def _dfs1(self, u):
        self.visited[u] = True
        for v in self.g[u]:
            if not self.visited[v]:
                self._dfs1(v)
        self.order.append(u)

    def _dfs2(self, u, cid):
        self.comp[u] = cid
        for v in self.rg[u]:
            if self.comp[v] == -1:
                self._dfs2(v, cid)

    def satisfiable(self):
        """判断是否可满足，若可满足返回 (True, assignment)，否则返回 (False, None)"""
        self.visited = [False] * self.N
        self.order = []
        for i in range(self.N):
            if not self.visited[i]:
                self._dfs1(i)

        self.comp = [-1] * self.N
        cid = 0
        for u in reversed(self.order):
            if self.comp[u] == -1:
                self._dfs2(u, cid)
                cid += 1

        # 检查每个变量是否矛盾
        for i in range(self.n):
            false_node = 2 * i       # x_i 为 False
            true_node = 2 * i + 1    # x_i 为 True
            if self.comp[false_node] == self.comp[true_node]:
                return False, None

        # 构造一组满足赋值
        assignment = [False] * self.n
        for i in range(self.n):
            false_node = 2 * i
            true_node = 2 * i + 1
            # 若 True 节点的 SCC 编号大于 False 节点，则令 x_i = True
            assignment[i] = self.comp[true_node] > self.comp[false_node]
        return True, assignment


# ========== 测试样例 ==========
def test_2sat():
    # 样例1：可满足 (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2)
    sat = TwoSAT(2)
    sat.add_clause(1, 2)      # (x1 ∨ x2)
    sat.add_clause(-1, 2)     # (¬x1 ∨ x2)
    sat.add_clause(1, -2)     # (x1 ∨ ¬x2)
    ok, assign = sat.satisfiable()
    print("样例1 可满足?", ok)
    if ok:
        print("赋值：x1 =", assign[0], ", x2 =", assign[1])

    # 样例2：不可满足 (x1) ∧ (¬x1)
    sat2 = TwoSAT(1)
    sat2.add_clause(1, 1)     # (x1 ∨ x1) 即 x1
    sat2.add_clause(-1, -1)   # (¬x1 ∨ ¬x1) 即 ¬x1
    ok2, assign2 = sat2.satisfiable()
    print("\n样例2 可满足?", ok2)

    # 样例3：可满足 (x1 ∨ x2) ∧ (¬x1 ∨ ¬x2)
    sat3 = TwoSAT(2)
    sat3.add_clause(1, 2)
    sat3.add_clause(-1, -2)
    ok3, assign3 = sat3.satisfiable()
    print("\n样例3 可满足?", ok3)
    if ok3:
        print("赋值：x1 =", assign3[0], ", x2 =", assign3[1])

if __name__ == "__main__":
    test_2sat()