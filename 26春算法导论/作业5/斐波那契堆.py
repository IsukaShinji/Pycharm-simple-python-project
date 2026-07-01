import random
import math

NUM_ELEMENTS = 12  # 初始随机入堆的元素个数
RANDOM_MIN = 1  # 随机生成的最小数值
RANDOM_MAX = 100  # 随机生成的最大数值
RANDOM_SEED = 42  # 随机数种子 (设为 None 则每次运行结果不同)
EXTRACT_COUNT = 1  # 测试时执行 Extract-Min (弹出最小值) 的次数



class FibonacciNode:
    """斐波那契堆的节点定义"""

    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.parent = None
        self.child = None
        # 初始时左右指针都指向自己，形成仅包含自身的循环链表
        self.left = self
        self.right = self
        self.mark = False


class FibonacciHeap:
    """斐波那契堆实现"""

    def __init__(self):
        self.min_node = None
        self.total_nodes = 0

    def insert(self, key):
        """插入新元素，直接加入根链表"""
        new_node = FibonacciNode(key)
        if self.min_node is None:
            self.min_node = new_node
        else:
            self._add_to_root_list(new_node)
            if new_node.key < self.min_node.key:
                self.min_node = new_node
        self.total_nodes += 1
        return new_node

    def extract_min(self):
        """弹出并返回最小节点，随后合并相同度数的树"""
        z = self.min_node
        if z is not None:
            # 将最小节点的所有子节点提升到根链表中
            if z.child is not None:
                # 遍历子节点循环链表，将其加入根链表
                children = []
                curr = z.child
                while True:
                    children.append(curr)
                    curr = curr.right
                    if curr == z.child:
                        break
                for child in children:
                    self._add_to_root_list(child)
                    child.parent = None

            # 从根链表中移除 z
            self._remove_from_root_list(z)

            if z == z.right:
                self.min_node = None
            else:
                self.min_node = z.right
                self._consolidate()
            self.total_nodes -= 1
        return z

    def _consolidate(self):
        """合并根链表中度数相同的树"""
        # 计算最大可能的度数
        max_degree = int(math.log2(self.total_nodes) * 1.618) + 2 if self.total_nodes > 0 else 0
        A = [None] * max_degree

        # 将根链表的所有节点暂存到一个普通列表中，防止在遍历时修改指针导致死循环
        roots = []
        curr = self.min_node
        if curr is not None:
            while True:
                roots.append(curr)
                curr = curr.right
                if curr == self.min_node:
                    break

        # 遍历所有根节点进行合并
        for w in roots:
            x = w
            d = x.degree
            while A[d] is not None:
                y = A[d]  # y 是一棵具有相同度数的树
                if x.key > y.key:
                    x, y = y, x  # 确保 x 的键值更小，y 成为 x 的孩子
                self._link(y, x)
                A[d] = None
                d += 1
            A[d] = x

        # 重建根链表并寻找新的最小节点
        self.min_node = None
        for i in range(max_degree):
            if A[i] is not None:
                # 重置左右指针为自身，准备重新加入根链表
                A[i].left = A[i]
                A[i].right = A[i]

                if self.min_node is None:
                    self.min_node = A[i]
                else:
                    self._add_to_root_list(A[i])
                    if A[i].key < self.min_node.key:
                        self.min_node = A[i]

    def _link(self, y, x):
        """将 y 链接为 x 的孩子"""
        self._remove_from_root_list(y)
        y.left = y
        y.right = y
        y.parent = x

        if x.child is None:
            x.child = y
        else:
            y.left = x.child
            y.right = x.child.right
            x.child.right.left = y
            x.child.right = y

        x.degree += 1
        y.mark = False

    def _add_to_root_list(self, node):
        """辅助方法：将节点插入到 min_node 的右侧"""
        node.left = self.min_node
        node.right = self.min_node.right
        self.min_node.right.left = node
        self.min_node.right = node

    def _remove_from_root_list(self, node):
        """辅助方法：从它所在的循环链表中移除自己"""
        node.left.right = node.right
        node.right.left = node.left


    def print_tree(self, title="当前斐波那契堆结构"):
        print(f"\n--- {title} (总节点数: {self.total_nodes}) ---")
        if self.min_node is None:
            print("堆为空")
            return

        curr = self.min_node
        while True:
            is_min = (curr == self.min_node)
            mark = " <-- [MIN]" if is_min else ""
            print(f"[{curr.key}]{mark}")
            if curr.child is not None:
                self._print_child(curr.child, "   ")
            curr = curr.right
            if curr == self.min_node:
                break
        print("-" * 40)

    def _print_child(self, start_node, prefix):
        curr = start_node
        while True:
            print(f"{prefix}|-- [{curr.key}]")
            if curr.child is not None:
                self._print_child(curr.child, prefix + "|   ")
            curr = curr.right
            if curr == start_node:
                break


if __name__ == "__main__":
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    heap = FibonacciHeap()

    print(f"开始生成并插入 {NUM_ELEMENTS} 个随机数...")
    # 1. 随机生成数据并入堆
    for _ in range(NUM_ELEMENTS):
        val = random.randint(RANDOM_MIN, RANDOM_MAX)
        heap.insert(val)

    # 此时仅执行了 Insert，根据斐波那契堆的惰性策略，所有节点都在根链表上
    heap.print_tree("初始入堆完毕 (尚未触发合并)")

    # 2. 执行 Extract-Min 触发合并 (Consolidate)
    for i in range(EXTRACT_COUNT):
        min_val = heap.extract_min()
        print(f"\n>> 执行 Extract-Min 操作，弹出的最小值为: {min_val.key}")
        heap.print_tree(f"第 {i + 1} 次 Extract-Min 后的结构 (触发树合并)")