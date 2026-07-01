# 最小优先队列实现 (上机1)
class MinPriorityQueue:
    def __init__(self):
        self.heap = []

    def parent(self, i): return (i - 1) // 2 # 获取父节点索引
    def left(self, i): return 2 * i + 1      # 获取左子节点索引
    def right(self, i): return 2 * i + 2     # 获取右子节点索引

    def min_heapify(self, i):
        l, r = self.left(i), self.right(i)
        smallest = i
        if l < len(self.heap) and self.heap[l] < self.heap[smallest]: smallest = l
        if r < len(self.heap) and self.heap[r] < self.heap[smallest]: smallest = r
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i] # 交换节点
            self.min_heapify(smallest) # 递归维护最小堆性质

    def minimum(self):
        return self.heap[0] if self.heap else None # 返回堆顶即最小值

    def extract_min(self):
        if not self.heap: return None
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1] # 将末尾元素移至堆顶
        self.heap.pop()
        if self.heap: self.min_heapify(0) # 重新调整堆
        return min_val

    def decrease_key(self, i, key):
        if key > self.heap[i]: return # 新值必须小于当前值
        self.heap[i] = key
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]: # 向上冒泡调整
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)

    def insert(self, key):
        self.heap.append(float('inf')) # 尾部插入无穷大
        self.decrease_key(len(self.heap) - 1, key) # 更新尾部节点为实际key并调整


if __name__ == '__main__':
    print("=== 上机1：最小优先队列测试 ===")
    pq = MinPriorityQueue()

    # 测试插入
    elements = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
    print(f"正在插入元素: {elements}")
    for el in elements:
        pq.insert(el)

    print(f"建堆后的底层数组: {pq.heap}")
    print(f"当前最小值 (MINIMUM): {pq.minimum()}")

    # 测试提取最小值
    print(f"提取出的最小值 (EXTRACT-MIN): {pq.extract_min()}")
    print(f"提取后的底层数组: {pq.heap}")

    # 测试减小键值
    print("\n测试 DECREASE-KEY: 将索引 4 的值(当前为10)减小为 0")
    pq.decrease_key(4, 0)
    print(f"调整后的底层数组: {pq.heap}")