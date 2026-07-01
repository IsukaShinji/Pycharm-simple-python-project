# 6.5-7 优先队列实现FIFO队列与栈 (上机2)
import heapq # 借助标准库的最小堆实现优先队列

class FIFOQueue:
    def __init__(self):
        self.pq = []
        self.counter = 0 # 计数器模拟时间戳

    def enqueue(self, val):
        heapq.heappush(self.pq, (self.counter, val)) # 优先级为递增计数器，越早入队key越小
        self.counter += 1

    def dequeue(self):
        if not self.pq: return None
        return heapq.heappop(self.pq)[1] # 弹出key最小的元素，即最早入队的元素

class Stack:
    def __init__(self):
        self.pq = []
        self.counter = 0

    def push(self, val):
        heapq.heappush(self.pq, (-self.counter, val)) # 优先级为负计数器，越晚入队key越小
        self.counter += 1

    def pop(self):
        if not self.pq: return None
        return heapq.heappop(self.pq)[1] # 最小堆弹出负值最小的，即最后入队的元素


if __name__ == '__main__':
    print("=== 上机2：优先队列实现FIFO队列 ===")
    fifo = FIFOQueue()
    fifo.enqueue("任务A")
    fifo.enqueue("任务B")
    fifo.enqueue("任务C")
    print(f"出队顺序: {fifo.dequeue()} -> {fifo.dequeue()} -> {fifo.dequeue()}")

    print("\n=== 上机2：优先队列实现栈(LIFO) ===")
    stack = Stack()
    stack.push("页面1")
    stack.push("页面2")
    stack.push("页面3")
    print(f"出栈顺序: {stack.pop()} -> {stack.pop()} -> {stack.pop()}")