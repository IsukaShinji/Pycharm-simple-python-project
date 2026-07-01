import heapq
import random
import string

# 配置参数
STR_LEN = 30  # 随机生成的字符串长度
ALPHABET = string.ascii_uppercase[:8]  # 备选字符范围（前8个大写字母）


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char  # 字符
        self.freq = freq  # 频率
        self.left = None  # 左子
        self.right = None  # 右子

    def __lt__(self, other):
        return self.freq < other.freq  # 堆排序优先级定义


def build_tree(freq_map):
    heap = [HuffmanNode(c, f) for c, f in freq_map.items()]
    heapq.heapify(heap)  # 建立最小堆

    while len(heap) > 1:
        node1 = heapq.heappop(heap)  # 弹出最小节点
        node2 = heapq.heappop(heap)  # 弹出次小节点
        merged = HuffmanNode(None, node1.freq + node2.freq)  # 合并节点
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)  # 重新入堆
    return heap[0]


def get_codes(root, current_code, code_book):
    if not root: return
    if root.char:  # 到达叶子节点
        code_book[root.char] = current_code
        return
    get_codes(root.left, current_code + "0", code_book)  # 遍历左子树
    get_codes(root.right, current_code + "1", code_book)  # 遍历右子树


def draw_tree(node, level=0, prefix="Root: "):
    if node:
        print("  " * level + prefix + (f"[{node.char}:{node.freq}]" if node.char else f"({node.freq})"))
        if node.left or node.right:
            draw_tree(node.left, level + 1, "L-- ")
            draw_tree(node.right, level + 1, "R-- ")


def main():
    # 生成随机输入
    source_str = ''.join(random.choice(ALPHABET) for _ in range(STR_LEN))
    freq_map = {c: source_str.count(c) for c in set(source_str)}

    print(f"Original String: {source_str}")
    print(f"Frequencies: {freq_map}\n")

    # 构建并解析
    root = build_tree(freq_map)
    code_book = {}
    get_codes(root, "", code_book)

    # 输出结果
    print("Huffman Tree Structure:")
    draw_tree(root)

    print("\nCharacter Encodings:")
    for char in sorted(code_book.keys()):
        print(f"Char: {char} | Code: {code_book[char]}")

    encoded_str = "".join(code_book[c] for c in source_str)
    print(f"\nOriginal length: {len(source_str) * 8} bits")
    print(f"Compressed length: {len(encoded_str)} bits")


if __name__ == "__main__":
    main()