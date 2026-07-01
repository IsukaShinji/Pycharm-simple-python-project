import random
import time
import sys

sys.setrecursionlimit(20000)

# 从同目录下的 代码.py 导入我们需要测试的函数
try:
    from 代码 import find_kth_largest_randomized, find_kth_largest_worst_case
except ImportError:
    print("导入失败，请确保 '代码.py' 与 '评测.py' 在同一个文件夹下。")
    sys.exit(1)


def check_correctness():
    print("================ 正确性验证 ================")
    for i in range(3):
        # 生成随机数组
        size = random.randint(10, 20)
        A = [random.randint(1, 100) for _ in range(size)]
        k = random.randint(1, size)

        # 使用 Python 内置排序获取标准答案
        sorted_A = sorted(A, reverse=True)
        correct_answer = sorted_A[k - 1]

        # 测试算法 (传副本防止数组被修改污染后续测试)
        ans_rand = find_kth_largest_randomized(A.copy(), k)
        ans_worst = find_kth_largest_worst_case(A.copy(), k)

        print(f"测试 {i + 1} (长度={size}, 找第{k}大):")
        print(f"  标准答案: {correct_answer}")
        print(f"  Randomized-Select: {ans_rand} -> {'通过' if ans_rand == correct_answer else '失败'}")
        print(f"  Worst-Case Select: {ans_worst} -> {'通过' if ans_worst == correct_answer else '失败'}")
        print("-" * 40)


def check_efficiency():
    print("\n================ 效率对比 ================")
    test_sizes = [10000, 50000, 100000]

    for size in test_sizes:
        print(f"正在生成 {size} 规模的随机数组...")
        A = [random.randint(1, size * 10) for _ in range(size)]
        k = size // 2  # 统一寻找中位数进行基准测试
        print(f"-> 寻找第 {k} 大元素 (中位数)")

        A_copy1 = A.copy()
        A_copy2 = A.copy()

        # 测试 Randomized-Select
        start_time = time.time()
        find_kth_largest_randomized(A_copy1, k)
        time_rand = time.time() - start_time

        # 测试 Worst-Case Linear Select
        start_time = time.time()
        find_kth_largest_worst_case(A_copy2, k)
        time_worst = time.time() - start_time

        print(f"  Randomized-Select 耗时: {time_rand:.4f} 秒")
        print(f"  Worst-Case Select 耗时: {time_worst:.4f} 秒")
        print("-" * 40)


if __name__ == "__main__":
    check_correctness()
    check_efficiency()