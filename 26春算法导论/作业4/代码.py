import random


# ==========================================
# 算法 1：期望线性时间的随机选择 (Randomized-Select)
# ==========================================
def randomized_partition(A, p, r):
    i = random.randint(p, r)
    A[r], A[i] = A[i], A[r]
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


def randomized_select(A, p, r, i):
    if p == r:
        return A[p]
    q = randomized_partition(A, p, r)
    k = q - p + 1
    if i == k:
        return A[q]
    elif i < k:
        return randomized_select(A, p, q - 1, i)
    else:
        return randomized_select(A, q + 1, r, i - k)


def find_kth_largest_randomized(A, k):
    """求第k大元素的对外接口 (Randomized)"""
    n = len(A)
    # 第k大即为第 n-k+1 小
    return randomized_select(A, 0, n - 1, n - k + 1)


# ==========================================
# 算法 2：最坏情况线性时间的选择 (Worst-Case Linear Select)
# ==========================================
def partition_by_value(A, p, r, pivot_val):
    # 找到主元并换到末尾
    for j in range(p, r + 1):
        if A[j] == pivot_val:
            A[j], A[r] = A[r], A[j]
            break
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


def insertion_sort(A, p, r):
    for i in range(p + 1, r + 1):
        key = A[i]
        j = i - 1
        while j >= p and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key


def worst_case_select(A, p, r, i):
    if r - p <= 140:
        insertion_sort(A, p, r)
        return A[p + i - 1]

    n = r - p + 1
    groups = (n + 4) // 5  # 分成5个一组

    for j in range(groups):
        start = p + 5 * j
        end = min(p + 5 * j + 4, r)
        insertion_sort(A, start, end)
        median_idx = start + (end - start) // 2
        A[p + j], A[median_idx] = A[median_idx], A[p + j]

    x = worst_case_select(A, p, p + groups - 1, (groups + 1) // 2)

    q = partition_by_value(A, p, r, x)
    k = q - p + 1

    if i == k:
        return A[q]
    elif i < k:
        return worst_case_select(A, p, q - 1, i)
    else:
        return worst_case_select(A, q + 1, r, i - k)


def find_kth_largest_worst_case(A, k):
    """求第k大元素的对外接口 (Worst-Case)"""
    n = len(A)
    # 第k大即为第 n-k+1 小
    return worst_case_select(A, 0, n - 1, n - k + 1)


# 当作脚本独立运行时，执行一次简单的功能演示
if __name__ == "__main__":
    test_arr = [3, 2, 9, 1, 8, 5, 7, 4, 6]
    k = 3
    print("--- 代码.py 本地测试 ---")
    print(f"原数组: {test_arr}")
    print(f"寻找第 {k} 大的元素...")

    # 注意：算法会在原地修改数组，需要传拷贝
    res_rand = find_kth_largest_randomized(test_arr.copy(), k)
    res_worst = find_kth_largest_worst_case(test_arr.copy(), k)

    print(f"Randomized-Select 结果: {res_rand}")
    print(f"Worst-Case Select 结果: {res_worst}")