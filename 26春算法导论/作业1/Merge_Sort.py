import random

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # 1. 分 (Divide)
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # 2. & 3. 治与合 (Conquer & Combine)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def print_wrapped(data, prefix="", width=10):
    print(prefix)
    for i in range(0, len(data), width):
        print(" ".join(map(str, data[i:i + width])))

if __name__ == "__main__":
    # 样本：30个左右的随机不同数字
    sample_data = random.sample(range(1, 100), 30)

    print("=" * 40)
    print_wrapped(sample_data, "排序前随机序列 (Sample Size: 30):")

    sorted_data = merge_sort(sample_data)

    print("-" * 40)
    print_wrapped(sorted_data, "归并排序后序列:")
    print("=" * 40)