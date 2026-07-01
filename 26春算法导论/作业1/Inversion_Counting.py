import random

def count_inversions(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    # 递归计算左半部分、右半部分以及跨越两部分的逆序对
    left, left_count = count_inversions(arr[:mid])
    right, right_count = count_inversions(arr[mid:])
    merged, cross_count = merge_and_count(left, right)

    return merged, left_count + right_count + cross_count


def merge_and_count(left, right):
    result = []
    count = 0
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            # 当右边元素小于左边元素时，左边剩余的所有元素都与该右边元素构成逆序对
            result.append(right[j])
            count += (len(left) - i)
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result, count


def print_wrapped(data, prefix="", width=10):
    print(prefix)
    for i in range(0, len(data), width):
        print(" ".join(map(str, data[i:i + width])))

if __name__ == "__main__":
    sample_data = random.sample(range(1, 100), 25)

    print("=" * 40)
    print_wrapped(sample_data, "输入序列:")

    _, total_inversions = count_inversions(sample_data)

    print("-" * 40)
    print(f"该序列中的逆序对总数为: {total_inversions}")
    print("=" * 40)