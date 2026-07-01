import random
def sample_without_replacement(population, k):
    n = len(population)
    if k < 0 or k > n:
        return None

    arr = population.copy()
    # Fisher-Yates 随机洗牌
    for i in range(n):
        j = random.randint(i, n - 1)
        arr[i], arr[j] = arr[j], arr[i]

    return arr[:k]

# 自定义输入部分
print("=== 无放回抽样工具 ===")
print("说明：请先输入总体元素，用空格分隔；再输入要抽取的样本数量")
print("示例：总体输入 'a b c d'，样本数输入 '2'")

# 获取总体输入
while True:
    pop_input = input("\n请输入总体元素（空格分隔）：").strip()
    if pop_input:
        population = pop_input.split()
        break
    print("输入不能为空，请重新输入")

# 获取样本数输入
while True:
    k_input = input(f"请输入抽取数量（1到{len(population)}之间）：").strip()
    if k_input.isdigit():
        k = int(k_input)
        if 1 <= k <= len(population):
            break
        print(f"数量需在1到{len(population)}之间，请重新输入")
    else:
        print("请输入有效的正整数")

# 执行抽样
result = sample_without_replacement(population, k)
print(f"\n抽样结果：{' '.join(result)}")