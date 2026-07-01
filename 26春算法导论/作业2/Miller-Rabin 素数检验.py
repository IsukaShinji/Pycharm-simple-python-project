import random
import math

def is_prime(n, test_rounds=5):
    # 边界情况快速判断
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # 将n-1分解为d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # 多轮随机基检验
    for _ in range(test_rounds):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        # 平方迭代s-1次
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            # 未找到n-1，确定为合数
            return False
    return True

def factorize(n):
    factors = []
    # 先分解2
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    # 再分解奇数，从3到sqrt(n)
    i = 3
    max_factor = math.isqrt(n) + 1
    while i <= max_factor and n > 1:
        while n % i == 0:
            factors.append(i)
            n = n // i
            max_factor = math.isqrt(n) + 1
        i += 2
    if n > 1:
        factors.append(n)
    return factors

def get_factor_combinations(factors):
    # 生成不同的因数组合（示例：两个因数相乘的形式）
    combinations = []
    original_num = 1
    for f in factors:
        original_num *= f
    # 找因数对a*b=original_num，a<=b且a≠b
    for a in range(2, math.isqrt(original_num) + 1):
        if original_num % a == 0:
            b = original_num // a
            if a != b:
                combinations.append(f"{a} × {b}")
            if len(combinations) >= 5:
                break
    return combinations

if __name__ == "__main__":
    print("=== 素数检验与合数分解工具 ===")
    print("说明：请输入一个正整数（建议不超过100位，超过会提示溢出）")

    while True:
        num_input = input("请输入待检验的正整数：").strip()
        # 检查负数
        if num_input.startswith('-'):
            print("请输入正整数，不要输入负数。")
            continue
        # 检查纯数字
        if not num_input.isdigit():
            print("输入无效，请输入纯数字。")
            continue
        # 检查位数（模拟“溢出”提示）
        if len(num_input) > 100:
            print("输入数字过大，超出处理范围（建议不超过100位）。")
            continue
        n = int(num_input)
        break

    # 素性检验
    if is_prime(n):
        print(f"\n结果：{n} 是素数。")
    else:
        print(f"\n结果：{n} 是合数。")
        # 分解质因数
        factors = factorize(n)
        print(f"质因数分解：{' × '.join(map(str, factors))}")
        # 生成不同的因数组合
        combinations = get_factor_combinations(factors)
        if combinations:
            print("不同的因数组合（示例）：")
            for combo in combinations:
                print(f"  {combo}")