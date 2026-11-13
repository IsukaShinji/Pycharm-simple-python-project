"""
第八周上机任务 —— 任务 1：注意力机制的基本计算

运行方式：python attention.py
    会依次执行 5 项自检（Test 1 ~ Test 5）和“缩放点积”实验，并保存图片 figures/scaling.png。

记号与课件的对应关系（课件中向量为列向量；代码中一个 batch 的 B 个样本并行计算）：
    课件                                  代码中的张量
    查询向量 q                            query  : [B, Dq]
    键   K = [k_1, ..., k_L]              keys   : [B, L, Dk]   第 b 个样本的第 j 个键为 keys[b, j]
    值   V = [v_1, ..., v_L]              values : [B, L, Dv]
    打分 s(k_j, q)                        scores : [B, L]
    注意力分布 α_j = softmax(s(k_j, q))    weights: [B, L]       每一行之和为 1
    att((K, V), q) = Σ_j α_j v_j          context: [B, Dv]

要求：
    1. 只能使用 torch 的基础张量运算（如 torch.bmm、torch.matmul、广播、F.softmax 等）实现，
       禁止调用 nn.MultiheadAttention、F.scaled_dot_product_attention 等现成的注意力实现；
    2. 不要使用 for 循环遍历 batch 或序列位置；
    3. 不要修改函数名、类名、参数列表和返回值形式（助教会用隐藏测试检查）。
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 打分函数（课件“常见相似度函数 / 打分函数”）
# ============================================================
def dot_score(query, keys):
    """
    点积打分函数：s(k_j, q) = k_j^T q

    参数:
        query: [B, D]
        keys:  [B, L, D]
    返回:
        scores: [B, L]，scores[b, j] = keys[b, j] 与 query[b] 的内积
    """
    # TODO 1: 实现点积打分函数
    # 提示：可以先把 query 变成 [B, D, 1]，再用 torch.bmm 与 keys 相乘
    return torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)


def scaled_dot_score(query, keys):
    """
    缩放点积打分函数：s(k_j, q) = k_j^T q / sqrt(D)，其中 D 为 q 和 k 的维度

    参数:
        query: [B, D]
        keys:  [B, L, D]
    返回:
        scores: [B, L]
    """
    # TODO 2: 实现缩放点积打分函数（可以直接调用 dot_score）
    return dot_score(query, keys) / math.sqrt(query.size(-1))


class AdditiveScore(nn.Module):
    """
    加性打分函数：s(k_j, q) = v^T tanh(W k_j + U q)
    其中 W ∈ R^{H×Dk}、U ∈ R^{H×Dq}、v ∈ R^{H} 都是可学习参数（均不带偏置）。

    用法:
        score_fn = AdditiveScore(key_dim=Dk, query_dim=Dq, hidden_dim=H)
        scores = score_fn(query, keys)   # query: [B, Dq], keys: [B, L, Dk] -> scores: [B, L]
    """

    def __init__(self, key_dim, query_dim, hidden_dim):
        super().__init__()
        # TODO 3: 定义三个不带偏置的线性层，属性名必须分别为 self.W、self.U、self.v
        #   self.W: Dk -> H     self.U: Dq -> H     self.v: H -> 1
        self.W = nn.Linear(key_dim, hidden_dim, bias=False)
        self.U = nn.Linear(query_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        参数:
            query: [B, Dq]
            keys:  [B, L, Dk]
        返回:
            scores: [B, L]
        """
        # TODO 4: 实现 s(k_j, q) = v^T tanh(W k_j + U q)
        # 提示：W k_j 的形状为 [B, L, H]，U q 的形状为 [B, H]，需要先 unsqueeze 成 [B, 1, H] 再与前者广播相加
        key_proj = self.W(keys)
        query_proj = self.U(query).unsqueeze(1)
        return self.v(torch.tanh(key_proj + query_proj)).squeeze(-1)


# ============================================================
# 2. 归一化与加权求和
# ============================================================
def masked_softmax(scores, mask=None):
    """
    带掩码的 softmax，在最后一维（长度 L 这一维）上归一化。

    参数:
        scores: [B, L]
        mask:   None，或形状为 [B, L] 的 bool 张量：True 表示该位置是有效输入，
                False 表示该位置是填充（PAD），不应该被关注。mask=None 表示所有位置都有效。
    返回:
        weights: [B, L]，满足：(1) 每一行之和为 1；(2) mask 为 False 的位置权重严格等于 0。
    约定：每一行至少有一个有效位置。
    """
    # TODO 5: 实现带掩码的 softmax
    # 提示：先用 masked_fill 把无效位置的分数置为 -inf（exp(-inf) = 0），再做 softmax
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
    return F.softmax(scores, dim=-1)


def attention(query, keys, values, score_fn, mask=None, hard=False):
    """
    注意力机制：att((K, V), q) = Σ_j α_j v_j，其中 α_j = softmax_j( s(k_j, q) )

    参数:
        query:    [B, Dq]
        keys:     [B, L, Dk]
        values:   [B, L, Dv]
        score_fn: 打分函数，调用方式为 score_fn(query, keys)，返回 [B, L]；
                  可以是 dot_score、scaled_dot_score，或 AdditiveScore 的一个实例
        mask:     同 masked_softmax
        hard:     False（默认）为软注意力；
                  True 为硬注意力：只选取注意力权重最大的那一个位置，即把 weights 替换成
                  argmax 处为 1、其余位置为 0 的 one-hot 向量，再计算 context（课件“硬注意力机制”）
    返回:
        context: [B, Dv]
        weights: [B, L]（hard=True 时返回的是 one-hot 权重）
    """
    # TODO 6: 按以下步骤实现注意力
    #   (1) 打分：scores = score_fn(query, keys)
    #   (2) 归一化：weights = masked_softmax(scores, mask)
    #   (3) 若 hard=True：weights 替换为 one-hot（提示：argmax + F.one_hot，注意数据类型转换为浮点型）
    #   (4) 加权求和：context = Σ_j weights[:, j] * values[:, j]（提示：torch.bmm）
    scores = score_fn(query, keys)
    weights = masked_softmax(scores, mask)
    if hard:
        indices = weights.argmax(dim=-1)
        weights = F.one_hot(indices, weights.size(-1)).to(weights.dtype)
    context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
    return context, weights


# ============================================================
# 3. 自检（无需修改）
# ============================================================
def _check(name, fn):
    try:
        fn()
        print(f'[PASS] {name}')
        return True
    except NotImplementedError as e:
        print(f'[FAIL] {name}：{e}')
    except Exception as e:  # noqa
        print(f'[FAIL] {name}：{type(e).__name__}: {e}')
    return False


def _test_lecture_example():
    # 课件第 29~30 页的例子：q = (1,1,0)^T，k1=(1,0,0)^T，k2=(0,1,0)^T，k3=(0,0,1)^T，打分函数为点积
    # 这里取 a1=1, a2=2, a3=3，即 v1=(1,1,1)^T，v2=(2,2,2)^T，v3=(3,3,3)^T
    q = torch.tensor([[1., 1., 0.]])                                  # [1, 3]
    keys = torch.eye(3).unsqueeze(0)                                  # [1, 3, 3]
    values = (torch.tensor([1., 2., 3.]).view(3, 1) * torch.ones(3, 3)).unsqueeze(0)  # [1, 3, 3]
    context, weights = attention(q, keys, values, dot_score)
    e = math.e
    expected_w = torch.tensor([[e / (2 * e + 1), e / (2 * e + 1), 1 / (2 * e + 1)]])
    expected_c = expected_w @ torch.tensor([1., 2., 3.])
    assert weights.shape == (1, 3) and context.shape == (1, 3), \
        f'形状错误：weights {tuple(weights.shape)}，context {tuple(context.shape)}'
    assert torch.allclose(weights, expected_w, atol=1e-4), f'weights = {weights}'
    assert torch.allclose(context, expected_c.expand(1, 3), atol=1e-4), f'context = {context}'
    print('       weights =', [round(x, 4) for x in weights[0].tolist()], '（课件中约为 0.42, 0.42, 0.16）')
    print('       context =', [round(x, 4) for x in context[0].tolist()])


def _test_dot_scores():
    g = torch.Generator().manual_seed(1)
    q, k = torch.randn(3, 8, generator=g), torch.randn(3, 5, 8, generator=g)
    s1, s2 = dot_score(q, k), scaled_dot_score(q, k)
    assert s1.shape == (3, 5) and s2.shape == (3, 5), f'形状应为 (3, 5)，实际为 {tuple(s1.shape)} / {tuple(s2.shape)}'
    expected = (k * q.unsqueeze(1)).sum(-1)
    assert torch.allclose(s1, expected, atol=1e-5), 'dot_score 数值错误'
    assert torch.allclose(s2, expected / math.sqrt(8), atol=1e-5), 'scaled_dot_score 数值错误'


def _test_additive_score():
    torch.manual_seed(0)
    m = AdditiveScore(key_dim=6, query_dim=4, hidden_dim=5)
    for name, shape in [('W', (5, 6)), ('U', (5, 4)), ('v', (1, 5))]:
        layer = getattr(m, name, None)
        assert isinstance(layer, nn.Linear), f'self.{name} 应为 nn.Linear'
        assert tuple(layer.weight.shape) == shape, f'self.{name}.weight 形状应为 {shape}'
        assert layer.bias is None, f'self.{name} 不应带偏置'
    q, k = torch.randn(2, 4), torch.randn(2, 3, 6)
    s = m(q, k)
    assert s.shape == (2, 3), f'形状应为 (2, 3)，实际为 {tuple(s.shape)}'
    expected = torch.zeros(2, 3)
    with torch.no_grad():
        for b in range(2):
            for j in range(3):
                expected[b, j] = m.v.weight[0] @ torch.tanh(m.W.weight @ k[b, j] + m.U.weight @ q[b])
    assert torch.allclose(s, expected, atol=1e-5), 'AdditiveScore 数值错误'


def _test_masked_softmax():
    scores = torch.tensor([[1., 2., 3., 4.], [0.5, 0.5, 0.5, 0.5]])
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    w = masked_softmax(scores, mask)
    expected = torch.tensor([[1 / (1 + math.e), math.e / (1 + math.e), 0., 0.], [0.25, 0.25, 0.25, 0.25]])
    assert torch.allclose(w, expected, atol=1e-5), f'结果错误：{w}'
    assert torch.all(w[0, 2:] == 0), '被 mask 的位置权重必须严格为 0'
    assert torch.allclose(masked_softmax(scores), F.softmax(scores, dim=-1)), 'mask=None 时应等价于普通 softmax'


def _test_attention():
    g = torch.Generator().manual_seed(2)
    q, k, v = torch.randn(2, 4, generator=g), torch.randn(2, 6, 4, generator=g), torch.randn(2, 6, 7, generator=g)
    # 软注意力
    c, w = attention(q, k, v, scaled_dot_score)
    assert c.shape == (2, 7) and w.shape == (2, 6), f'形状错误：context {tuple(c.shape)}，weights {tuple(w.shape)}'
    w_ref = F.softmax((k * q.unsqueeze(1)).sum(-1) / 2.0, dim=-1)
    assert torch.allclose(w, w_ref, atol=1e-5), '软注意力权重错误'
    assert torch.allclose(c, (w_ref.unsqueeze(-1) * v).sum(1), atol=1e-5), '软注意力的 context 错误'
    # 带 mask
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
    c_m, w_m = attention(q, k, v, scaled_dot_score, mask=mask)
    assert torch.all(w_m[1, 3:] == 0) and torch.allclose(w_m.sum(-1), torch.ones(2)), '带 mask 的注意力错误'
    # 硬注意力
    c_h, w_h = attention(q, k, v, scaled_dot_score, hard=True)
    idx = w_ref.argmax(-1)
    assert torch.equal(w_h, F.one_hot(idx, 6).to(w_h.dtype)), '硬注意力的权重应为 argmax 处的 one-hot 向量'
    assert torch.allclose(c_h, v[torch.arange(2), idx], atol=1e-6), '硬注意力的 context 应等于被选中位置的 value'


def run_self_tests():
    print('=' * 20, '任务 1 自检', '=' * 20)
    results = [
        _check('Test 1 课件例子（第 29~30 页）', _test_lecture_example),
        _check('Test 2 点积 / 缩放点积打分', _test_dot_scores),
        _check('Test 3 加性打分函数', _test_additive_score),
        _check('Test 4 带掩码的 softmax', _test_masked_softmax),
        _check('Test 5 软注意力 / 硬注意力', _test_attention),
    ]
    print(f'通过 {sum(results)}/{len(results)} 项自检')
    return all(results)


# ============================================================
# 4. 缩放点积实验（无需修改，运行后在报告中分析结果）
# ============================================================
def softmax_jacobian_norm(weights):
    """
    softmax 的雅可比矩阵 ∂α/∂s = diag(α) - α α^T 的 Frobenius 范数。
    它衡量“分数 s 的微小变化能在多大程度上改变注意力权重 α”，即梯度能否顺利地穿过 softmax 回传。
    weights: [N, L] -> 返回 [N]
    """
    jac = torch.diag_embed(weights) - weights.unsqueeze(-1) * weights.unsqueeze(-2)  # [N, L, L]
    return jac.flatten(1).norm(dim=1)


def scaling_experiment(dims=(4, 16, 64, 256, 1024), num_keys=10, num_trials=2000, seed=0,
                       save_path='figures/scaling.png'):
    """
    对每个维度 D：随机采样 num_trials 组 (q, k_1, ..., k_10)，各分量独立服从 N(0, 1)，
    分别用点积和缩放点积打分，统计：
        (1) 分数的方差；
        (2) 注意力权重最大值 max_j α_j 的平均值（越接近 1，分布越接近 one-hot）；
        (3) softmax 雅可比矩阵范数的平均值（越接近 0，梯度越难穿过 softmax 回传）。
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for d in dims:
        q = torch.randn(num_trials, d, generator=g)
        k = torch.randn(num_trials, num_keys, d, generator=g)
        row = {'D': d}
        for name, fn in [('dot', dot_score), ('scaled', scaled_dot_score)]:
            s = fn(q, k)
            w = F.softmax(s, dim=-1)
            row[f'var_{name}'] = s.var().item()
            row[f'maxw_{name}'] = w.max(dim=-1).values.mean().item()
            row[f'jac_{name}'] = softmax_jacobian_norm(w).mean().item()
        rows.append(row)

    print('=' * 20, '缩放点积实验（L = %d 个键，%d 次采样）' % (num_keys, num_trials), '=' * 20)
    print('列说明：var = 分数的方差；max_w = 最大注意力权重的平均值；jac = softmax 雅可比矩阵范数的平均值')
    print(f"{'D':>6} | {'var(dot)':>10} {'var(scaled)':>11} | {'max_w(dot)':>10} {'max_w(scaled)':>13} | "
          f"{'jac(dot)':>9} {'jac(scaled)':>11}")
    for r in rows:
        print(f"{r['D']:>6} | {r['var_dot']:>10.2f} {r['var_scaled']:>11.3f} | {r['maxw_dot']:>10.3f} "
              f"{r['maxw_scaled']:>13.3f} | {r['jac_dot']:>9.4f} {r['jac_scaled']:>11.4f}")

    try:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        ds = [r['D'] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, key, title in [(axes[0], 'var', 'Variance of scores'),
                               (axes[1], 'maxw', 'Mean of max attention weight'),
                               (axes[2], 'jac', 'Mean norm of softmax Jacobian')]:
            ax.plot(ds, [r[f'{key}_dot'] for r in rows], 'o-', color='#2a78d6', linewidth=1.8, label='dot: $k^Tq$')
            ax.plot(ds, [r[f'{key}_scaled'] for r in rows], 's-', color='#eb6834', linewidth=1.8,
                    label=r'scaled dot: $k^Tq/\sqrt{D}$')
            ax.set_xscale('log', base=2)
            if key == 'var':
                ax.set_yscale('log', base=2)
            ax.set_xlabel('dimension D')
            ax.set_title(title)
            ax.grid(color='#e1e0d9', linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f'图像已保存到 {save_path}')
    except ImportError:
        print('未安装 matplotlib，跳过绘图')
    return rows


if __name__ == '__main__':
    ok = run_self_tests()
    print()
    try:
        scaling_experiment()
    except NotImplementedError as e:
        print('缩放点积实验需要先完成 TODO 1 和 TODO 2：', e)
    if not ok:
        print('\n仍有自检未通过，请检查对应的 TODO。')
