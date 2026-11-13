"""
第八周上机任务：汇总多次实验的结果并画对比图（本文件无需修改）

用法：
    python plot_results.py               # 读取 results/ 目录下的所有实验结果
    python plot_results.py --seed 1234   # 只汇总指定 seed 的实验
输出：
    figures/acc_vs_len.png        各模型在不同序列长度下的 token 准确率（任务 2、任务 3）
    figures/training_curves.png   训练 loss 曲线与测试准确率随训练步数的变化（任务 3）
    终端打印汇总表（可直接整理到实验报告中）
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# 每种模型固定使用同一种颜色和标记（颜色跟随模型，而不是跟随画图顺序）
STYLE = {
    'none':                 ('#2a78d6', 'o', 'none (fixed c)'),
    'dot':                  ('#eb6834', 's', 'dot'),
    'scaled_dot':           ('#1baf7a', '^', 'scaled_dot'),
    'additive':             ('#eda100', 'D', 'additive'),
    'scaled_dot_hardtrain': ('#e87ba4', 'v', 'scaled_dot + hard_train'),
    'dot_hardtrain':        ('#008300', 'P', 'dot + hard_train'),
    'additive_hardtrain':   ('#6250d6', 'X', 'additive + hard_train'),
}
ORDER = list(STYLE.keys())
LINESTYLES = ['-', '--', ':', '-.']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'


def model_key(record):
    a = record['args']
    return a['attn'] + ('_hardtrain' if a.get('hard_train') else '')


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED)


def load_records(seed=None):
    records = []
    for path in sorted(glob.glob(os.path.join('results', '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if seed is None or r['args']['seed'] == seed:
            records.append(r)
    records.sort(key=lambda r: (ORDER.index(model_key(r)) if model_key(r) in ORDER else 99, r['args']['seed']))
    return records


def line_style(record, seeds):
    color, marker, name = STYLE.get(model_key(record), ('#898781', 'o', model_key(record)))
    ls = LINESTYLES[seeds.index(record['args']['seed']) % len(LINESTYLES)]
    label = name if len(seeds) == 1 else f"{name} (seed {record['args']['seed']})"
    return dict(color=color, marker=marker, linestyle=ls, label=label)


def plot_acc_vs_len(records, seeds, save_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in records:
        lengths = sorted(int(L) for L in r['final'])
        acc = [r['final'][str(L)]['token_acc'] for L in lengths]
        ax.plot(lengths, acc, linewidth=1.8, markersize=5, **line_style(r, seeds))
    ax.axhline(1 / 26, color='#898781', linestyle=':', linewidth=1.2)
    ax.text(ax.get_xlim()[0] + 0.3, 1 / 26 + 0.015, 'chance level (1/26)', color=MUTED, fontsize=8)
    ax.set_xlabel('sequence length L')
    ax.set_ylabel('test token accuracy')
    ax.set_ylim(0, 1.03)
    ax.set_title('Token accuracy vs. sequence length', color=INK)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(records, seeds, save_path, window=50):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in records:
        st = line_style(r, seeds)
        loss = np.array(r['loss_history'])
        if len(loss) >= window:
            smooth = np.convolve(loss, np.ones(window) / window, mode='valid')
            steps = np.arange(window, len(loss) + 1)
        else:
            smooth, steps = loss, np.arange(1, len(loss) + 1)
        axes[0].plot(steps, smooth, color=st['color'], linestyle=st['linestyle'], linewidth=1.6, label=st['label'])
        ev = r['eval_history']
        axes[1].plot([e['step'] for e in ev], [e['token_acc'] for e in ev], linewidth=1.6, markersize=4, **st)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('training step')
    axes[0].set_ylabel(f'training loss (moving average, window={window})')
    axes[0].set_title('Training loss', color=INK)
    axes[1].set_xlabel('training step')
    axes[1].set_ylabel('test token accuracy (all lengths)')
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title('Test accuracy during training', color=INK)
    for ax in axes:
        style_axes(ax)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def first_step_reaching(record, key='seq_acc', threshold=0.9):
    for e in record['eval_history']:
        if e[key] >= threshold:
            return e['step']
    return None


def print_table(records):
    header = (f"{'model':<28}{'seed':>6}{'params':>9}{'token_acc':>11}{'seq_acc':>9}"
              f"{'L=5':>7}{'L=10':>7}{'L=20':>7}{'L=30':>7}{'align@30':>10}{'max_w@30':>10}{'step(seq>=0.9)':>16}")
    print(header)
    print('-' * len(header))
    for r in records:
        f = r['final']
        acc = [f.get(str(L), {}).get('token_acc', float('nan')) for L in (5, 10, 20, 30)]
        align = r.get('alignment_acc', {}).get('30')
        maxw = r.get('mean_max_weight', {}).get('30')
        reach = first_step_reaching(r)
        print(f"{model_key(r):<28}{r['args']['seed']:>6}{r['num_params']:>9}{r['final_token_acc']:>11.4f}"
              f"{r['final_seq_acc']:>9.4f}" + ''.join(f'{a:>7.3f}' for a in acc) +
              (f'{align:>10.3f}' if align is not None else f"{'-':>10}") +
              (f'{maxw:>10.3f}' if maxw is not None else f"{'-':>10}") +
              (f'{reach:>16}' if reach is not None else f"{'not reached':>16}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='只汇总指定 seed 的实验')
    args = parser.parse_args()

    records = load_records(args.seed)
    if not records:
        print('results/ 目录下没有找到实验结果，请先运行 train.py')
        return
    seeds = sorted({r['args']['seed'] for r in records})
    os.makedirs('figures', exist_ok=True)
    plot_acc_vs_len(records, seeds, os.path.join('figures', 'acc_vs_len.png'))
    plot_training_curves(records, seeds, os.path.join('figures', 'training_curves.png'))
    print_table(records)
    print('\n图像已保存到 figures/acc_vs_len.png 和 figures/training_curves.png')


if __name__ == '__main__':
    main()
