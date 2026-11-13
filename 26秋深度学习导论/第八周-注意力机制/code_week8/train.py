"""
第八周上机任务 —— 训练与评估脚本（任务 2、任务 3）

用法（--seed 请设置为本人学号后四位，本次所有实验使用同一个 seed）：
    python train.py --attn none        --seed 1234     # 无注意力基线（固定上下文向量 c）
    python train.py --attn scaled_dot  --seed 1234     # 缩放点积注意力
    python train.py --attn dot         --seed 1234     # 点积注意力（任务 3）
    python train.py --attn additive    --seed 1234     # 加性注意力（任务 3）
    python train.py --attn scaled_dot  --seed 1234 --hard_train    # 用硬注意力训练（任务 3）

运行结束后会生成：
    logs/         训练日志（需要提交）
    results/      实验结果 json，供 plot_results.py 画对比图
    figures/      注意力热力图（仅注意力模型）
    checkpoints/  训练好的模型参数

修改 TODO 8 后，如需重新画热力图，可加 --plot_only 直接载入训练好的模型，无需重新训练：
    python train.py --attn scaled_dot --seed 1234 --plot_only
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ATTN_TYPES, Seq2Seq
from utils import VOCAB_SIZE, build_test_set, decode, get_local_time, make_copy_batch, set_seed

HEATMAP_LENGTHS = (10, 20, 30)  # 画热力图 / 统计对齐准确率时使用的序列长度


@torch.no_grad()
def evaluate(model, test_set, hard=False):
    """
    在测试集上贪心解码（每个长度为 L 的序列解码 L 步），按长度分别统计：
        token_acc：字符级准确率（预测正确的字符数 / 总字符数）
        seq_acc：  整句准确率（整条序列完全正确的比例）
    返回: {L: {'token_acc': float, 'seq_acc': float}}
    """
    model.eval()
    results = {}
    for length, (src, tgt) in test_set.items():
        logits, _ = model(src, max_len=length, hard=hard)
        correct = logits.argmax(dim=-1) == tgt
        results[length] = {'token_acc': correct.float().mean().item(),
                           'seq_acc': correct.all(dim=1).float().mean().item()}
    model.train()
    return results


def summarize(results):
    """对所有长度取平均，返回 (平均 token_acc, 平均 seq_acc)"""
    token_acc = float(np.mean([r['token_acc'] for r in results.values()]))
    seq_acc = float(np.mean([r['seq_acc'] for r in results.values()]))
    return token_acc, seq_acc


def alignment_accuracy(attn):
    """
    对齐准确率（已实现）：复制任务中第 t 个输出字符就是第 t 个输入字符，
    理想情况下第 t 步的注意力应当集中在输入位置 t 上。
    统计“第 t 步注意力权重最大的位置恰好等于 t”的比例。
        attn: [B, T, L]（复制任务中 T = L）
    """
    pred_pos = attn.argmax(dim=-1)                                     # [B, T]
    target_pos = torch.arange(attn.size(1), device=attn.device).unsqueeze(0)  # [1, T]
    return (pred_pos == target_pos).float().mean().item()


def plot_attention_heatmap(ax, weights, src_text, out_text, title):
    """
    在给定的子图 ax 上画一张注意力热力图。

    参数:
        ax:       matplotlib 的 Axes 对象（一个子图）
        weights:  形状为 [T, L] 的 numpy 数组，weights[t, j] 是第 t 个输出字符对第 j 个输入字符的注意力权重
        src_text: 长度为 L 的输入字符串
        out_text: 长度为 T 的模型输出字符串
        title:    子图标题
    要求:
        (1) 横轴为输入字符（键），纵轴为输出字符（查询），每个刻度都要标出对应的字符；
        (2) 使用 ax.imshow 绘制，颜色映射使用单一色系（如 cmap='Blues'），颜色范围固定为 vmin=0、vmax=1
            （便于比较不同模型），并为该子图添加颜色条（colorbar）；
        (3) 设置横、纵坐标轴名称（如 'input (keys)'、'output (queries)'）和标题。
    """
    # TODO 8: 完成注意力热力图的绘制
    # 提示：ax.imshow(...)、ax.set_xticks / ax.set_xticklabels、ax.set_yticks / ax.set_yticklabels、
    #       ax.figure.colorbar(im, ax=ax)、ax.set_xlabel / ax.set_ylabel / ax.set_title
    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(src_text)))
    ax.set_xticklabels(list(src_text))
    ax.set_yticks(range(len(out_text)))
    ax.set_yticklabels(list(out_text))
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('input (keys)')
    ax.set_ylabel('output (queries)')
    ax.set_title(title)


@torch.no_grad()
def draw_heatmaps(model, test_set, save_path, tag):
    """对 L = 10、20、30 各取测试集中的第 1 条序列，画注意力热力图（已实现，内部调用 TODO 8）"""
    import matplotlib.pyplot as plt
    model.eval()
    lengths = [L for L in HEATMAP_LENGTHS if L in test_set]
    fig, axes = plt.subplots(1, len(lengths), figsize=(5.6 * len(lengths), 5.4))
    axes = np.atleast_1d(axes)
    for ax, length in zip(axes, lengths):
        src = test_set[length][0][:1]                     # [1, L]
        logits, attn = model(src, max_len=length)
        out_text = decode(logits.argmax(dim=-1)[0].tolist())
        plot_attention_heatmap(ax, attn[0].cpu().numpy(), decode(src[0].tolist()), out_text,
                               title=f'L = {length}')
    note = ' - soft weights before one-hot' if 'hardtrain' in tag else ''
    fig.suptitle(f'Attention weights ({tag}{note})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    model.train()


def log(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn', type=str, default='scaled_dot', choices=ATTN_TYPES,
                        help="注意力类型：none / dot / scaled_dot / additive")
    parser.add_argument('--seed', type=int, default=1234, help='随机种子，请设置为本人学号后四位')
    parser.add_argument('--hard_train', action='store_true', help='训练和测试时都使用硬注意力（任务 3）')
    parser.add_argument('--plot_only', action='store_true', help='不训练，载入已训练好的模型重新绘制热力图')
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=4000, help='训练步数（每步一个 batch）')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率（训练到 80% 时降为 0.1 倍）')
    parser.add_argument('--clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--eval_interval', type=int, default=250)
    args = parser.parse_args()

    if args.hard_train and args.attn == 'none':
        parser.error('--hard_train 只能与注意力模型一起使用')

    for d in ('logs', 'results', 'figures', 'checkpoints'):
        os.makedirs(d, exist_ok=True)
    tag = f"{args.attn}{'_hardtrain' if args.hard_train else ''}_seed{args.seed}"
    ckpt_path = os.path.join('checkpoints', f'{tag}.pt')
    heatmap_path = os.path.join('figures', f'heatmap_{tag}.png')

    set_seed(args.seed)
    test_set = build_test_set(args.min_len, args.max_len)
    model = Seq2Seq(VOCAB_SIZE, args.emb_dim, args.hidden_dim, args.attn)

    if args.plot_only:  # 只重新画热力图
        if args.attn == 'none':
            parser.error('无注意力模型（--attn none）没有注意力权重，无法画热力图')
        if not os.path.exists(ckpt_path):
            parser.error(f'找不到 {ckpt_path}，请先训练该模型')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        draw_heatmaps(model, test_set, heatmap_path, tag)
        print(f'注意力热力图已保存到 {heatmap_path}')
        return

    log_file = os.path.join('logs', f'week8-{tag}-{get_local_time()}.txt')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 训练进行到 80% 时学习率降为原来的 0.1 倍，使训练末期的结果更稳定
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.8 * args.steps)], gamma=0.1)
    data_gen = torch.Generator().manual_seed(args.seed)  # 训练数据的随机数生成器
    len_rng = random.Random(args.seed)                    # 每个 batch 的序列长度

    log(str(vars(args)), log_file)
    log(f'模型可训练参数量: {num_params}', log_file)

    loss_history, eval_history = [], []
    start = time.time()
    for step in range(1, args.steps + 1):
        # 每个 batch 随机选一个长度 L，batch 内所有序列长度相同，因此不需要 padding
        length = len_rng.randint(args.min_len, args.max_len)
        src, tgt = make_copy_batch(args.batch_size, length, data_gen)

        logits, _ = model(src, tgt, hard=args.hard_train)  # teacher forcing
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if step % args.eval_interval == 0 or step == args.steps:
            token_acc, seq_acc = summarize(evaluate(model, test_set, hard=args.hard_train))
            eval_history.append({'step': step, 'token_acc': token_acc, 'seq_acc': seq_acc})
            avg_loss = np.mean(loss_history[-args.eval_interval:])
            log(f'step {step:5d}/{args.steps} | train loss {avg_loss:.4f} | '
                f'test token_acc {token_acc:.4f} | test seq_acc {seq_acc:.4f} | '
                f'time {time.time() - start:.0f}s', log_file)

    # ---------------- 最终评估（hard_train 模型在测试时同样使用硬注意力） ----------------
    final = evaluate(model, test_set, hard=args.hard_train)
    token_acc, seq_acc = summarize(final)
    log(f'[最终结果] 平均 token_acc {token_acc:.4f} | 平均 seq_acc {seq_acc:.4f}', log_file)
    log('[各长度 token_acc] ' + ' '.join(f'{L}:{r["token_acc"]:.3f}' for L, r in final.items()), log_file)

    record = {'tag': tag, 'args': vars(args), 'num_params': num_params,
              'train_time_sec': time.time() - start, 'loss_history': loss_history,
              'eval_history': eval_history, 'final': {str(L): r for L, r in final.items()},
              'final_token_acc': token_acc, 'final_seq_acc': seq_acc}

    if args.attn != 'none':
        # 对齐准确率与注意力分布的“尖锐”程度（平均最大权重）。
        # 注意：对 hard_train 模型，这里统计的是 one-hot 之前的软注意力权重，即打分函数实际学到的分布。
        if args.hard_train:
            log('[说明] 该模型训练和测试均使用硬注意力；以下对齐准确率、平均最大注意力权重和热力图'
                '基于 one-hot 之前的软注意力权重（即打分函数学到的分布）', log_file)
        model.eval()
        with torch.no_grad():
            align, max_w = {}, {}
            for length in HEATMAP_LENGTHS:
                if length not in test_set:
                    continue
                _, attn = model(test_set[length][0], max_len=length)
                align[str(length)] = alignment_accuracy(attn)
                max_w[str(length)] = attn.max(dim=-1).values.mean().item()
        model.train()
        record['alignment_acc'], record['mean_max_weight'] = align, max_w
        log('[对齐准确率] ' + ' '.join(f'L={L}:{v:.3f}' for L, v in align.items()), log_file)
        log('[平均最大注意力权重] ' + ' '.join(f'L={L}:{v:.3f}' for L, v in max_w.items()), log_file)

    # 先保存结果，再画热力图（即使画图出错，训练结果也不会丢失）
    result_path = os.path.join('results', f'{tag}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
    torch.save(model.state_dict(), ckpt_path)
    log(f'实验结果已保存到 {result_path}，模型参数已保存到 {ckpt_path}，总用时 {time.time() - start:.0f}s', log_file)

    if args.attn != 'none':
        try:
            draw_heatmaps(model, test_set, heatmap_path, tag)
            log(f'注意力热力图已保存到 {heatmap_path}', log_file)
        except Exception as e:  # noqa
            log(f'热力图绘制失败（{type(e).__name__}: {e}），请检查 TODO 8', log_file)


if __name__ == '__main__':
    main()
