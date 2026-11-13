"""
第八周上机任务：工具函数（本文件无需修改）

本次实验使用“序列复制”任务：输入一个随机的小写字母序列，要求模型原样输出该序列。
例如：输入 "qwhzk" -> 输出 "qwhzk"。
任务本身很简单，但编码器必须把整个输入序列的信息传递给解码器，
因此非常适合用来观察“固定上下文向量 c”带来的信息瓶颈，以及注意力机制的作用。
"""
import datetime
import random

import numpy as np
import torch

# ---------------- 词表 ----------------
# 0 号为解码器的起始符 <SOS>，1~26 号依次为小写字母 a~z
SOS_IDX = 0
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
VOCAB = ['<SOS>'] + list(LETTERS)
VOCAB_SIZE = len(VOCAB)  # 27


def set_seed(seed):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_local_time():
    """返回当前时间字符串，用于日志文件命名。"""
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')


def encode(text):
    """字符串 -> 下标列表，例如 'abc' -> [1, 2, 3]"""
    return [LETTERS.index(ch) + 1 for ch in text]


def decode(ids):
    """下标列表 -> 字符串（忽略 <SOS>），例如 [1, 2, 3] -> 'abc'"""
    return ''.join(VOCAB[int(i)] for i in ids if int(i) != SOS_IDX)


def make_copy_batch(batch_size, length, generator=None):
    """
    生成一个 batch 的“序列复制”数据（同一个 batch 内所有序列长度相同，因此不需要填充 PAD）。

    返回:
        src: [batch_size, length] 输入序列，取值为 1~26（即 a~z）
        tgt: [batch_size, length] 目标序列，与 src 完全相同
    """
    src = torch.randint(1, VOCAB_SIZE, (batch_size, length), generator=generator)
    tgt = src.clone()
    return src, tgt


def build_test_set(min_len, max_len, num_per_len=200, seed=2026):
    """
    构造测试集：对每个长度 L ∈ [min_len, max_len] 各生成 num_per_len 条序列。
    测试集使用固定种子 2026（与训练种子无关），因此所有同学的测试集完全相同。

    返回: dict，{L: (src, tgt)}
    """
    g = torch.Generator().manual_seed(seed)
    return {L: make_copy_batch(num_per_len, L, g) for L in range(min_len, max_len + 1)}
