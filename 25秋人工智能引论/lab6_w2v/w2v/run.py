#!/usr/bin/env python
import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')  # 确保PNG正常生成
import matplotlib.pyplot as plt
import time
from word2vec import *
from sgd import *
import os
import multiprocessing

# -------------------------- 优化1：满额CPU利用（提速核心，1秒10轮关键） --------------------------
# 新增简介：用全部物理核心（非逻辑核心），避免CPU算力闲置；若发热严重，可改回 CPU_CORES//2
CPU_CORES = multiprocessing.cpu_count()  # 例：4核→4，8核→8，最大化并行计算
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_CORES)

# -------------------------- 优化2：预加载数据集（修复顺序，避免属性错误） --------------------------
def preload_dataset():
    random.seed(314)
    dataset = StanfordSentiment()
    # 先初始化tokens（创建_wordcount）→再缓存allSentences（避免重复处理）
    tokens = dataset.tokens()
    nWords = len(tokens)
    dataset._allsentences = dataset.allSentences()
    return dataset, tokens, nWords

dataset, tokens, nWords = preload_dataset()

# -------------------------- 原始参数保留（仅调整计算量相关参数） --------------------------
dimVectors = 10  # 词向量维度不变
C = 5  # 窗口大小不变（文档要求）
random.seed(31415)
np.random.seed(9265)
startTime = time.time()

# -------------------------- 优化3：内存高效初始化（float32+连续存储） --------------------------
wordVectors = np.ascontiguousarray(
    np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors,
         np.zeros((nWords, dimVectors), dtype=np.float32)),
        axis=0, dtype=np.float32
    ),
    dtype=np.float32
)

# -------------------------- 优化4：训练函数精简（减少冗余） --------------------------
# 关键：调用负采样时指定K=5（默认10→改5，减少50%负样本计算，提速显著）
def train_skipgram(vec):
    return word2vec_sgd_wrapper(
        skipgram, tokens, vec, dataset, C,
        lambda c, o, ov, d: negSamplingLossAndGradient(c, o, ov, d, K=5)  # K=5减少负采样计算
    )

# -------------------------- 核心修改：恢复每10轮打印+提速配置 --------------------------
# 迭代40000轮不变，PRINT_EVERY=10（恢复每10轮打印），step=0.3不变（保证收敛）
wordVectors = sgd(
    f=train_skipgram,
    x0=wordVectors,
    step=0.3,
    iterations=40000,
    postprocessing=None,
    useSaved=False,
    PRINT_EVERY=10  # 恢复每10轮打印日志（用户要求）
)

# -------------------------- 收敛验证与耗时统计 --------------------------
print("sanity check: cost at convergence should be around or below 10")  # 文档1-42标准
print("total training took %d seconds (avg: %.2f rounds/sec)" % (
    time.time() - startTime, 40000 / (time.time() - startTime)
))  # 新增：显示平均每秒迭代数，验证“1秒10轮”

# -------------------------- 可视化与PNG生成（完全保留原始逻辑） --------------------------
wordVectors = np.ascontiguousarray(
    np.concatenate((wordVectors[:nWords,:], wordVectors[nWords:,:]), axis=0),
    dtype=np.float32
)

visualizeWords = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
    "hail", "coffee", "tea"
]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]

# 高效降维计算
temp = visualizeVecs - np.mean(visualizeVecs, axis=0)
covariance = np.cov(temp, rowvar=False)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

# 保存PNG
for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))
plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
plt.savefig('word_vectors.png')