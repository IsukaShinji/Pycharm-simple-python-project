---
documentclass: article
fontsize: 10pt
geometry: margin=2.0cm
mainfont: Times New Roman
mainfontoptions:
  - Path=/mnt/c/Windows/Fonts/
  - Extension=.ttf
  - UprightFont=times
  - BoldFont=timesbd
  - ItalicFont=timesi
  - BoldItalicFont=timesbi
CJKmainfont: Microsoft YaHei
CJKoptions:
  - Path=/mnt/c/Windows/Fonts/
  - Extension=.ttc
  - UprightFont=msyh
  - BoldFont=msyhbd
monofont: DejaVu Sans Mono
header-includes: |
  \usepackage{graphicx}
  \usepackage{float}
  \usepackage{caption}
  \usepackage{booktabs}
  \usepackage{setspace}
  \setstretch{1.20}
  \setlength{\parindent}{2em}
  \setlength{\parskip}{0.82em}
  \setlength{\intextsep}{1.0em}
  \setlength{\textfloatsep}{1.2em}
  \captionsetup{font=small,labelsep=quad,skip=5pt}
  \renewcommand{\figurename}{图}
  \renewcommand{\tablename}{表}
---

\begin{center}
{\Large\bfseries 深度学习导论 第八周上机实验报告}\\[6pt]
黄浩博\quad 2024201630\quad seed = 1630
\end{center}

\vspace{-2pt}
\noindent\textbf{一、实验设置与任务 1}

实验在 Windows 的 Python 3.11.9 与 PyTorch 2.9.0 上完成，使用 NumPy 和 Matplotlib。训练统一取 `emb_dim=32`、`hidden_dim=128`、batch size 64、4000 步、Adam 学习率 $10^{-3}$，梯度裁剪阈值 1.0，每 250 步在完整测试集上评估一次。五组实验都用 seed 1630，下面的数值均来自对应的 `logs/`、`results/` 和 `figures/`。

任务 1 中，点积打分用 `torch.bmm` 一次算完一个 batch；缩放点积在点积结果上除以 $\sqrt{D}$。加性打分把键和查询分别送入无偏置线性层 `W`、`U`，相加后过 `tanh`，再由无偏置层 `v` 投影成一个分数。`masked_softmax` 在 softmax 前把无效位置置为 $-\infty$，因此这些位置的权重严格为 0。`attention` 将权重与 value 做加权求和；`hard=True` 时先把权重换成 argmax 处的 one-hot 再求和。未使用现成注意力接口，也没有遍历 batch 或序列位置。

运行 `python attention.py` 后得到：

```text
weights = [0.4223, 0.4223, 0.1554]
[PASS] Test 1 课件例子（第 29~30 页）
[PASS] Test 2 点积 / 缩放点积打分
[PASS] Test 3 加性打分函数
[PASS] Test 4 带掩码的 softmax
[PASS] Test 5 软注意力 / 硬注意力
通过 5/5 项自检
```

\begin{table}[H]
\centering
\caption{缩放点积实验：$L=10$，每个维度采样 2000 次}
\small
\begin{tabular}{rrrrrrr}
\toprule
$D$ & var(dot) & var(scaled) & max\_w(dot) & max\_w(scaled) & jac(dot) & jac(scaled) \\
\midrule
4    & 4.12   & 1.031 & 0.495 & 0.306 & 0.3372 & 0.3412 \\
16   & 16.07  & 1.004 & 0.725 & 0.316 & 0.2643 & 0.3459 \\
64   & 64.29  & 1.004 & 0.862 & 0.318 & 0.1608 & 0.3470 \\
256  & 260.69 & 1.018 & 0.933 & 0.324 & 0.0864 & 0.3475 \\
1024 & 1033.72& 1.009 & 0.964 & 0.324 & 0.0477 & 0.3471 \\
\bottomrule
\end{tabular}
\end{table}

查询和键的各分量独立取 $N(0,1)$ 时，$k_j^Tq$ 是 $D$ 个独立乘积之和，每个乘积的方差为 1，所以点积分数的方差约为 $D$；除以 $\sqrt{D}$ 后方差回到 1 附近。表中点积的 max\_w 从 0.495 升到 0.964，$D$ 继续增大时趋近 1，softmax 越来越接近 one-hot；相应的 Jacobian 范数从 0.3372 降到 0.0477，梯度穿过 softmax 时被压得很小，训练容易卡住。缩放点积的 max\_w 一直保持在 0.31 到 0.33，Jacobian 约 0.35，没有出现这种饱和。

\noindent\textbf{二、任务 2：固定上下文与注意力}

无注意力模型把编码器最后时刻的 $h_L$ 当成所有解码步共用的 $c$；注意力模型用上一步隐状态 $s_{t-1}$ 查询整串编码器输出，得到随解码步变化的 $c_t$。两者参数量都是 182235。

\begin{table}[H]
\centering
\caption{两种模型在各长度上的 token 准确率}
\small
\begin{tabular}{lrrrrrr}
\toprule
模型 & $L=5$ & $L=10$ & $L=20$ & $L=30$ & 平均 token & 平均整句 \\
\midrule
none       & 0.996 & 0.823 & 0.338 & 0.207 & 0.5112 & 0.2217 \\
scaled\_dot & 1.000 & 1.000 & 1.000 & 1.000 & 1.0000 & 0.9998 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.76\textwidth]{code_week8/figures/acc_vs_len.png}
\caption{token 准确率随序列长度的变化}
\end{figure}

无注意力模型在 $L=5$ 时有 0.996，到 $L=30$ 只剩 0.207，整句准确率更低，只有 0.2217。固定上下文向量只有一个 128 维向量，序列越长，要同时保存的字符信息越多；编码器最后的状态还要经过 GRU 多步压缩，早期字符的信息很容易被覆盖，解码器又没有别的途径回看输入。注意力模型的上下文每一步重新计算，可以直接取用与当前输出位置对应的编码状态，所以从 5 到 30 都保持 1.000，长度带来的影响基本消失。

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{code_week8/figures/heatmap_scaled_dot_seed1630.png}
\caption{scaled\_dot 模型在 $L=10,20,30$ 上的注意力热力图}
\end{figure}

热力图大体沿对角线分布。$L=10$ 的第一行最亮点落在输入末位，其余行回到对角线；$L=20$ 和 $L=30$ 也有同样的首步偏移，之后基本逐位对齐，所以对齐准确率分别是 0.900、0.950、0.967，正好接近 $(L-1)/L$。第一步的查询是 $s_0=h_L$，它会先看向编码器末位；隐状态更新后，解码第 $t$ 个字符主要看输入第 $t$ 个位置。复制任务要求输入和输出同位置字符相同，这种近似恒等对齐正好把需要的字符直接送入解码器。

\noindent\textbf{三、任务 3：打分函数与硬注意力}

\begin{table}[H]
\centering
\caption{plot\_results.py --seed 1630 的汇总结果}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
模型 & 参数 & token & seq & L=5 & L=10 & L=20 & L=30 & align@30 & max\_w@30 & step(seq$\ge$0.9) \\
\midrule
none                 & 182235 & 0.5112 & 0.2217 & 0.996 & 0.823 & 0.338 & 0.207 & --    & --    & 未达到 \\
dot                  & 182235 & 0.9989 & 0.9960 & 1.000 & 1.000 & 0.998 & 0.989 & 0.956 & 0.998 & 2500 \\
scaled\_dot          & 182235 & 1.0000 & 0.9998 & 1.000 & 1.000 & 1.000 & 1.000 & 0.967 & 0.453 & 2750 \\
additive             & 215131 & 0.9999 & 0.9992 & 1.000 & 1.000 & 0.998 & 0.999 & 0.999 & 0.703 & 1000 \\
scaled\_dot + hard   & 182235 & 0.4146 & 0.1471 & 0.981 & 0.653 & 0.269 & 0.172 & 0.011 & 0.043 & 未达到 \\
\bottomrule
\end{tabular}
\end{table}

dot 和 scaled\_dot 本身没有可学习参数，参数量都是 182235。加性打分多出的 32896 个参数全部来自打分网络：$W$ 是 $128\times128=16384$，$U$ 同样是 16384，$v$ 是 $128\times1=128$，合计 $16384+16384+128=32896$，所以 $182235+32896=215131$。

三种软注意力里，dot 的 max\_w@30 最高，为 0.998；加性是 0.703，scaled\_dot 最平缓，为 0.453。dot 的热力图几乎是清晰的对角 one-hot，scaled\_dot 的对角线更宽，周围还留有明显权重，加性介于两者之间。这个顺序和任务 1 的维度实验一致：这里键和查询都是 128 维，未缩放点积的分数尺度随维度增大，softmax 容易饱和；除以 $\sqrt{D}$ 后分布被拉平。三种模型最终都能完成复制，说明这个任务每一步只需要抓住一个主要位置，分布是否尖锐不影响最终准确率。

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{code_week8/figures/heatmap_dot_seed1630.png}
\caption{dot 模型热力图，与 scaled\_dot 相比更接近 one-hot}
\end{figure}

硬注意力训练的结果明显差于 none 和 scaled\_dot：平均 token 准确率 0.4146，整句准确率 0.1471，$L=30$ 时 token 准确率只有 0.172，对齐准确率只有 0.011。问题在于 one-hot 的索引来自 argmax，argmax 和 one-hot 对注意力分数的导数几乎处处为 0，梯度无法告诉打分函数应该往哪里移。模型只能从被选中的 value 得到梯度，选择动作本身不可导，所以位置学不准；这一步也没有使用直通估计或 REINFORCE。软注意力把权重作为连续变量参与反向传播，梯度可以同时传到所有位置，因而能稳定学出对角线。

\noindent\textbf{四、思考题}

本实验的查询向量是解码器上一步的隐状态 $s_{t-1}$，键和值都是编码器各时刻的隐状态。若换成文本分类、没有解码器，可以用整句隐状态的均值池化结果作为查询，也可以设一个可学习的全局查询向量；键和值仍取各 token 的隐状态，最后把注意力上下文送入分类头。这样查询就不再依赖解码步，而是来自整句表示或任务级参数。
