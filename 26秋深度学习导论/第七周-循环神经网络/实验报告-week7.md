---
header-includes: |
  \usepackage{graphicx}
  \usepackage{float}
  \usepackage{caption}
  \captionsetup{font=small,labelsep=quad}
  \renewcommand{\figurename}{图}
  \renewcommand{\tablename}{表}
---

\begin{center}
{\Large\bfseries 深度学习导论 第七周上机实验报告}\\[10pt]
\end{center}

\vspace{3pt}
\textbf{一、实验环境与任务}

实验在 Python 3.13 和 PyTorch 2.9.1（CPU 版）上完成，另外用到 NumPy、pandas、scikit-learn 和 matplotlib。数据集包含 10 980 个姓氏及其国籍标签，共 18 类，按 8:2 分层划分为训练集 8 784 条、测试集 2 196 条。任务一要求不使用 `nn.RNN` 和 `nn.GRU`，手动把循环网络搭出来并完成国籍分类；任务二把输入序列拉长到 3 000 个字符，观察梯度消失或爆炸，再把激活函数换成 LeakyReLU 做对比，最后用课上讲的方法修好。

\vspace{3pt}
\textbf{二、任务一：手写 RNN 做国籍分类}

模型沿用课件里的循环网络公式，实现上把 $x_t$ 与 $h_{t-1}$ 在特征维拼接，一次线性变换就同时完成 $W_{ih}x_t$ 和 $W_{hh}h_{t-1}$ 两项乘法：

$$z_t=\begin{bmatrix}x_t\\ h_{t-1}\end{bmatrix},\quad
h_t=f(W z_t+b)=f\!\left(W_{ih}x_t+W_{hh}h_{t-1}+b_h\right),\quad
y_t=W_o h_t+b_o$$

拼接后的权重矩阵为 $W=[\,W_{ih}\mid W_{hh}\,]$，所以 `i2h` 用 `Linear(V+H, H)`、`i2o` 用 `Linear(H, 18)`，初始隐状态取全零。输出层没有再套 softmax，因为 `F.cross_entropy` 内部已经带了 log\_softmax，重复一次只会让反传的梯度变小。

```python
self.i2h = nn.Linear(input_size + hidden_size, hidden_size)   # W_ih 与 W_hh 合并
self.i2o = nn.Linear(hidden_size, output_size)                # 输出层
def forward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)                  # [1, V+H]
    hidden = self.activation(self.i2h(combined))              # h_t = f(W z_t + b)
    return self.i2o(hidden), hidden
```

`train()` 中每个 batch 先用 `init_hidden()` 拿到 $h_0$，再按字符顺序逐个时刻送入，最后一个字符走完时的输出就当作整个姓名的表示；16 条样本的梯度先累加，最后统一 `step()` 一次。`nationality_vector()` 返回长度为 1 的类别索引张量，与 `[1,18]` 的输出直接配对即可算交叉熵。

训练配置为：词表 55 个字符（52 个字母加 `.,;`）、隐藏层 128 维、batch 16（有放回采样）、Adam、学习率 1e-3，共更新 5 000 次。

| epoch | 0 | 1000 | 2000 | 3000 | 4000 | 4500 | 4999 |
|---|---|---|---|---|---|---|---|
| loss | 2.883 | 1.878 | 0.726 | 0.784 | 0.742 | 1.230 | 0.525 |
| test acc | 0.087 | 0.644 | 0.684 | 0.694 | 0.709 | 0.726 | 0.718 |

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{code_week7/figures/train_curve.png}
\caption{国籍分类的训练 loss 与测试准确率（每 500 epoch 评估一次）}
\end{figure}

跑满 5 000 次更新，测试准确率到 71.8%，中间 4 500 次时最高，72.6%。前 1 000 次涨得很快，之后每 1 000 次只能再进一两个点，3 000 次往后就基本平了，单层 128 维的 tanh-RNN 容量差不多到顶。loss 在 0.53 到 1.88 之间来回跳，是因为每个 epoch 只随机抽 16 条样本，小批量的方差很大；不过同一个 batch 内的梯度是累加后再更新的，所以趋势还在往下走。

错误集中在几类情况：Lee、Martin、Thomas、Rossi 这种多个语系都在用的姓氏，光看字母排列分不开；English 样本占比很高、类别不均衡，样本少的语系召回率偏低；单层 RNN 对长姓氏开头几个字符的记忆也很弱，走到最后一个字符时前面的信息已经被反复压缩得差不多了。

\vspace{3pt}
\textbf{三、任务二：长序列上的梯度消失与爆炸}

梯度沿时间反传时，每往回走一步都要乘一次 $\partial h_t/\partial h_{t-1}$：

$$\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}\!\big(f'(z_t)\big)W_{hh},\qquad
\frac{\partial L}{\partial W_{hh}}=\sum_{t=1}^{L}\delta_t h_{t-1}^{\top},\quad
\delta_t=\operatorname{diag}\!\big(f'(z_t)\big)W_{hh}^{\top}\delta_{t+1}$$

跨 $L$ 步就是连乘 $L$ 次，量级大致为 $\big(\rho(W_{hh})\cdot|f'|\big)^{L}$，大于 1 往爆炸走，小于 1 往消失走。这次实验的输入直接用了字符的 ASCII 码（65 到 122），权重又被放大 2 倍，$|z_t|$ 一开始就在 100 左右，这一点对结果影响很大。

| seq_len | tanh | LeakyReLU | 截断后 |
|---|---|---|---|
| 300 | 281 | 2.83e8 | 1.00 |
| 600 | 240 | 3.17e8 | 1.00 |
| 1500 | 295 | 2.77e8 | 1.00 |
| 2999 | 264 | 2.94e8 | 1.00 |

*表里都是全局梯度范数：前两列是截断前的值，最后一列是打开 `--clip 1.0` 之后的值。*

\begin{figure}[H]
\centering
\includegraphics[width=0.76\textwidth]{code_week7/figures/grad_compare.png}
\caption{超长序列下实际用于更新的梯度范数（对数坐标）}
\end{figure}

tanh 那组最直接：日志里 `i2h.weight` 和 `i2h.bias` 的梯度全是 0，只有直接接在输出上的 `i2o` 还剩 -152 左右。$|z|\approx100$，tanh 早就贴到 $\pm1$，导数 $1-\tanh^2z$ 在 float32 下就是 0，连乘 300 次还是 0，循环部分一点梯度都拿不到，等于没学。

换成 LeakyReLU 之后，梯度范数从 $2\times10^2$ 跳到 $2.8\times10^8$，高了六个数量级，参数梯度到了 $10^8$，MSE loss 从 4 785 涨到 $5.2\times10^6$。

任务包里还带了一份 `gradient_explosion_example.txt`，是从别人的环境跑出来的示例日志，只有梯度、没有 loss，也不是我这次的运行结果。它的量级和本次差得很远：300 步时 `i2h.weight` 已经到 $6.4\times10^{15}$，600 步到 $3.9\times10^{26}$，1 500 步之后直接变成 inf 和 nan。后面拿它做对照，只是因为同一个脚本能在两种尺度下各跑出一次。

差别出在激活函数有没有饱和。中间也怀疑过是更新步长的问题，把 lr 从 1e-5 降到 1e-9，梯度范数还是 $2.83\times10^8$，说明问题不在步长，而在反传本身。tanh 的导数不超过 1，而这次每一步都落在饱和区，$f'$ 直接就是 0，梯度在第一个时刻就断了，`i2h` 拿到的是精确的 0，不是慢慢衰减到很小那种。LeakyReLU 正半轴导数恒为 1、负半轴 0.01，不会出现 0，$\operatorname{diag}(f')$ 要么接近单位矩阵、要么是 0.01 倍单位矩阵，连乘能一直往后传，量级最后由 $W_{hh}^{L}$ 的谱半径决定。

这次 LeakyReLU 的梯度到了 $2.8\times10^8$，却没有像那份示例日志一样随长度往上翻。看日志里的 `i2h.weight`，第一行三个元素在 $10^7\sim10^8$，第二行只有 $10^3\sim10^5$，同一列上下差了三百倍左右，说明第二个隐单元基本压在负半轴，被 0.01 的斜率掐住，梯度沿这条通路几乎传不回去；剩下那条通路的主导因子又略小于 1，于是停在一个很大的常数上。示例日志那组权重应该两个维度都落在正半轴，乘数大于 1，300 步就差出七个数量级，再往后 float32 装不下，直接成了 inf 和 nan。一个乘数略小于 1、一个大于 1，差别全在这。

还有两个因素在放大问题。字符用的是 ASCII 原值（65 到 122），既没归一化也没换成 embedding，权重再乘 2，$|z|$ 一上来就是 100 左右，对 tanh 来说早就在饱和区里了。另外 $W_{hh}$ 在所有时刻共享，$\partial L/\partial W_{hh}$ 是 $L$ 项相加，只要有一项失控，整体就跟着失控。

修复用梯度截断，加在 `loss.backward()` 之后、`optmizer.step()` 之前，输入序列总长 3 000 没有改动：

```python
# 先算出全局范数留作记录，再按阈值等比缩放
grad_norm = torch.sqrt(sum((p.grad ** 2).sum()
                           for p in model.parameters() if p.grad is not None))
if cfg.clip > 0:                              # cfg.clip = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
```

截断前范数是 $2.8\times10^8$，截断后恒为 1.00，相当于整体缩了 $3.5\times10^{-9}$，参数梯度回到 $10^{-1}$ 量级。序列长度从 300 到 2 999 都稳定在 1.0，不是靠缩短序列绕过去的。

阈值试过 0.5、1.0 和 5.0，四段序列的 loss 完全一样（$5.25\times10^6$ 到 $5.54\times10^6$），因为每段只更新一步、lr 又只有 1e-5，单步的差别在数值上看不出来，报告里统一取 1.0。

截断到底救的是什么，我一开始理解错了。Adam 的更新量是 $m/(\sqrt{v}+\epsilon)$，一阶矩和二阶矩同比例放大，结果与梯度大小无关：拿 $3\times10^2$ 和 $3\times10^8$ 两个梯度分别试 Adam，第一步都只走了 1e-5。所以 $2.8\times10^8$ 这个范数本身并不会让 Adam 崩掉，这也是本次实验里 loss 虽然到了 $5.2\times10^6$、参数却没有立刻变成 NaN 的原因。真正致命的是溢出，示例日志里梯度到了 $1\times10^{26}$，平方之后超出 float32 的上限（约 $3.4\times10^{38}$），直接变 inf，再往下算就是 NaN，参数一旦被 NaN 更新就回不来了；换成 SGD 更直接，更新量正比于梯度，同样 lr=1e-5 下 $3\times10^8$ 的梯度一步就走 3e3。

所以在我这套设置里，截断实际管的是别让梯度大到溢出成 NaN；对 Adam 算一层保险，对 SGD 才是真正在限制步长。想更根本地解决，还是换 LSTM 或 GRU，或者把输入归一化、对 $W_{hh}$ 做正交初始化，梯度消失那侧可以用 ReLU 系列激活配合残差连接。

\vspace{3pt}
\textbf{四、结论}

这份 RNN 在 18 类姓氏国籍分类上做到 71.8%，十几个字符的短序列它是够用的。序列拉到 3 000 就不行了：tanh 因为输入没归一化很快饱和，梯度精确为 0，循环部分完全学不动；换成不饱和的 LeakyReLU，梯度量级又失控到 $2.8\times10^8$，和示例日志里 $10^{26}$ 之后变 NaN 是同一个连乘机制。截断到 1.0 能让训练继续跑，但这次没崩很大程度上是因为 Adam 对梯度尺度不敏感，换 SGD 早就发散了，所以它只是一层保险，长依赖还得靠 LSTM 或 GRU。
