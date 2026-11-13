# 模拟输入序列过长导致梯度爆炸的产生
# 此份代码没有需要填空的地方，在RNN搭建好后可以直接运行
import torch
import string
from torch import nn
from model import RNN
import torch.nn.functional as F
from utils import get_local_time

import numpy as np
import random
import argparse

# ---------------- 任务2 新增的可配置项（其余代码保持原样） ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--activation', type=str, default='leaky_relu',
                    choices=['tanh', 'leaky_relu'],
                    help='RNN 的激活函数，默认 LeakyReLU（任务2要求）')
parser.add_argument('--clip', type=float, default=0.0,
                    help='梯度截断阈值（全局范数），<=0 表示不截断')
parser.add_argument('--log', type=str, default='./week7-gradient_explosion.txt',
                    help='梯度日志文件路径')
cfg = parser.parse_args()
# ------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 手动固定种子，不要修改，可能会复现不出来
seed = 42
set_seed(seed)

# 定义输入文本序列
text = ''.join([string.ascii_letters[(i+3)%52] for i in range(3000)])        # 初始化一个长度len(text) = 3000的文本序列作为输入
inputs = torch.tensor([ord(c) for c in text[:-1]]).view(-1, 1, 1).float()   # 对输入序列进行ASCII编码，每个字符对应一个编码
targets = torch.tensor([ord(c) for c in text[1:]]).view(-1, 1)              

input_size = 1
hidden_size = 2
output_size = 1
learning_rate = 0.00001
epoch = 10
sequence_length = [int(0.1*len(text)),int(0.2*len(text)),int(0.5*len(text)),len(text)-1]

model = RNN(input_size, hidden_size, output_size, activation=cfg.activation)

# 配置日志文件,记录训练过程
cur_time = get_local_time()
log_file_path = cfg.log
with open(log_file_path, 'a') as f:
    f.write(f'---BOOM 0.0--- activation={cfg.activation}, clip={cfg.clip}, time={cur_time}\n')

# torch网络搭建过程中会随机初始化权重参数
# 这里初始化权重为较大的值
# 如果你得RNN中网络层的名字不是这个，则需要修改这里的代码，或者注释掉这里
with torch.no_grad():
    model.i2h.weight.data *= 2
    model.i2o.weight.data *= 2

criterion = nn.MSELoss()
optmizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

def global_grad_norm():
    """所有参数的全局梯度范数（L2）"""
    return torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))

for seq_len in sequence_length:
    message = f'\n---------------SEQ LEN: {seq_len}-------------'
    print(message)
    # 记录日志
    with open(log_file_path, 'a') as f:
        f.write(f'{message}\n')
    hidden = model.init_hidden()
    optmizer.zero_grad()
    output = inputs.new_zeros(1, 1)

    for j in range(seq_len):
        output, hidden = model(inputs[j], hidden)   # 一次送入一个字符

    # 得到序列的最后一个输出后再求loss，并尝试梯度回传，此时的计算图已经包括前面的所有序列，因此此loss值会很大
    loss = criterion(output, targets[j].float().view(-1,1))
    print(f"Loss: {loss.item()}")

    loss.backward()

    # 梯度截断（任务2 的修复），clip<=0 时不生效
    grad_norm = global_grad_norm()
    if cfg.clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    message = f'global grad norm before clip: {grad_norm.item():.6e} | after clip: {global_grad_norm().item():.6e}'
    print(message)
    with open(log_file_path, 'a') as f:
        f.write(f'{message}\n')

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            message = f'{name}, parameters grad: {param.grad}'
            print(message)
            # 记录日志
            with open(log_file_path, 'a') as f:
                f.write(f'{message}\n')
    optmizer.step()
