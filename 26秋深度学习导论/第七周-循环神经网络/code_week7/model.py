import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import string
import numpy as np

# Build the RNN Model

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        # todo
        # 完成模型的搭建，禁止使用nn.RNN,nn.GRU<循环>神经网络类，可以使用nn.Linear等其他类，手动完成基础网络的搭建
        # 提示: 模型的输入有input_t,和h_{t-1}, 可以分别过网络，也可以合并
        super(RNN,self).__init__()
        self.hidden_size = hidden_size          # init_hidden() 需要用到

        # x_t 与 h_{t-1} 拼接后一次线性变换
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)          # 输出层

        # 激活可换：任务1 用 tanh，任务2 换 LeakyReLU
        self.activation = {'tanh': torch.tanh,
                           'leaky_relu': F.leaky_relu,
                           'relu': F.relu}[activation]

    def forward(self, input, hidden):
        """
        input.shape = [1,input_size]
        hidden.shape = [1,args.hidden_size]
        output.shape = [1,output_size]
        """
        # todo
        # 参考RNN的计算公式完成forward函数
        # forward的输出在这里直接固定为output和hidden，统一接口
        # 激活函数自行选择
        combined = torch.cat((input, hidden), 1)          # h_t = f(W_ih x_t + W_hh h_{t-1} + b)
        hidden = self.activation(self.i2h(combined))      # [1, hidden_size]
        output = self.i2o(hidden)                         # [1, output_size]
        return output, hidden
        
    def init_hidden(self): # Define the initail hidden state
        return torch.zeros(1, self.hidden_size)

