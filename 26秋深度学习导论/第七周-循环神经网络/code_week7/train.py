import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import numpy as np
import os
import argparse
from tqdm import tqdm

from utils import load_data, dataloader, name_vector, get_local_time
from model import RNN
import matplotlib.pyplot as plt

#function to train the data

def train(args, model, optmizer, batch_size, X_train, y_train, all_letters, languages):
    
    total_loss = 0
    optmizer.zero_grad()
    
    # load the data 
    data = dataloader(all_letters, languages, args.batch_size, X_train, y_train)
    for name, language, name_tensor, nationality_tensor in data:
        # todo
        # 1. rnn的输入需要用到上一个时刻的输出，所有要优先初始化h0
        # 2. 本次的rnn是针对单个字符的字符串序列数据，因此要依次把name_tensor送入RNN，让其预测最终的output
        # 3. 计算output和真实标签nationality_tensor之间的loss，梯度回传
        hidden = model.init_hidden()                     # h_0
        for i in range(name_tensor.size()[0]):           # 逐字符送入，最后一步的输出代表整名
            output, hidden = model(name_tensor[i], hidden)
        loss = F.cross_entropy(output, nationality_tensor)
        loss.backward()                                  # 一个 batch 内累加梯度
        total_loss += loss.detach()
    optmizer.step()
    
    return total_loss/batch_size

def eval(args, model, X_test, y_test, all_letters,languages):

    model.eval()
    correct = 0
    resluts = []

    for surname in X_test:
        out = inference(model, surname, all_letters)
        index = torch.argmax(out)
        resluts.append(languages[index])
    correct = sum(x == y for x, y in zip(y_test, resluts))
    acc = correct/len(X_test)
    print(f'acc: {acc}')
    return acc

def inference(model, name, all_letters):
    model.eval()
    name_tensor = name_vector(name, all_letters)
    hidden = model.init_hidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = model(name_tensor[i], hidden)
    return output

def main():

    # 配置参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filepath', type=str, default='surnames/surnames.csv')
    argparser.add_argument('--test_size', type=float, default=0.2)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--max_epoch', type=int, default=5000)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--log_file_path', type=str, default='./')
    
    args = argparser.parse_args()

    # 配置日志文件,记录训练过程
    cur_time = get_local_time()
    args_str = f'week7-{cur_time}'
    args.log_file = os.path.join(args.log_file_path, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # ---1. 加载数据，并划分---
    X, y, data, languages = load_data(args.filepath)
    n_languages = len(languages)        # 所有姓氏的类别总数，共18类

    torch.manual_seed(42)
    np.random.seed(42)      # 固定采样种子
    # Split the dataset in train and test, no validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size, random_state = 123, stratify = y)
    
    print("Training Data: ", len(X_train))
    print("Testing Data: ", len(X_test))

    # ---2. 构建此表，词表由所有英文字符组成，大小为55---
    all_letters = string.ascii_letters + ".,;"
    vocab_size = len(all_letters)
    
    # ---3. 初始化模型、优化器---
    model = RNN(vocab_size, args.hidden_size, n_languages)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    # ---4. 开始训练---
    for i in tqdm(range(args.max_epoch)):
        loss = train(args, model, optimizer, args.batch_size, X_train, y_train, all_letters, languages)
        
        # eval
        if i%500 == 0 or i == args.max_epoch-1:
            message = f'loss for {i}/{args.max_epoch} : {loss:.4f}'
            print(message)
            acc = eval(args, model, X_test, y_test, all_letters,languages)
            # 记录测试结果
            with open(args.log_file, 'a') as f:
                f.write(f'{message} | ACC: {acc}\n')

        loss_list.append(loss.log10().item())
    

    plt.plot(torch.tensor(loss_list).view(-1,50).mean(1))
    plt.show()


if __name__ == '__main__':
    main()