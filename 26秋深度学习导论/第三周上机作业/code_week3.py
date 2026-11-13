import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def load_dataset(batch_size):

    """
    -功能: 
    使用torchvision.datasets下载所需数据集
    通过DataLoader类指定batch大小并返回tensor

    -任务:
    请完善此函数 对齐各方法所需的数据类型接口 并正确读取到数据
    """

    mnist_train = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True,
                                                   transform=transforms.ToTensor())
    print(f'Fashion mnist_train: {len(mnist_train)}\nFashion mnist_test: {len(mnist_test)}')
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class Net(nn.Module):

    """
    -功能:
    全连接神经网络框架

    -任务:
    请自行实现全连接神经网络框架的搭建
    """

    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train(model, train_loader, criterion, optimizer, epoch):

    """
    -功能:
    实现模型的训练过程

    -任务:
    请完善train函数
    """

    model.train()
    train_loss = 0
    for data, label in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return train_loss


def val(model, test_loader, criterion, epoch):
    
    """
    -功能:
    实现模型训练完成后的验证过程

    -任务:
    请完善val函数
    """

    model.eval()
    val_loss = 0
    groundtruth_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            groundtruth_labels.append(label.numpy())
            pred_labels.append(predicted.numpy())

    val_loss = val_loss / len(test_loader.dataset)
    groundtruth_labels, pred_labels = np.concatenate(groundtruth_labels), np.concatenate(pred_labels)
    acc = np.sum(groundtruth_labels == pred_labels) / len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:.6f}'.format(epoch, val_loss, acc))
    return val_loss, acc



def main():
    #===== 1. 参数配置 =====
    batch_size = 256
    lr = 1e-3
    epochs = 20

    
    #===== 2. 读入数据 =====
    train_loader, test_loader = load_dataset(batch_size)
    

    #===== 3. 初始化模型 =====
    model = Net()

    #===== 4. 初始化学习准则及优化器 =====
    # 指定训练loss为交叉熵loss  优化器默认为Adam
    # PyTorch会自动把整数型的label转为one-hot型 以便计算交叉熵
    # 思考：使用交叉熵损失函数时，预测模型的输出层是否需添加softmax层? 为什么？  （这个不用提交， 答案是否）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ===== 新增：用于绘制曲线的数据记录列表 =====
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        t_loss = train(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = val(model, test_loader, criterion, epoch)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

    # ===== 5. 绘制曲线 =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()