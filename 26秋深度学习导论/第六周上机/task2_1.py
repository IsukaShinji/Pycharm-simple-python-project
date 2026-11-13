# TODO:传统图像分类方法
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch
from torch.utils.data import DataLoader
from model import ResNet,device
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 设置数据预处理
transform = tf.Compose([
    tf.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    tf.Resize((224, 224)),                # 调整大小以匹配 ResNet 输入要求
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet标准化
])

# 下载并加载 FashionMNIST 数据集 , root 可直接替换为本地数据路径
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)


# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 检查转换后的图像形状
images, labels = next(iter(train_loader))
print("图像的形状：", images.shape)  # 输出应为 [batch_size, 3, 224, 224]


torch.manual_seed(666)  #固定随机数种子
resnet = ResNet()
resnet = resnet.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 测试模型
    resnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')