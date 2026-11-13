# 实现简化CLIP的计算
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import SimpleCLIP,ContrastiveLoss,device
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 设置数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    transforms.Resize((224, 224)),                # 调整大小以匹配 ResNet 输入要求
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet标准化
])

# 下载并加载 FashionMNIST 数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 自定义数据集类
class MyFashionMNIST(Dataset):
    def __init__(self, datasets):
        self.dataset = datasets
        self.label_dict = {
                            0: 'T-shirt/top',
                            1: 'Trouser',
                            2: 'Pullover',
                            3: 'Dress',
                            4: 'Coat',
                            5: 'Sandal',
                            6: 'Shirt',
                            7: 'Sneaker',
                            8: 'Bag',
                            9: 'AnkleBoot'
                        }
        # 注意这里我们总的需要的词汇表就这么多，真实clip不是这样
        self.vocab = {
                            'a': 0,
                            'photo': 1,
                            'of': 2,
                            'T-shirt/top': 3,
                            'Trouser': 4,
                            'Pullover': 5,
                            'Dress': 6,
                            'Coat': 7,
                            'Sandal': 8,
                            'Shirt': 9,
                            'Sneaker': 10,
                            'Bag': 11,
                            'AnkleBoot': 12
                        }
    # TODO: 添加额外属性:  a photo of [label]  
    # step1: 生成caption  step2：分词   step3：根据词汇表vacab进行token转id
    def generate_extra_attribute(self,label):
        # label_caption = 
        # tokens = 
        # token_ids = 
        label_caption = f"a photo of {self.label_dict[label]}"
        tokens = label_caption.split()
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取图像和标签
        image, label = self.dataset[idx]
        # 获取额外属性
        extra = self.generate_extra_attribute(label)
        # 返回图像、标签和额外属性
        return image, label, extra

train_dataset_new = MyFashionMNIST(train_dataset)
test_dataset_new = MyFashionMNIST(test_dataset)

# 创建数据加载器
train_loader_new = DataLoader(dataset=train_dataset_new, batch_size=32, shuffle=True)
test_loader_new = DataLoader(dataset=test_dataset_new, batch_size=32, shuffle=False)

images, labels, extras = next(iter(train_loader_new))
print(extras.shape)

clip = SimpleCLIP(len(train_dataset_new.vocab),emb_size=128)
clip_model = clip.to(device)



def get_test_text(vocab,label_dict):
    # 输出是10个自己生成的文本描述（a photo of [label]）的token_id形式
    text_10 = []
    for label in range(10):
        # TODO：补全
        caption = f"a photo of {label_dict[label]}"
        tokens = caption.split()
        text_10.append([vocab[token] for token in tokens])
    return torch.tensor(text_10, dtype=torch.long)

text_10 = get_test_text(train_dataset_new.vocab,train_dataset_new.label_dict)

def test_clip_model(clip_model, test_loader,text_10):
    clip_model.eval()
    correct_predictions = 0
    total_num = 0
    with torch.no_grad():
        for images, labels, texts in tqdm(test_loader):
            images, text_10 = images.to(device), text_10.to(device)
            image_features, text_features = clip_model(images, text_10)

            # 计算相似度
            similarities = torch.matmul(image_features, text_features.T)
            # 获取最大相似度对应的文本索引
            best_match_idx = similarities.argmax(dim=1)
            
            # 计算预测正确的样本数
            correct_predictions += (best_match_idx == labels.to(device)).sum().item()
            total_num += images.shape[0]
            
            # 输出图像标签和最匹配的文本标签
#             for i in range(len(labels)):
#                 print(f"Image Label: {labels[i]}, Best Matched Text Label: {best_match_idx[i].item()}")
        accuracy = correct_predictions / total_num
        print(f"Accuracy: {accuracy * 100:.2f}%")


# 定义损失函数和优化器
criterion = ContrastiveLoss()
optimizer = optim.Adam(clip.parameters(), lr=0.001)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    clip_model.train()
    total_loss = 0.0
    for images, labels, texts in tqdm(train_loader_new):
        images, texts = images.to(device), texts.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        image_features, text_features = clip_model(images, texts)
        loss = criterion(image_features, text_features)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader_new):.4f}")
    test_clip_model(clip_model,test_loader_new,text_10)
