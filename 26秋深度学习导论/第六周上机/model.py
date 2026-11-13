import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet101,resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 若本机为 Intel XPU 版 torch，则优先使用 XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')

# 定义图像编码器和文本编码器
class SimpleImageEncoder(nn.Module):
    def __init__(self,emb_size=128):
        super(SimpleImageEncoder, self).__init__()
        # 使用预训练的ResNet模型作为图像编码器
        self.model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, emb_size)  # 图像特征向量

    def forward(self, x):
        return self.model(x)
    
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, emb_size=128):
        super(SimpleTextEncoder, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 输出全连接层
        self.fc = nn.Linear(hidden_dim, emb_size) 
    
    def forward(self, input_ids):
        # 输入嵌入
        embedded = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)
        # LSTM处理
        _, (hidden, _) = self.lstm(embedded)  # hidden shape: (1, batch_size, hidden_dim)
        sentence_features = hidden[-1]  # 使用最后一层的隐状态作为句子特征
        return self.fc(sentence_features)  # 文本特征向量
    
# 定义CLIP模型
class SimpleCLIP(nn.Module):
    def __init__(self,vocab_size, emb_size=128):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = SimpleImageEncoder(emb_size=emb_size)
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, emb_size=emb_size)
    
    def forward(self, images, input_ids):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids)
        # 对特征进行L2归一化
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        return image_features, text_features
    
# 定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, image_features, text_features):
        # 计算图像和文本特征之间的余弦相似度
        logits = self.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0)) / self.temperature
        labels = torch.arange(len(logits)).to(logits.device)
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
    
# TODO:传统图像分类
class ResNet(nn.Module):
    def __init__(self, num_classes = 10, emb_size=128):  
        super().__init__()
        self.backbone = SimpleImageEncoder(emb_size=emb_size)
        # 自定义全连接层以适应 10 个类别
        # self.classifier = 
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        h = self.backbone(x) 
        h_new = self.classifier(h)
        return h_new
