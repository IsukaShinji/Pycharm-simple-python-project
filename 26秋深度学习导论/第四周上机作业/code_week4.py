import copy
import os
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 图片保存到当前文件目录
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int):
    """固定随机种子，保证不同实验可比较。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(batch_size: int, seed: int):
    """
    - 功能：
        读取 FashionMNIST 数据集，并在相同 seed 下保证数据打乱顺序一致。
    - 任务：
        完善数据类型转换和 DataLoader。
    """
    # TODO 1: 将 PIL 图像转换为 Tensor
    # PIL转Tensor，像素归一化到[0,1]
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.FashionMNIST(
        root='./datasets', train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        root='./datasets', train=False, transform=transform, download=True
    )

    # 已提供：同一 seed 下不同实验使用一致的数据 shuffle 顺序
    generator = torch.Generator()
    generator.manual_seed(seed)

    # TODO 2: 创建 train_loader 和 test_loader
    # 训练集打乱，用固定generator保证各实验顺序一致；测试集不打乱
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MLP(nn.Module):
    """
    - 功能：
        实现 784 -> 256 -> 128 -> 64 -> 10 的多层感知机。
    - 任务：
        1) 完成四个全连接层；
        2) 支持 ReLU / Sigmoid 两种激活函数；
        3) 完成 forward；
        4) 输出层不要添加 softmax。
    """

    def __init__(self, activation='relu'):
        super().__init__()

        # TODO 3: 定义四个全连接层
        # 784 -> 256 -> 128 -> 64 -> 10
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # TODO 4: 根据 activation 选择激活函数
        self.activation_name = activation.lower()
        if self.activation_name == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation_name == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f'Unsupported activation: {activation}')

    def forward(self, x):
        # TODO 5:
        # 展平输入 → 三层线性+激活 → 最后一层输出logits
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


class ManualSGD:
    """禁止调用 torch.optim.SGD。"""

    def __init__(self, params, lr):
        # TODO 6: 保存参数列表和学习率
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        # TODO 7: 清空梯度
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        # TODO 8: 根据 theta <- theta - lr * grad 更新参数
        # 提示：参数更新不应被自动微分继续记录。
        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue
                param.add_(param.grad, alpha=-self.lr)


class ManualMomentum:
    """带动量的 SGD。禁止调用 torch.optim。"""

    def __init__(self, params, lr, beta=0.9):
        # TODO 9:
        # 保存参数，初始化速度项
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        # TODO 10
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        # TODO 11:
        # v_t = beta * v_{t-1} + g_t
        # theta_t = theta_{t-1} - lr * v_t
        with torch.no_grad():
            for param, velocity in zip(self.params, self.velocities):
                if param.grad is None:
                    continue
                velocity.mul_(self.beta).add_(param.grad)
                param.add_(velocity, alpha=-self.lr)


class ManualAdam:
    """手动实现 Adam。禁止调用 torch.optim。"""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # TODO 12:
        # 保存参数，初始化一二阶矩和步数
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        # TODO 13
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        # TODO 14:
        # 更新步数 → 更新矩 → 偏差修正 → 更新参数
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t

        with torch.no_grad():
            for param, m, v in zip(self.params, self.m, self.v):
                if param.grad is None:
                    continue
                grad = param.grad
                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                param.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


def build_optimizer(name, params, lr):
    """根据字符串返回自己实现的优化器。"""
    # TODO 15
    name = name.lower()
    if name == 'sgd':
        return ManualSGD(params, lr)
    if name == 'momentum':
        return ManualMomentum(params, lr)
    if name == 'adam':
        return ManualAdam(params, lr)
    raise ValueError(f'Unknown optimizer: {name}')


def train(model, train_loader, criterion, optimizer, device, record_grad=False):
    """
    - 功能：完成一个 epoch 的训练。
    - 当 record_grad=True 时，记录 fc1/fc2/fc3 权重梯度的平均 L2 norm。
    """
    model.train()
    total_loss = 0.0
    total_num = 0
    grad_sums = {'fc1': 0.0, 'fc2': 0.0, 'fc3': 0.0}
    grad_steps = 0

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        # TODO 16:
        # 清梯度 → 前向 → 算loss → 反向 → 记梯度 → 更新参数
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        if record_grad:
            # step前记录各层梯度范数
            with torch.no_grad():
                grad_sums['fc1'] += model.fc1.weight.grad.norm().item()
                grad_sums['fc2'] += model.fc2.weight.grad.norm().item()
                grad_sums['fc3'] += model.fc3.weight.grad.norm().item()
            grad_steps += 1

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_num += data.size(0)

    # TODO 17: 计算平均loss和梯度
    avg_loss = total_loss / total_num
    avg_grad = None
    if record_grad and grad_steps > 0:
        avg_grad = {name: value / grad_steps for name, value in grad_sums.items()}
    return avg_loss, avg_grad


@torch.no_grad()
def val(model, test_loader, criterion, device):
    """计算测试集平均 loss 与 accuracy。"""
    model.eval()
    total_loss = 0.0
    total_num = 0
    correct = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        # TODO 18: 前向计算loss和准确率
        output = model(data)
        loss = criterion(output, label)
        total_loss += loss.item() * data.size(0)
        total_num += data.size(0)
        predicted = output.argmax(dim=1)
        correct += (predicted == label).sum().item()

    # TODO 19
    val_loss = total_loss / total_num
    val_acc = correct / total_num
    return val_loss, val_acc


def run_experiment(
    optimizer_name,
    activation,
    init_state,
    seed,
    batch_size,
    lr,
    epochs,
    device,
    record_grad=False,
):
    """
    已提供实验骨架。
    注意：每个实验重新创建 DataLoader，并载入同一份 init_state，保证公平比较。
    """
    train_loader, test_loader = load_dataset(batch_size, seed)

    model = MLP(activation=activation).to(device)
    model.load_state_dict(copy.deepcopy(init_state))
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'fc1_grad': [],
        'fc2_grad': [],
        'fc3_grad': [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, grad_info = train(
            model, train_loader, criterion, optimizer, device, record_grad=record_grad
        )
        val_loss, val_acc = val(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if record_grad:
            history['fc1_grad'].append(grad_info['fc1'])
            history['fc2_grad'].append(grad_info['fc2'])
            history['fc3_grad'].append(grad_info['fc3'])

        print(
            f'[{optimizer_name:8s} | {activation:7s}] '
            f'Epoch {epoch:02d}: train_loss={train_loss:.4f}, '
            f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}'
        )

    return history


def plot_optimizer_comparison(histories, save_path='optimizer_compare.png'):
    """绘制三种优化器的训练 loss 和测试 accuracy。"""
    # TODO 20: 绘制并保存两张图
    if not os.path.isabs(save_path):
        save_path = os.path.join(SAVE_DIR, save_path)

    plt.rcParams['axes.unicode_minus'] = False
    num_epochs = len(next(iter(histories.values()))['train_loss'])
    epochs = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, history in histories.items():
        axes[0].plot(epochs, history['train_loss'], marker='o', markersize=4,
                     label=name.upper())
        axes[1].plot(epochs, history['val_acc'], marker='o', markersize=4,
                     label=name.upper())

    axes[0].set_title('Training Loss vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Validation Accuracy vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[plot] optimizer comparison saved to {save_path}')


def plot_gradient_comparison(histories, save_path='gradient_compare.png'):
    """比较 ReLU / Sigmoid 下浅层与深层的梯度 norm。"""
    # TODO 21: 至少画出 fc1 和 fc3 的梯度曲线；建议 y 轴使用对数刻度
    if not os.path.isabs(save_path):
        save_path = os.path.join(SAVE_DIR, save_path)

    plt.rcParams['axes.unicode_minus'] = False
    layers = ['fc1_grad', 'fc2_grad', 'fc3_grad']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (activation, history) in zip(axes, histories.items()):
        num_epochs = len(history['train_loss'])
        epochs = range(1, num_epochs + 1)
        for key in layers:
            ax.plot(epochs, history[key], marker='o', markersize=4,
                    label=key.split('_')[0].upper())

        ax.set_yscale('log')
        ax.set_title(f'{activation.upper()} MLP')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average gradient L2 norm')
        ax.grid(alpha=0.3, which='both')
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[plot] gradient comparison saved to {save_path}')


def count_parameters(model):
    """返回可训练参数总数。"""
    # TODO 22
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # ===== 1. 参数配置 =====
    # 请改成自己的“学号后四位”
    seed = 1630
    batch_size = 128
    epochs = 10
    lr = 1e-2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    # ===== 2. 创建统一的初始参数 =====
    base_model = MLP(activation='relu')
    init_state = copy.deepcopy(base_model.state_dict())
    print('Number of trainable parameters:', count_parameters(base_model))

    # ===== 3. 优化器对比：ReLU + 相同初始化 + 相同数据顺序 =====
    optimizer_histories = {}
    for optimizer_name in ['sgd', 'momentum', 'adam']:
        optimizer_histories[optimizer_name] = run_experiment(
            optimizer_name=optimizer_name,
            activation='relu',
            init_state=init_state,
            seed=seed,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            device=device,
            record_grad=False,
        )
    plot_optimizer_comparison(optimizer_histories)

    # ===== 4. 激活函数与梯度传播对比：统一使用 SGD =====
    gradient_histories = {}
    for activation in ['relu', 'sigmoid']:
        gradient_histories[activation] = run_experiment(
            optimizer_name='sgd',
            activation=activation,
            init_state=init_state,
            seed=seed,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            device=device,
            record_grad=True,
        )
    plot_gradient_comparison(gradient_histories)


if __name__ == '__main__':
    main()
