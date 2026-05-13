#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# 基于 PyTorch 的卷积神经网络（CNN）实现
# 数据集：MNIST 手写数字识别（0-9，共10类）
# 网络结构：2个卷积层 + 2个全连接层 + Dropout正则化
# =============================================================================

# 导入操作系统模块，用于路径管理和环境变量设置
import os

# 导入 NumPy，用于高效的数值计算（矩阵、向量操作等）
import numpy as np

# 导入 PyTorch 主库
import torch

# 导入进度条库
from tqdm import tqdm

# 导入神经网络模块（构建模型的基础类和各类网络层）
import torch.nn as nn

# 导入函数接口模块，包含激活函数、损失函数等常用操作
import torch.nn.functional as F

# 导入数据处理模块，用于封装数据集和批量加载
import torch.utils.data as Data

# 导入 torchvision，包含常用视觉数据集、模型和图像处理工具
import torchvision

# =============================================================================
# 超参数与路径设置
# =============================================================================
LEARNING_RATE  = 1e-3   # 初始学习率
KEEP_PROB_RATE = 0.5    # Dropout 强度
MAX_EPOCH      = 5      # 训练轮数
BATCH_SIZE     = 64     # 批大小

# 路径设置：确保在不同目录下运行都能正确找到数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'mnist')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_cnn_model.pth')

# 自动选择计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 数据集加载与增强
# =============================================================================

# 数据增强：通过对图像进行随机旋转，提高模型对角度变化的鲁棒性
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(15),      # 随机旋转 ±15 度
    torchvision.transforms.ToTensor(),              # 转为张量并归一化
    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 官方均值和标准差进行标准化
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# 检查本地是否已存在 MNIST 数据集
DOWNLOAD_MNIST = False
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    DOWNLOAD_MNIST = True

# 加载训练数据集
train_data = torchvision.datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=train_transform,
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 if os.name != 'nt' else 0 # Windows 下 num_workers > 0 可能会有问题
)

# 加载测试数据集
test_data = torchvision.datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=test_transform
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================================================================
# CNN 模型定义：深度架构优化
# =============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一阶段：基础特征提取 (1->32)
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )

        # 第二阶段：中层特征提取 (32->64)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )

        # 第三阶段：高层语义提取 (64->128)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化 (GAP)：将特征图降维至 1x1，极大减少全连接层参数
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=1 - KEEP_PROB_RATE),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# =============================================================================
# 训练与评估逻辑
# =============================================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

def train(model):
    # 优化器与学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_func = nn.CrossEntropyLoss()

    history = {'loss': [], 'acc': []}
    best_acc = 0.0

    print(f"训练设备: {DEVICE}")
    print("=" * 60)

    for epoch in range(MAX_EPOCH):
        model.train()
        running_loss = 0.0
        
        # 使用 tqdm 包装训练加载器
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{MAX_EPOCH}]")
        
        for step, (images, labels) in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新进度条信息
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 每个 epoch 结束更新学习率
        scheduler.step()
        
        # 记录历史并评估
        epoch_acc = evaluate(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['acc'].append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCH}] 结束 | Avg Loss: {avg_loss:.4f} | Test Acc: {epoch_acc*100:.2f}%")
        
        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[*] 检测到更高准确率，模型已保存至: {MODEL_SAVE_PATH}")

    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(range(1, MAX_EPOCH + 1), history['loss'], color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(range(1, MAX_EPOCH + 1), history['acc'], color=color, marker='s', label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('CNN Training History (MNIST)')
        fig.tight_layout()
        plot_path = os.path.join(BASE_DIR, 'cnn_training_history.png')
        plt.savefig(plot_path)
        print(f"\n[OK] 训练历史曲线已保存至: {plot_path}")
    except Exception as e:
        print(f"\n[Warning] 绘图失败: {e}")

    print("\n" + "=" * 60)
    print(f"训练完成！最高测试准确率: {best_acc*100:.2f}%")

if __name__ == '__main__':
    model = CNN().to(DEVICE)
    train(model)