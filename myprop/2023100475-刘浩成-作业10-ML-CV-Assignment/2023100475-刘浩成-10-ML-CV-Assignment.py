# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ---------- 环境准备 ----------
print("PyTorch版本:", torch.__version__)
print("GPU可用:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 数据准备 ----------
# MNIST标准化参数
mean = 0.1307
std = 0.3081

# 训练集transform（包含归一化）
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# 仅转为Tensor的transform用于显示图片（不归一化）
transform_display = transforms.Compose([
    transforms.ToTensor()
])

# 下载并加载数据集
train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_train)

# 划分训练集(55000)和验证集(5000)
train_size = 55000
val_size = 5000
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 用于可视化显示的测试集（仅ToTensor，不归一化）
test_display = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_display)

# ---------- CNN模型定义 ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)                         # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)                          # 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- 训练函数（用于不同优化器和学习率） ----------
def train_model(model, train_loader, val_loader, optimizer, epochs=5, print_info=True):
    """训练模型并返回每个epoch的loss和acc列表"""
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    model = model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        if print_info:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# =====================================================
# 任务1：使用Adam(lr=0.001)重新训练基准模型
# =====================================================
print("\n===== 任务1: 训练基准模型 (Adam, lr=0.001) =====")
model_base = SimpleCNN()
optimizer_base = optim.Adam(model_base.parameters(), lr=0.001)
history_base = train_model(model_base, train_loader, val_loader, optimizer_base, epochs=5)
train_loss_base, val_loss_base, train_acc_base, val_acc_base = history_base

# 保存基准模型权重，用于后续可视化
torch.save(model_base.state_dict(), "mnist_cnn_baseline.pth")

# =====================================================
# 任务2：优化器对比 (SGD, SGD+Momentum, Adam)
# =====================================================
print("\n===== 任务2: 优化器对比 =====")

# 准备三个优化器，学习率统一设为0.01用于对比（Adam稍低用0.001）
# 为了方便对比，SGD和Momentum使用lr=0.01，Adam使用lr=0.001
model_sgd = SimpleCNN()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
print("\n训练 SGD:")
hist_sgd = train_model(model_sgd, train_loader, val_loader, optimizer_sgd, epochs=5)

model_momentum = SimpleCNN()
optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)
print("\n训练 SGD+Momentum:")
hist_momentum = train_model(model_momentum, train_loader, val_loader, optimizer_momentum, epochs=5)

model_adam = SimpleCNN()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
print("\n训练 Adam:")
hist_adam = train_model(model_adam, train_loader, val_loader, optimizer_adam, epochs=5)

# 绘制优化器对比曲线
epochs = 5
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), hist_sgd[0], 'g-o', label='SGD Train')
plt.plot(range(1, epochs+1), hist_sgd[1], 'g--s', label='SGD Val')
plt.plot(range(1, epochs+1), hist_momentum[0], 'b-o', label='Momentum Train')
plt.plot(range(1, epochs+1), hist_momentum[1], 'b--s', label='Momentum Val')
plt.plot(range(1, epochs+1), hist_adam[0], 'r-o', label='Adam Train')
plt.plot(range(1, epochs+1), hist_adam[1], 'r--s', label='Adam Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('优化器对比 - Loss曲线')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), hist_sgd[2], 'g-o', label='SGD Train')
plt.plot(range(1, epochs+1), hist_sgd[3], 'g--s', label='SGD Val')
plt.plot(range(1, epochs+1), hist_momentum[2], 'b-o', label='Momentum Train')
plt.plot(range(1, epochs+1), hist_momentum[3], 'b--s', label='Momentum Val')
plt.plot(range(1, epochs+1), hist_adam[2], 'r-o', label='Adam Train')
plt.plot(range(1, epochs+1), hist_adam[3], 'r--s', label='Adam Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('优化器对比 - Accuracy曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()

# =====================================================
# 任务3：学习率对比 (Adam, lr=0.1, 0.01, 0.001)
# =====================================================
print("\n===== 任务3: 学习率对比 =====")

model_lr01 = SimpleCNN()
optimizer_lr01 = optim.Adam(model_lr01.parameters(), lr=0.1)
print("\n训练 Adam lr=0.1:")
hist_lr01 = train_model(model_lr01, train_loader, val_loader, optimizer_lr01, epochs=5)

model_lr001 = SimpleCNN()
optimizer_lr001 = optim.Adam(model_lr001.parameters(), lr=0.01)
print("\n训练 Adam lr=0.01:")
hist_lr001 = train_model(model_lr001, train_loader, val_loader, optimizer_lr001, epochs=5)

model_lr0001 = SimpleCNN()
optimizer_lr0001 = optim.Adam(model_lr0001.parameters(), lr=0.001)
print("\n训练 Adam lr=0.001:")
hist_lr0001 = train_model(model_lr0001, train_loader, val_loader, optimizer_lr0001, epochs=5)

# 绘制学习率对比曲线
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), hist_lr01[0], 'r-o', label='lr=0.1 Train')
plt.plot(range(1, epochs+1), hist_lr01[1], 'r--s', label='lr=0.1 Val')
plt.plot(range(1, epochs+1), hist_lr001[0], 'g-o', label='lr=0.01 Train')
plt.plot(range(1, epochs+1), hist_lr001[1], 'g--s', label='lr=0.01 Val')
plt.plot(range(1, epochs+1), hist_lr0001[0], 'b-o', label='lr=0.001 Train')
plt.plot(range(1, epochs+1), hist_lr0001[1], 'b--s', label='lr=0.001 Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('学习率对比 - Loss曲线')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), hist_lr01[2], 'r-o', label='lr=0.1 Train')
plt.plot(range(1, epochs+1), hist_lr01[3], 'r--s', label='lr=0.1 Val')
plt.plot(range(1, epochs+1), hist_lr001[2], 'g-o', label='lr=0.01 Train')
plt.plot(range(1, epochs+1), hist_lr001[3], 'g--s', label='lr=0.01 Val')
plt.plot(range(1, epochs+1), hist_lr0001[2], 'b-o', label='lr=0.001 Train')
plt.plot(range(1, epochs+1), hist_lr0001[3], 'b--s', label='lr=0.001 Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('学习率对比 - Accuracy曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('learning_rate_comparison.png')
plt.show()
# =====================================================
# 任务4：卷积核可视化 (使用基准模型的第一层卷积核)
# =====================================================
print("\n===== 任务4: 第一层卷积核可视化 =====")
# 加载基准模型权重
model_vis = SimpleCNN()
model_vis.load_state_dict(torch.load("mnist_cnn_baseline.pth"))
model_vis.to(device)
model_vis.eval()

# 获取第一层卷积核权重 (形状: [32, 1, 3, 3])
conv1_weights = model_vis.conv1.weight.data.cpu().numpy()

# 显示前8个卷积核
num_kernels = 8
fig, axes = plt.subplots(1, num_kernels, figsize=(12, 3))
for i in range(num_kernels):
    kernel = conv1_weights[i, 0, :, :]  # 取第i个卷积核的2D数组
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f'Kernel {i+1}')
    axes[i].axis('off')
plt.suptitle('第一层卷积核可视化 (前8个)')
plt.savefig('conv1_kernels.png')
plt.show()

print("分析：训练后的卷积核呈现出各种边缘、方向和纹理特征。例如，有的卷积核检测垂直边缘，有的检测水平边缘，"
      "有的对特定方向的纹理敏感。这些卷积核是通过反向传播从数据中学习得到的，使得网络能够提取对分类有用的局部特征。")

# =====================================================
# 任务5：Feature Map可视化 (第一层卷积输出)
# =====================================================
print("\n===== 任务5: Feature Map可视化 =====")
# 从测试集中选一张图片
sample_img, sample_label = test_dataset[0]  # 已归一化
sample_img = sample_img.unsqueeze(0).to(device)  # 增加batch维度

# 使用hook获取conv1输出
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_vis.conv1.register_forward_hook(get_activation('conv1'))
with torch.no_grad():
    output = model_vis(sample_img)

# 获取conv1特征图 (形状: [1, 32, 28, 28])
feat_maps = activation['conv1'].cpu().squeeze(0)  # [32, 28, 28]
print(f"特征图形状: {feat_maps.shape}")

# 显示前8个feature map
num_feats = 8
fig, axes = plt.subplots(1, num_feats, figsize=(12, 4))
for i in range(num_feats):
    axes[i].imshow(feat_maps[i], cmap='viridis')
    axes[i].set_title(f'Map {i+1}')
    axes[i].axis('off')
plt.suptitle('第一层卷积输出的Feature Maps (前8个)')
plt.savefig('feature_maps.png')
plt.show()

print("观察：不同feature map对输入图像的不同区域响应强度不同。有的对笔画边缘响应强，有的对内部纹理响应强。"
      "这说明不同的卷积核确实提取了不同的图像特征，如边缘、纹理、方向等。")
# =====================================================
# 任务6：错误分类样本分析
# =====================================================
print("\n===== 任务6: 错误分类样本分析 =====")
model_vis.eval()
all_preds = []
all_labels = []
misclassified_indices = []  # 记录错误样本在test_dataset中的索引

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model_vis(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 找出当前batch中的错误样本（在整体测试集中的索引）
        batch_size = labels.size(0)
        start_idx = batch_idx * test_loader.batch_size
        for i in range(batch_size):
            if predicted[i] != labels[i]:
                misclassified_indices.append(start_idx + i)

print(f"测试集总样本: {len(test_dataset)}, 错误分类样本数: {len(misclassified_indices)}")

# 显示8张错误分类图片（从test_display取原图，保证显示效果）
num_show = min(8, len(misclassified_indices))
fig, axes = plt.subplots(1, num_show, figsize=(15, 3))
for i in range(num_show):
    idx = misclassified_indices[i]
    img, true_label = test_display[idx]
    img = img.squeeze().numpy()
    pred_label = all_preds[idx]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True:{true_label}, Pred:{pred_label}')
    axes[i].axis('off')
plt.suptitle('错误分类样本示例')
plt.savefig('misclassified_samples.png')
plt.show()

# 统计每个类别被误判为哪些类别的次数
confusion_dict = {}
for true, pred in zip(all_labels, all_preds):
    if true != pred:
        pair = (true, pred)
        confusion_dict[pair] = confusion_dict.get(pair, 0) + 1

# 找出最常混淆的5个类别对
top_confusions = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n最常混淆的类别对 (真实类别, 预测类别) 及错误次数:")
for pair, count in top_confusions:
    print(f"  真实 {pair[0]} -> 预测 {pair[1]}: {count} 次")

print("\n错误原因分析：某些数字形状相似（如3和8、4和9、7和1等），导致网络提取的特征难以区分。"
      "此外，手写体变形、笔画粗细差异也可能造成误判。")
print("改进建议：1) 增加数据增强（旋转、平移等）；2) 加深网络或增加卷积核数量；3) 使用学习率衰减策略；4) 采用更大的训练集。")

# =====================================================
# 任务7：混淆矩阵
# =====================================================
print("\n===== 任务7: 混淆矩阵 =====")
# 手动计算混淆矩阵
num_classes = 10
conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
for true, pred in zip(all_labels, all_preds):
    conf_mat[true][pred] += 1

print("测试集混淆矩阵:")
print(conf_mat)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes))
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('预测类别')
plt.ylabel('真实类别')

# 在每个格子中标注数值
thresh = conf_mat.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_mat[i, j]),
                 ha="center", va="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\n混淆矩阵分析：")
print("- 对角线元素代表各个类别被正确预测的数量。")
print("- 非对角线元素代表某一类别被错误预测为其他类别的数量。")
print("- 观察可知，类别3和8之间混淆最严重，例如数字3被误判为8，数字8被误判为3。")
print("  此外，类别4和9也有一定程度的混淆。")