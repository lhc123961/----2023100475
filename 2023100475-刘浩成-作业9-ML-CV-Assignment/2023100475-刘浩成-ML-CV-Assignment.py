import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# ==================== 任务1：环境准备 ====================
print("=" * 60)
print("任务1：环境准备")
print("PyTorch 版本:", torch.__version__)
print("torchvision 版本:", torchvision.__version__)
print("NumPy 版本:", np.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

# 简单张量操作测试
x = torch.randn(3, 3).to(device)
y = x + x
print("张量操作成功，结果:\n", y)
print("环境准备完成！\n")

# ==================== 任务2：加载图像数据集 ====================
print("=" * 60)
print("任务2：加载 MNIST 数据集")

# 数据预处理：转为 Tensor 并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载 MNIST
full_train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# 划分训练集和验证集 (80% 训练，20% 验证)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 显示 8 张样本图像并标注真实类别
def imshow(img, ax, title=None):
    """反标准化并显示图像"""
    img = img * 0.3081 + 0.1307
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    if title:
        ax.set_title(title)
    ax.axis('off')

data_iter = iter(train_loader)
images, labels = next(data_iter)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    imshow(images[i], ax, f"Label: {labels[i].item()}")
fig.suptitle("MNIST 样本图像", fontsize=14)
plt.tight_layout()
plt.show()

# ==================== 任务3：定义 CNN 模型 ====================
print("=" * 60)
print("任务3：定义 CNN 模型")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)                     # 28x28 → 14x14 → 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))     # 16×14×14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))     # 32×7×7
        x = x.view(x.size(0), -1)                          # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10).to(device)
print(model)
print("模型定义完成！\n")

# ==================== 辅助函数：训练一个 epoch 与评估 ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ==================== 任务4 & 任务5：训练与验证 ====================


print("=" * 60)
print("任务4 & 5：训练与验证模型")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

# ==================== 初始化记录列表 ====================
train_loss_history = []  # 用于记录每个 epoch 的训练损失
train_acc_history = []   # 用于记录每个 epoch 的训练准确率
val_loss_history = []    # 用于记录每个 epoch 的验证损失
val_acc_history = []     # 用于记录每个 epoch 的验证准确率

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}]  "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  |  "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ==================== 任务6：测试模型 ====================
print("=" * 60)
print("任务6：测试模型")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"测试集 Loss: {test_loss:.4f}, 测试集 Accuracy: {test_acc:.2f}%")

# 显示 8 张测试图像及其真实类别与预测类别
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    imshow(images[i].cpu(), ax)
    ax.set_title(f"True: {labels[i].item()}, Pred: {preds[i].item()}")
fig.suptitle("测试图像：真实类别 vs 预测类别", fontsize=14)
plt.tight_layout()
plt.show()

# ==================== 任务7：绘制训练曲线 ====================
print("=" * 60)
print("任务7：绘制训练曲线")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(1, num_epochs + 1)

ax1.plot(epochs, train_loss_history, label='Training Loss')
ax1.plot(epochs, val_loss_history, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, train_acc_history, label='Training Accuracy')
ax2.plot(epochs, val_acc_history, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ==================== 任务8：结果分析 ====================

print("=" * 50)
print("任务8：结果分析（请结合以上图表回答）")
print("1. 训练 loss 是否随着 epoch 增加而下降？")
print(f"从输出的损失值可以看到，训练 loss 从 {train_loss_history[0]:.4f} 下降到了 {train_loss_history[-1]:.4f}。")
print("   整体呈下降趋势，说明模型正在学习。")
print("2. 验证 accuracy 是否随着训练逐渐提升？")
print(f"   验证准确率从 {val_acc_history[0]:.2f}% 提升到了 {val_acc_history[-1]:.2f}% 左右，逐渐提高并趋于稳定。")
print("3. 训练 accuracy 和验证 accuracy 是否存在明显差距？")
if max(train_acc_history) - max(val_acc_history) > 2:
    print("   存在一定差距，可能存在轻微过拟合。")
else:
    print("   差距很小（通常小于1%），说明模型泛化能力较好，没有明显过拟合。")
print("4. 如果存在明显差距，可能是什么原因？")
print("   原因可能有：模型容量过大导致过拟合、训练数据太少、缺少正则化（如 dropout 或 weight decay）、训练 epoch 过多。")
print("5. 哪些数字或类别更容易被错误分类？")
print("   观察预测结果或混淆矩阵（未实现），通常数字 4 和 9、7 和 2、3 和 5 等容易混淆，因为字形相似。")
print("6. MNIST 和 CIFAR-10 哪个更难？为什么？")
print("   CIFAR-10 更难。MNIST 是简单的手写数字，背景单一，类间差异大；CIFAR-10 是彩色自然图像，")
print("   背景复杂，物体外观、光照、姿态变化很大，类内差异大，所以分类难度更高。")
print()

# ==================== 进阶任务1：修改网络结构 ====================
print("=" * 50)
print("进阶任务1：修改网络结构并比较性能")

# 定义一个改进的 CNN：增加一层卷积，全连接层不变
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 新增卷积层
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # 经过三次池化：28 -> 14 -> 7 -> 3（实际是 28/2=14, 14/2=7, 7/2=3.5 向下取整为3）
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 16x14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x7x7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 64x3x3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("训练改进模型...")
model_v2 = ImprovedCNN().to(device)
optimizer_v2 = optim.Adam(model_v2.parameters(), lr=0.001)
criterion_v2 = nn.CrossEntropyLoss()

# 训练改进模型（这里只训练 10 个 epoch 以对比）
for epoch in range(num_epochs):
    model_v2.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_v2(images)
        loss = criterion_v2(outputs, labels)
        optimizer_v2.zero_grad()
        loss.backward()
        optimizer_v2.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc_v2 = 100.0 * correct / total
    # 简单验证
    model_v2.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_v2(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_acc_v2 = 100.0 * correct_val / total_val
    if (epoch + 1) % 2 == 0:
        print(f"改进模型 Epoch {epoch+1}: Train Acc={train_acc_v2:.2f}%, Val Acc={val_acc_v2:.2f}%")

# 测试改进模型
model_v2.eval()
correct_test_v2 = 0
total_test_v2 = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_v2(images)
        _, predicted = torch.max(outputs, 1)
        total_test_v2 += labels.size(0)
        correct_test_v2 += (predicted == labels).sum().item()
test_acc_v2 = 100.0 * correct_test_v2 / total_test_v2

print(f"\n原始模型测试准确率: {test_acc:.2f}%")
print(f"改进模型测试准确率: {test_acc_v2:.2f}%")
if test_acc_v2 > test_acc:
    print("修改后准确率有所提升，说明增加网络深度可以提取更复杂的特征。")
else:
    print("修改后准确率变化不大或略有下降，可能是因为 MNIST 任务太简单，更深的网络容易过拟合或增益有限。")
print()

# ==================== 进阶任务2：比较不同优化器 ====================
print("=" * 50)
print("进阶任务2：比较 SGD 和 Adam 优化器")

# 定义一个函数来训练一个模型并返回测试准确率，避免重复代码
def train_and_evaluate(optimizer_type, lr, epochs=10):
    # 重新创建一个相同结构的模型
    temp_model = SimpleCNN().to(device)
    temp_criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        temp_optimizer = optim.SGD(temp_model.parameters(), lr=lr, momentum=0.9)
    else:  # Adam
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=lr)

    for epoch in range(epochs):
        temp_model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = temp_model(images)
            loss = temp_criterion(outputs, labels)
            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()

    # 最后测试
    temp_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = temp_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

print("使用 SGD 训练...")
sgd_acc = train_and_evaluate('SGD', lr=0.01, epochs=10)
print("使用 Adam 训练...")
adam_acc = train_and_evaluate('Adam', lr=0.001, epochs=10)

print("\n优化器比较记录表：")
print("Optimizer | Learning Rate | Test Accuracy")
print(f"SGD       | 0.01          | {sgd_acc:.2f}%")
print(f"Adam      | 0.001         | {adam_acc:.2f}%")
print("分析：Adam 通常收敛更快，对学习率不敏感；SGD 需要仔细调整学习率，但在某些情况下泛化能力略好。")
print("在 MNIST 简单任务上两者都能达到很高准确率，差别不大。\n")
# ==================== 进阶任务3：比较 MNIST 和 CIFAR-10 ====================
print("=" * 50)
print("进阶任务3：比较 MNIST 和 CIFAR-10")

# 1. 加载 CIFAR-10 数据集
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cifar_full_train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_cifar)
cifar_test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_cifar)

# 划分训练/验证集
cifar_train_size = int(0.8 * len(cifar_full_train))
cifar_val_size = len(cifar_full_train) - cifar_train_size
cifar_train, cifar_val = random_split(cifar_full_train, [cifar_train_size, cifar_val_size])

cifar_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_val_loader   = DataLoader(cifar_val, batch_size=64, shuffle=False)
cifar_test_loader  = DataLoader(cifar_test, batch_size=64, shuffle=False)

print("CIFAR-10 训练集样本数:", len(cifar_train))
print("CIFAR-10 验证集样本数:", len(cifar_val))
print("CIFAR-10 测试集样本数:", len(cifar_test))

# 2. 定义适用于 CIFAR-10 的 CNN（输入是3通道彩色图，尺寸32x32）
class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 3 通道输入
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)                # 32->16->8->4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32x16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 64x8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cifar_model = CifarCNN().to(device)
cifar_criterion = nn.CrossEntropyLoss()
cifar_optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)

print("开始训练 CIFAR-10 模型（可能需要几分钟）...")
cifar_epochs = 10
for epoch in range(cifar_epochs):
    # 训练
    cifar_model.train()
    train_loss_c = 0.0
    correct_c = 0
    total_c = 0
    for images, labels in cifar_train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = cifar_model(images)
        loss = cifar_criterion(outputs, labels)
        cifar_optimizer.zero_grad()
        loss.backward()
        cifar_optimizer.step()
        train_loss_c += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_c += labels.size(0)
        correct_c += (predicted == labels).sum().item()
    train_acc_c = 100.0 * correct_c / total_c

    # 验证
    cifar_model.eval()
    val_loss_c = 0.0
    correct_val_c = 0
    total_val_c = 0
    with torch.no_grad():
        for images, labels in cifar_val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = cifar_model(images)
            loss = cifar_criterion(outputs, labels)
            val_loss_c += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val_c += labels.size(0)
            correct_val_c += (predicted == labels).sum().item()
    val_acc_c = 100.0 * correct_val_c / total_val_c

    if (epoch + 1) % 2 == 0:
        print(f"CIFAR-10 Epoch {epoch+1}: Train Acc={train_acc_c:.2f}%, Val Acc={val_acc_c:.2f}%")

# 测试 CIFAR-10
cifar_model.eval()
correct_test_c = 0
total_test_c = 0
with torch.no_grad():
    for images, labels in cifar_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = cifar_model(images)
        _, predicted = torch.max(outputs, 1)
        total_test_c += labels.size(0)
        correct_test_c += (predicted == labels).sum().item()
cifar_test_acc = 100.0 * correct_test_c / total_test_c

print(f"\nCIFAR-10 测试准确率: {cifar_test_acc:.2f}%")

# 3. 比较两个数据集
print("\nMNIST 与 CIFAR-10 比较记录表：")
print("数据集      | 图像类型       | 类别数 | 测试准确率 | 难度")
print(f"MNIST       | 灰度手写数字   | 10     | {test_acc:.2f}%      | 低")
print(f"CIFAR-10    | 彩色自然图像   | 10     | {cifar_test_acc:.2f}%       | 高")
print("分析：CIFAR-10 是彩色自然图像，包含多种物体，光照、角度、背景变化大，因此分类难度远高于 MNIST。")

print("\n所有实验任务完成！")