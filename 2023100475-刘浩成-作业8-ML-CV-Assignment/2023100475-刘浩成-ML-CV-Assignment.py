


"""
传统机器学习在手写数字分类中的应用
使用 sklearn digits 数据集，比较 KNN, 朴素贝叶斯, 逻辑回归, SVM, 决策树, 随机森林
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

# 设置中文字体（用来在图中显示中文，如果系统不支持可忽略）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
digits = load_digits()
X, y = digits.data, digits.target   # X.shape = (1797, 64), y.shape = (1797,)

print(f"样本数量: {X.shape[0]}")
print(f"特征维度: {X.shape[1]} (每张8x8图像展开为64维向量)")
print(f"类别标签: {np.unique(y)}")

# 显示若干样本图像并保存
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {digits.target[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.show()

# 2. 划分训练集和测试集（25% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 3. 定义模型（标准化与否）
models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Logistic Regression': make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
),
    'SVM (RBF)': make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    ),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest (100)': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 训练和评估
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: 测试准确率 = {acc:.4f}")

# 4. 绘制准确率对比表（控制台输出）
print("\n" + "="*50)
print(f"{'模型':<25} {'测试准确率':>10}")
print("="*50)
for name, acc in results.items():
    print(f"{name:<25} {acc:>10.4f}")
print("="*50)

# 5. 混淆矩阵（选择表现最好的 SVM 模型）
svm_model = models['SVM (RBF)']
y_pred_svm = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title('SVM 混淆矩阵 (归一化)', fontsize=14)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 6. 错误样本分析（SVM）
errors = np.where(y_pred_svm != y_test)[0]
print(f"\nSVM 错分样本数: {len(errors)} / {len(y_test)}")
# 显示前几个错误样本
n_show = min(4, len(errors))
fig, axes = plt.subplots(1, n_show, figsize=(10, 3))
if n_show == 1:
    axes = [axes]
for idx, ax in enumerate(axes):
    err_idx = errors[idx]
    ax.imshow(X_test[err_idx].reshape(8, 8), cmap='gray')
    ax.set_title(f'真:{y_test[err_idx]}→预:{y_pred_svm[err_idx]}')
    ax.axis('off')
plt.suptitle('部分错误分类样本 (SVM)')
plt.tight_layout()
plt.savefig('error_samples.png', dpi=150)
plt.show()

