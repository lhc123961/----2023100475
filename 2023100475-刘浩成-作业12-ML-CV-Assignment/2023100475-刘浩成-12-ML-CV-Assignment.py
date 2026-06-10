import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Sinusoidal Position Encoding
# ============================
def sinusoidal_position_encoding(max_len, d_model):
    """
    生成 Transformer 经典正弦位置编码矩阵
    参数:
        max_len: 最大序列长度
        d_model: 词向量维度（必须是偶数）
    返回:
        pe: shape (max_len, d_model) 的位置编码矩阵
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    # 分母中的指数项：10000^(2i/d_model)
    div_term = np.exp(
        -np.log(10000.0) * np.arange(0, d_model, 2) / d_model
    )
    # 偶数维度用 sin
    pe[:, 0::2] = np.sin(position * div_term)
    # 奇数维度用 cos
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 示例：生成一个小的位置编码并查看
pe_example = sinusoidal_position_encoding(max_len=10, d_model=8)
print("Sinusoidal PE shape:", pe_example.shape)
print("前两个位置的前4维:\n", pe_example[:2, :4])

# ============================
# 2. 二维向量旋转（RoPE 基础）
# ============================
def rotate_2d(x, theta):
    """
    对一个二维向量 x 旋转角度 theta
    参数:
        x: shape (..., 2) 的向量
        theta: 旋转角度（标量）
    返回:
        旋转后的向量
    """
    cos = np.cos(theta)
    sin = np.sin(theta)
    # 旋转矩阵 [[cos, -sin], [sin, cos]]
    x_rot = np.empty_like(x)
    x_rot[..., 0] = x[..., 0] * cos - x[..., 1] * sin
    x_rot[..., 1] = x[..., 0] * sin + x[..., 1] * cos
    return x_rot

# 验证二维旋转
v = np.array([1.0, 0.0])
theta = np.pi / 2
v_rot = rotate_2d(v, theta)
print("原向量:", v, "-> 旋转90度后:", np.round(v_rot, 5))  # 应接近 [0, 1]

# ============================
# 3. 高维 RoPE 实现
# ============================
def precompute_freqs(dim, end, theta=10000.0):
    """
    预计算 RoPE 所需的频率
    参数:
        dim: 每个 attention head 的维度（必须为偶数）
        end: 最大序列长度
        theta: 频率计算的基数（与原始 Transformer 一致）
    返回:
        freqs: 形状为 (end, dim//2) 的角度矩阵
    """
    # 频率指数 i 只取偶数维度的一半
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    # 序列中每个位置的 m
    m = np.arange(end)
    # 外积得到每个位置每个维度对的角度
    angles = np.outer(m, freqs)
    return angles

def apply_rotary_pos_emb(x, angles):
    """
    对输入 x 施加旋转位置编码 (RoPE)
    参数:
        x: 输入张量，形状为 (seq_len, dim)
        angles: 形状为 (seq_len, dim//2) 的角度矩阵
    返回:
        旋转后的张量，形状与 x 相同
    """
    seq_len, dim = x.shape
    # 将 x 按最后维度两两分组 (dim 必须是偶数)
    x_reshaped = x.reshape(seq_len, dim // 2, 2)
    cos = np.cos(angles)  # (seq_len, dim//2)
    sin = np.sin(angles)
    # 扩展 cos, sin 维度以进行广播运算
    cos = cos[:, :, np.newaxis]  # (seq_len, dim//2, 1)
    sin = sin[:, :, np.newaxis]
    # 应用旋转
    rot_x = np.empty_like(x_reshaped)
    rot_x[..., 0] = x_reshaped[..., 0] * cos[..., 0] - x_reshaped[..., 1] * sin[..., 0]
    rot_x[..., 1] = x_reshaped[..., 0] * sin[..., 0] + x_reshaped[..., 1] * cos[..., 0]
    return rot_x.reshape(seq_len, dim)

# 示例：对随机向量应用 RoPE
seq_len, dim = 5, 8
x = np.random.randn(seq_len, dim)
angles = precompute_freqs(dim, seq_len)
x_rope = apply_rotary_pos_emb(x, angles)
print("RoPE 输入形状:", x.shape, "输出形状:", x_rope.shape)

# ============================
# 4. E+pos 与 RoPE 输入方式对比
# ============================
def add_pos_encoding(x, pos_enc):
    """
    最简单的 E+pos 方式：直接将位置编码加到词向量上
    """
    return x + pos_enc

# 生成示例数据
d_model = 16
seq_len = 6
x_demo = np.random.randn(seq_len, d_model) * 0.1  # 模拟词向量
pe_demo = sinusoidal_position_encoding(seq_len, d_model)

# E+pos 注入
x_with_pos = add_pos_encoding(x_demo, pe_demo)
print("E+pos 后向量 (位置 0 和位置 1 的前 4 维):")
print(x_with_pos[:2, :4])

# RoPE 注入（仅作用于 Q 和 K，这里用 x 模拟某个 head 的表示）
angles_demo = precompute_freqs(d_model, seq_len)
x_rope_demo = apply_rotary_pos_emb(x_demo, angles_demo)
print("RoPE 后向量 (位置 0 和位置 1 的前 4 维):")
print(x_rope_demo[:2, :4])

# ============================
# 5. 数值实验：验证 RoPE 的相对位置性质
# ============================
def compute_attention_scores(q, k):
    """
    计算缩放点积注意力分数
    q, k: shape (seq_len, dim)
    返回: shape (seq_len, seq_len) 的注意力分数矩阵
    """
    d = q.shape[-1]
    scores = np.dot(q, k.T) / np.sqrt(d)
    return scores

# 参数设置
seq_len = 8
dim = 16
# 随机生成 query 和 key （模拟线性投影后的结果）
q = np.random.randn(seq_len, dim)
k = np.random.randn(seq_len, dim)

# --- RoPE 方式 ---
angles = precompute_freqs(dim, seq_len)
q_rope = apply_rotary_pos_emb(q, angles)
k_rope = apply_rotary_pos_emb(k, angles)

scores_rope = compute_attention_scores(q_rope, k_rope)

# --- E+pos 方式（直接加到 q, k 上）---
pe = sinusoidal_position_encoding(seq_len, dim)
q_add = q + pe
k_add = k + pe
scores_add = compute_attention_scores(q_add, k_add)

# --- 分析相对位置性质 ---
# 对 RoPE 而言，注意力分数理论上只依赖于 query 和 key 的位置差
# 这里绘制差值矩阵，看是否具有循环特性
def position_difference_matrix(seq_len):
    """返回一个矩阵，其中 (i,j) 元素为 i - j"""
    pos = np.arange(seq_len)
    return pos[:, np.newaxis] - pos[np.newaxis, :]

diff = position_difference_matrix(seq_len)
print("\n位置差矩阵 (i - j):")
print(diff)

print("\nRoPE 注意力分数矩阵:")
print(np.round(scores_rope, 3))
print("\nE+pos 注意力分数矩阵:")
print(np.round(scores_add, 3))

# 直观展示：将得分按位置差分组求平均，观察趋势
diff_flat = diff.flatten()
scores_rope_flat = scores_rope.flatten()
scores_add_flat = scores_add.flatten()

unique_diffs = np.unique(diff_flat)
mean_rope_by_diff = [np.mean(scores_rope_flat[diff_flat == d]) for d in unique_diffs]
mean_add_by_diff = [np.mean(scores_add_flat[diff_flat == d]) for d in unique_diffs]

plt.figure(figsize=(8, 4))
plt.plot(unique_diffs, mean_rope_by_diff, 'o-', label='RoPE')
plt.plot(unique_diffs, mean_add_by_diff, 's--', label='E+pos')
plt.xlabel('相对位置 (i - j)')
plt.ylabel('平均注意力分数')
plt.title('注意力分数与相对位置的关系')
plt.legend()
plt.grid(True)
plt.show()

# 进一步验证：对两个完全相同的向量序列应用 RoPE，
# 其注意力分数应当仅依赖于相对位置，且形成对角线主导模式。
q_same = np.ones((seq_len, dim))  # 所有位置内容相同
k_same = np.ones((seq_len, dim))
q_same_rope = apply_rotary_pos_emb(q_same, angles)
k_same_rope = apply_rotary_pos_emb(k_same, angles)
scores_same = compute_attention_scores(q_same_rope, k_same_rope)
print("\n内容完全相同时的 RoPE 注意力分数（应完美体现相对位置）:")
print(np.round(scores_same, 3))

# 检查是否只与相对位置有关：沿每条对角线（相同 i-j）值应完全相同
print("对角线一致性检查（最大差异）:")
diag_values = []
for offset in range(-seq_len+1, seq_len):
    diag = np.diagonal(scores_same, offset=offset)
    diag_values.append((offset, np.max(diag) - np.min(diag)))
    print(f"偏移 {offset}: 差异 {np.max(diag) - np.min(diag):.6f}")

