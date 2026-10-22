import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
from torch import nn
from d2l import torch as d2l


""" 
# 非自主性提示是基于环境中物体的突出性和易见性，注意力在基于自主性提示去辅助选择时将更为谨慎
# “是否包含自主性提示”将注意力机制与全连接层或汇聚层区别开来。在注意力机制的背景下，自主性提示被称为查询
# 给定任何查询，注意力机制将选择引导至感官输入，其被称为值。每个值都与一个键配对，可以想象为感官输入的非自主提示
# 可以通过设计注意力汇聚的方式，便于给定的查询（自主性提示）与键（非自主性提示）进行匹配，这将引导得出最匹配的值
# 实际上，注意力汇聚得到的是加权平均的总和值，其中权重是在给定的查询和不同的键之间计算得出的

# show_heatmaps函数可视化注意力权重。matrices形状(显示行数，显示列数，查询数目，键数目)
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'): # @save
    # matrices: 4D张量，形状为(显示行数, 显示列数, 查询数目, 键数目)
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    # squeeze=False: 即使只有一行或一列，也保持axes为2D数组；sharex=True, sharey=True: 所有子图共享x轴和y轴刻度
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)): # 遍历每一行子图和对应的矩阵数据
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)): # 遍历当前行的每个子图和对应的矩阵
            # 显示热力图：imshow: 显示矩阵为图像，颜色深浅表示数值大小
            # matrix.detach().numpy(): 将PyTorch张量转换为NumPy数组
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    # 为所有子图添加一个共用的颜色条，shrink=0.6: 颜色条的高度为主图的60%
    fig.colorbar(pcm, ax=axes, shrink=0.6)

# 当查询和键相同时，注意力权重为1，否则为0
# .eye(10): 创建10×10单位矩阵，重塑为(1,1,10,10)
# 表示：1行子图，1列子图，10个查询，10个键
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
# show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
# d2l.plt.show()


# 生成数据集
n_train = 50 # 训练样本数
# 排序后的训练样本，返回排序后的数据和索引（索引用_忽略）
x_train, _ = torch.sort(torch.rand(n_train) * 5)
def f(x):
    return 2 * torch.sin(x) + x**0.8
# 生成带噪声的训练标签，输出生成的训练样本
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test) # 测试样本数(50)
def plot_kernel_reg(y_hat):
    # 定义核回归结果可视化函数，绘制真实曲线和预测曲线
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5) # 用圆圈标记训练数据点，alpha=0.5设置半透明
    d2l.plt.show()
"""



""" 
# 注意力汇聚：Nadaraya-Watson 核回归
# 1.平均汇聚(不用)
# torch.repeat_interleave(): 重复训练集标签 y_train 平均值
# 创建一个长度为 n_test 的向量，其中每个元素都是训练集y值的平均值
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

# 2.非参数注意力汇聚
# 通用注意力汇聚公式：f(x) = 求和(α(x, xi) * yi)
# 其中x是查询，(xi, yi)是键值对。注意力汇聚是yi的加权平均
# 将查询x和键xi之间的关系建模为注意力权重 α(x, xi)，权重将分配给对应yi
# 对于任何查询，模型在所有键值对注意力权重都是有效概率分布(非负，总和为1)
# 如果键x_i越接近查询x，分配给对应值y_i的注意力权重就越大，获得更多注意力

# x_test.repeat...(n_train): 测试集中每个x值重复n_train次
# X_repeat形状:(n_test,n_train)，每一行都包含相同的测试输入
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# nn.functional.softmax(..., dim=1)：对每一行应用softmax函数
# dim=1 表示对每个测试点，在所有训练点上计算注意力权重，确保每行的权重之和为1
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
# plot_kernel_reg(y_hat)
# d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
# xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

# 3.带参数注意力汇聚
X, Y = torch.ones((2, 1, 4)), torch.ones((2, 4, 6))
# torch.bmm：批量矩阵乘法，(n,a,b) * (n,b,c) -> (n,a,c)
# print(torch.bmm(X, Y).shape) # torch.Size([2, 1, 6])
# 计算小批量数据中的加权平均值
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
# unsqueeze(1): 在第1个维度(从0开始)插入一个新维度
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
# 计算输出结果：tensor([[[ 4.5000]], [[14.5000]]])

# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries 、values 和 attention_weights 的形状(查询个数，键值对个数)
        # 查询数据扩展以匹配键的形状：结果 queries 形状变为(查询个数, 键值对个数)
        # queries.repeat_interleave(keys.shape[1]): 将每个查询重复"键值对个数"次
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 对每一行应用softmax函数，dim=1 表示对每个测试点，在所有训练点上计算注意力权重，确保每行的权重之和为1
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# 在带参注意力汇聚中，任何一个训练样本输入都会和所有训练样本键值对计算，得到对应预测输出
# X_tile / Y_tile 的形状:(n_train，n_train)，每一行都包含着相同的训练输入/输出
X_tile, Y_tile = x_train.repeat((n_train, 1)), y_train.repeat((n_train, 1))
# 关键技巧留一法：1 - torch.eye(n_train): 单位矩阵取反
# X_tile[...]: 使用布尔掩码选择元素，排除对角线元素(即自身)
# .reshape((n_train, -1)): 重塑为(n_train, n_train-1)形状
# 对于第i个训练样本：...[i]: 包含除第i个点外的所有其他训练点的坐标
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 训练模型
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
# keys / values 形状:(n_test，n_train)，每行包含相同训练输入
keys, values = x_train.repeat((n_test, 1)), y_train.repeat((n_test, 1))
# .unsqueeze(1): 增加维度，与绘图函数兼容；.detach(): 从计算图分离，避免梯度计算
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
# plot_kernel_reg(y_hat)

# 带参数的模型加入可学习的参数后，曲线在注意力权重较大的区域变得更不平滑
# d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
# xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
"""



""" 
# 注意力评分函数
# 1.掩蔽 softmax 操作
def masked_softmax(X, valid_lens): # @save
    # 通过在最后一个轴上掩蔽元素来执行softmax操作
    # X：3D张量，通常是注意力分数矩阵
    # valid_lens：1D或2D张量，表示每个序列的有效长度
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else: # 提供了有效长度，需要进行掩蔽处理
        shape = X.shape
        # 如果valid_lens是1D张量，假设每个批次中的序列有相同的有效长度
        # torch....(valid_lens, ...): 将每个有效长度重复shape[1]次
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # 如果valid_lens是2D张量，直接展平为1D(假设已为每个序列位置提供具体有效长度)
        else:
            valid_lens = valid_lens.reshape(-1)
        # 分解步骤：X.reshape(-1, shape[-1]): 将3D张量重塑为2D
        # (batch_size, seq_len, seq_len) -> (batch_size*seq_len, seq_len)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值 value 替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# 2.加性注意力
# 当查询和键是不同长度的矢量时，可以使用加性注意力作为评分函数
# 评分函数：a(q, k) = w_v(T) * tanh(W_q * q + W_k * k) ∈ R
class AdditiveAttention(nn.Module): # @save
    # 加性注意力
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 评分层：将隐藏表示映射为单个注意力分数，每个"键-值"对只产生一个分数
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 扩展前形状(batch_size，查询/键值对个数，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features) # 激活函数 tanh
        # self.w_v 仅有一个输出，因此从形状中移除最后那个维度
        # scores 形状(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# .repeat(2, 1, 1): 在批次维度重复2次，形状(2, 10, 4) = (批次大小, 键值对个数, 值维度)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6]) # [2, 6] 表示：批次0：前2个键值对有效；批次1：前6个键值对有效
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1).eval()
# print(attention(queries, keys, values, valid_lens)) # 前向传播输出形状(2, 1, 4) = (批次大小,查询个数,值维度)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
# d2l.plt.show()

# 3.缩放点积注意力
# 用点积可以得到计算效率更高的评分函数，但是点积操作要求查询和键具有相同的长度d
# 为确保无论向量长度如何，点积方差仍是1，则评分函数为：a(q, k) = q_T * k / √d
class DotProductAttention(nn.Module): # @save
    # 缩放点积注意力
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1] # 获取查询和键的维度
        # 设置transpose_b=True为了交换keys的(1, 2)两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # attention_weights 形状：(batch_size, num_queries, num_kv_pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5).eval()
# print(attention(queries, keys, values, valid_lens))
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
# d2l.plt.show()
"""