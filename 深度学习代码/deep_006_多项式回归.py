import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 创建长度为20的零向量
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 只有前4项有非零权重
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features) # 随机打乱数据
# 通过 power 函数创建从 x^0 到 x^19 的所有多项式项
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# 构造真实多项式：y = 5 + 1.2x - 3.4x²/2! + 5.6x³/3!
for i in range(max_degree): # 标准化处理
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w) # 矩阵乘法得到真实值
labels += np.random.normal(scale=0.1, size=labels.shape)
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
torch.float32) for x in [true_w, features, poly_features, labels]]
# print(features[:2], poly_features[:2, :], labels[:2])

# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss): # @save
    metric = d2l.Accumulator(2) # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X) # 预测值
        y = y.reshape(out.shape) # 真实值
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    # reduction='none' 表示不自动求平均，返回每个样本的损失
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    # 创建训练和测试数据迭代器，将标签重塑为列向量
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01) # 使用随机梯度下降优化器
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
        xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# 首先使用三阶多项式函数，该模型能有效降低训练损失和测试损失(选择前4个维度，即1,x,x^2/2!,x^3/3!)
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# 线性函数拟合，最后一个迭代周期完成后，训练损失仍然很高，线性模型容易欠拟合(选择前2个维度，即1和x)
# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# 阶数过高的多项式来训练模型，这个过于复杂的模型会轻易受到训练数据中噪声的影响，测试损失仍然很高，对数据造成过拟合(选取所有维度)
# train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)