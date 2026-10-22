import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l


# 通过简单的限制特征数量（在多项式回归中体现为限制阶数），可能仍使模型在过简单和过复杂徘徊
# 需要一个更细粒度的工具来调整函数的复杂性，使其达到一个合适的平衡位置，即范数与权重衰减
# 训练参数化机器学习模型时，权重衰减是最广泛使用正则化的技术之一，也被称为 L2 正则化
# L2 正则化通过函数与零的距离来衡量函数复杂度(即通过权重向量的某个范数大小)
# 要保证权重向量比较小，最常用方法是将其范数作为惩罚项加到最小化损失的问题中

# 生成随机的数据集
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
# def load_array(data_arrays, batch_size, is_train=True):
#     # 构造一个PyTorch数据迭代器
#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)



""" 
# 从零开始实现权重衰减
# 随机初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义 L2 范数惩罚(对所有项求平方后并将它们求和)
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 训练函数(唯一的变化是损失现在包括了惩罚项)
def train(lambd):
    w, b = init_params()
    # net: 定义网络为线性回归函数 ；loss: 使用平方损失函数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
          xlim=[5, num_epochs], legend=['train', 'test']) # 训练过程可视化
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 关键变化：在损失函数中添加L2正则化项
            # lambd * l2_penalty(w): L2惩罚项，lambd 控制正则化强度
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward() # 计算批次中所有样本损失总和
            d2l.sgd([w, b], lr, batch_size) # 使用随机梯度下降更新参数
        if (epoch + 1) % 5 == 0: # 每5个 epoch 进行评估与可视化
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
            d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

# 用 lambd = 0 禁用权重衰减后运行这个代码
print(train(lambd=0)) # 与下面使用权重衰减进行对比
print(train(lambd=3)) # 两个结果对比，检验权重衰减效果
d2l.plt.show()
"""



""" 
# 简洁实现权重衰退
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        # 使用正态分布初始化参数值
        param.data.normal_()
    # reduction='none' 表示不自动求平均，返回每个样本的损失
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 关键部分：创建优化器，分别设置权重和偏置的参数组
    # 第一组：网络第一层的权重，应用权重衰减 wd
    # 第二组：网络第一层的偏置，不应用权重衰减
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd}, {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

# 对比使用权重衰退的效果
# print(train_concise(0))
print(train_concise(2))
d2l.plt.show()
"""