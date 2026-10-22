import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l

""" 
# 隐藏层 & 激活函数(RELU | sigmoid | tanh)显示
# retain_graph=True 参数用于在反向传播后保留计算图，以便能够多次调用 backward() 方法
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
x.grad.data.zero_() # 清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
x.grad.data.zero_() # 清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
"""



"""
# 从零开始实现 MLP
batch_size = 256 # 获得训练、测试数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# 定义激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 训练函数(与 softmax 完全相同)
num_epochs, lr = 10, 0.50
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()

# 预测函数
d2l.predict_ch3(net, test_iter)
"""



""" 
# MLP 简洁实现
net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256),
      nn.ReLU(),nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear: # 只对线性层进行操作
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights) # 递归地对网络中的所有模块应用初始化函数
batch_size, lr, num_epochs = 256, 0.50, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
"""



"""
# 如果模型不能降低训练误差，这种现象被称为欠拟合
# 模型在训练数据上拟合的比在潜在分布中更接近的现象称为过拟合
# 用于对抗过拟合的技术称为正则化（regularization）

# 可以将数据分成三份，除了训练和测试数据集之外，还增加一个验证数据集
# K折交叉验证：原始训练数据被分成K个不重叠的子集，然后执行K次模型训练和验证，
# 每次在 K − 1 个子集上进行训练，并在剩余的一个子集上进行验证，
# 最后通过对K次实验的结果取平均来估计训练和验证误差
"""