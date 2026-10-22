import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l

# 批量规范化可持续加速深层网络的收敛速度
# 批量规范化：x_ = a * (x - u) / o + b
# a，b 为系数，u，o 分别为平均值和标准差

# moving_mean, moving_var：用于推理的移动平均均值和方差（running mean/var）
# eps：防止除零的极小常数；momentum：用于更新移动平均的衰减系数（越接近1，历史越重要）
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练分支，X的维度要么是2(全连接层)，要么是4(卷积)，否则抛错
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # mean = X.mean(dim=(0,2,3), keepdim=True)：对批次、高度、宽度三个维度求均值，
            # 只保留通道维，并用 keepdim=True 保留维度以便后续广播。结果形状 (1,channels,1,1)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差(momentum 大小决定了历史数据的重要性)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

# 处理数据移动到训练设备、分配和初始化任何变量、跟踪移动平均线均值和方差等
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 根据 num_dims 选择形状，以便后续广播输入 X
        if num_dims == 2: # 全连接层
            shape = (1, num_features)
        else: # 卷积层
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将 moving_mean 和 moving_var 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的 moving_mean 和 moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma,
        self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

# 将 BatchNorm 应用于LeNet模型，批量规范化是在卷积层或全连接层之后、相应的激活函数之前用的
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
nn.AvgPool2d(kernel_size=2, stride=2), nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4),
nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(), nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2),
nn.Sigmoid(),nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(), nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 第一个批量规范化层中学到的拉伸参数 gamma 和偏移参数 beta
# print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))

# 简洁实现，调用 d2l 库内部的 BatchNorm1d 或 BatchNorm2d 函数，速度更快
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(), nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
nn.Linear(84, 10))
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())