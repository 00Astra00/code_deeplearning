import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


"""
# ResNet 模型(需要 GPU)
# 残差网络核心思想：每个附加层都应更容易包含原始函数作为其元素之一
# 只有较复杂的函数类包含较小的函数类时，才能确保提高性能(图7.6.1)

class Residual(nn.Module): # @save
    # use_1x1conv: 是否使用1x1卷积来调整捷径连接的维度
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 两个批归一化层，分别对应两个卷积层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 残差连接核心: 主路径输出F(x) + 捷径连接x
        # 学习 F(x) = H(x) - x 而不是直接学习H(x)
        Y += X
        return F.relu(Y)

# blk = Residual(3,3)
# X = torch.rand(4, 3, 6, 6)
# print(blk(X), "\n", blk(X).shape)


# ResNet 模型实现
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
     nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块
# 第一个模块通道数 == 输入通道数。由于已使用步幅为2最大汇聚层，无须减小高和宽
# 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
# num_channels: 该模块的输出通道数(每个残差块的通道数)；num_residuals: 该模块包含的残差块数量
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 只有在不是第一个模块且是该模块的第一个残差块时，才进行下采样
        if i == 0 and not first_block:
            # 下采样残差块: 使用1x1卷积调整捷径连接，stride=2进行空间下采样
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 空间尺寸不断减小，通道数不断翻倍，除 b2 外每个残差模块第一个残差块进行下采样
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))

# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
    # X = layer(X)
    # print(layer.__class__.__name__,'output shape:\t', X.shape)

# 每层输出形状结果：
# Sequential output shape: torch.Size([1, 64, 56, 56])
# Sequential output shape: torch.Size([1, 64, 56, 56])
# Sequential output shape: torch.Size([1, 128, 28, 28])
# Sequential output shape: torch.Size([1, 256, 14, 14])
# Sequential output shape: torch.Size([1, 512, 7, 7])
# AdaptiveAvgPool2d output shape: torch.Size([1, 512, 1, 1])
# Flatten output shape: torch.Size([1, 512])
# Linear output shape: torch.Size([1, 10])


# 训练 ResNet 模型
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 习题3
class ResidualBottleneck(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 路径一：主路径（Bottleneck设计）
        # 第一阶段：1x1卷积，降维，通常降为 num_channels // 4
        self.conv1 = nn.Conv2d(input_channels, num_channels // 4, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels // 4)
        # 第二阶段：3x3卷积，核心特征提取
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels // 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels // 4)
        # 第三阶段：1x1卷积，升维到目标通道数
        self.conv3 = nn.Conv2d(num_channels // 4, num_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        # 路径二：快捷连接
        if use_1x1conv:
            # 当需要改变维度时，使用1x1卷积
            self.conv4 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None

    def forward(self, X): # 主路径
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y)) # 注意：最后一个BN后没有立即接ReLU
        if self.conv4: # 快捷连接
            X = self.conv4(X)
        Y += X # 残差连接
        return F.relu(Y) # 最终激活

# 普通情况：维度不变
blk = ResidualBottleneck(256, 256)
X = torch.randn(4, 256, 6, 6)
print("Output shape:", blk(X).shape) # [4, 256, 6, 6]

# 下采样情况：通道数翻倍，高宽减半
blk2 = ResidualBottleneck(256, 512, use_1x1conv=True, strides=2)
print("Output shape with downsample:", blk2(X).shape) # [4, 512, 3, 3]
"""



""" 
# DenseNet 模型
# ResNet 将 f 分解为两部分：一个简单的线性项和一个复杂的非线性项
# DenseNet 的改进在于，DenseNet 输出是连接[, ]，而不是简单相加
# 稠密网络构成：稠密块:定义如何连接输入输出；过渡层:控制通道，使其不太复杂

# 稠密块实现
def conv_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(), nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    # num_channels：每个卷积块输出的通道数，也就是增长速率
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 第i个卷积块的输入通道数 = 初始输入通道数 + 之前所有卷积块的输出通道数之和
            # 这是因为每个卷积块的输出都会与之前的所有特征图在通道维度拼接
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

# 执行流程分析（以示例参数为例）：
# 输入 X 形状：[4, 3, 8, 8]（batch=4, 通道=3, 高宽=8x8）
# 第一次循环 (i=0)：
# 第一个卷积块：输入通道数 = 10*0 + 3 = 3
# Y = blk(X)  # Y的形状：[4, 10, 8, 8]
# X = torch.cat((X, Y), dim=1)  # 新X形状：[4, 3+10=13, 8, 8]
# 第二次循环 (i=1)：
# 第二个卷积块：输入通道数 = 10*1 + 3 = 13
# Y = blk(X)  # Y的形状：[4, 10, 8, 8]
# X = torch.cat((X, Y), dim=1)  # 新X形状：[4, 13+10=23, 8, 8]
# 最终输出：print(Y.shape) # [4, 23, 8, 8]
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)

# 过渡层实现
def transition_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
    nn.Conv2d(input_channels, num_channels, kernel_size=1), nn.AvgPool2d(kernel_size=2, stride=2))

blk = transition_block(23, 10)
# print(blk(Y).shape) # torch.Size([4, 10, 4, 4])


# 构造 DenseNet 网络
# b1 作用：快速提取底层特征，降低计算量
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
     nn.BatchNorm2d(64), nn.ReLU(),nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels 为当前的通道数
# growth_rate 为每个DenseBlock中每个卷积块输出的通道数
num_channels, growth_rate, blks = 64, 32, []
# 定义4个 DenseBlock，每个包含4个卷积块
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # num_channels 在操作之前代表上一个稠密块的输出通道数
    # 每个DenseBlock输出通道数 = 输入通道数 + (卷积块数量 × 增长速率)
    num_channels += num_convs * growth_rate
    # 如果不是最后一个DenseBlock，就添加过渡层
    # 在稠密块之间添加一个过渡层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(b1, *blks, nn.BatchNorm2d(num_channels), nn.ReLU(),
nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(num_channels, 10))

# 训练 DenseNet 模型
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""