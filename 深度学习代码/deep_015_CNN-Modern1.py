import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


""" 
# AlexNet 卷积神经网络(需要GPU)
# 1. 八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层
# 2. AlexNet 使用 ReLU 而不是 sigmoid 作为其激活函数

# 使用11*11更大窗口捕捉对象，步幅为4，输出通道的数目远大于LeNet
net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
# 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
# 使用三个连续的卷积层和较小的卷积窗口，除了最后的卷积层，输出通道的数量进一步增加，在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
# 全连接层的输出数量是LeNet中的好几倍，使用dropout层来减轻过拟合
nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
# 最后是输出层，由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
nn.Linear(4096, 10))

X = torch.randn(1, 1, 224, 224)
# for layer in net:
    # X=layer(X)
    # print(layer.__class__.__name__,'output shape:\t',X.shape)

# 每层输出形状结果：
# Conv2d output shape:    torch.Size([1, 96, 54, 54])
# ReLU output shape:      torch.Size([1, 96, 54, 54])
# MaxPool2d output shape: torch.Size([1, 96, 26, 26])
# Conv2d output shape:    torch.Size([1, 256, 26, 26])
# ReLU output shape:      torch.Size([1, 256, 26, 26])
# MaxPool2d output shape: torch.Size([1, 256, 12, 12])
# Conv2d output shape:    torch.Size([1, 384, 12, 12])
# ReLU output shape:      torch.Size([1, 384, 12, 12])
# Conv2d output shape:    torch.Size([1, 384, 12, 12])
# ReLU output shape:      torch.Size([1, 384, 12, 12])
# Conv2d output shape:    torch.Size([1, 256, 12, 12])
# ReLU output shape:      torch.Size([1, 256, 12, 12])
# MaxPool2d output shape: torch.Size([1, 256, 5, 5])
# Flatten output shape:   torch.Size([1, 6400])
# Linear output shape:    torch.Size([1, 4096])
# ReLU output shape:      torch.Size([1, 4096])
# Dropout output shape:   torch.Size([1, 4096])
# Linear output shape:    torch.Size([1, 4096])
# ReLU output shape:      torch.Size([1, 4096])
# Dropout output shape:   torch.Size([1, 4096])
# Linear output shape:    torch.Size([1, 10])

# 读取数据集
batch_size = 128 # 为适应 AlexNet，调用 resize 函数将 mnist 图片重置为 224*224
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 调用 train_ch6 函数训练 AlexNet 模型(net 有所调整)
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device): # @save
    # 参数初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device) # 将数据移动到指定设备
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train() # 调整为训练模式

        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # 累加损失和精度（乘以批量大小以正确平均）
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(train_l, train_acc, None))
                d2l.plt.show()

        # 每个epoch结束后在测试集上评估精度
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        d2l.plt.show()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""



""" 
# VGG 卷积神经网络(需要 GPU)
# 原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。第一个模块有64个输出通道，
# 每个后续模块将输出通道数量翻倍，直到数字达到512。该网络使用8个卷积层和3个全连接层，因此通常被称为VGG‐11

# vgg_block 函数实现 VGG 块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs): # 只需遍历即可，不用提出具体变量
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers) # “*” 表示使用解包操作符打印列表所有元素

# 超参数变量conv_arch，指定每个VGG块里卷积层个数和输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 实现 VGG 网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels # 使得每个下层输入通道数都等于上层输出通道数
    # 全连接层部分
    return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
    nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))

# net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
    # X = blk(X)
    # print(blk.__class__.__name__,'output shape:\t',X.shape)

# 每层输出形状结果：
# Sequential output shape: torch.Size([1, 64, 112, 112])
# Sequential output shape: torch.Size([1, 128, 56, 56])
# Sequential output shape: torch.Size([1, 256, 28, 28])
# Sequential output shape: torch.Size([1, 512, 14, 14])
# Sequential output shape: torch.Size([1, 512, 7, 7])
# Flatten output shape:    torch.Size([1, 25088])
# Linear output shape:     torch.Size([1, 4096])
# ReLU output shape:       torch.Size([1, 4096])
# Dropout output shape:    torch.Size([1, 4096])
# Linear output shape:     torch.Size([1, 4096])
# ReLU output shape:       torch.Size([1, 4096])
# Dropout output shape:    torch.Size([1, 4096])
# Linear output shape:     torch.Size([1, 10])

# 训练 VGG 模型
ratio = 4 # 减少通道数，专门为 Fashion_MINST 数据集设计
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 调用 train_ch6 函数训练 VGG 网络(net 有所调整)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""



""" 
# NiN 卷积神经网络(需要GPU)
# AlexNet 和 VGG 对 LeNet 的改进主要在扩大和加深这两个模块，然而如果使用了全连接层，
# 可能会完全放弃表征的空间结构。NiN 提供简单的解决方案：在每个像素的通道上使用多层感知机

# nin_block 函数实现 NiN 块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

# NiN 网络实现(完全取消全连接层)
# 注：nn.AdaptiveAvgPool2d() 自适应平均池化层，主要优势：
# 尺寸无关性：处理任意输入尺寸；参数效率：大幅减少全连接层参数
# 防止过拟合：减少模型复杂度；全局上下文：每个通道捕获全局信息
net = nn.Sequential(nin_block(1, 96, kernel_size=11, strides=4, padding=0), nn.MaxPool2d(3, stride=2),
nin_block(96, 256, kernel_size=5, strides=1, padding=2), nn.MaxPool2d(3, stride=2),
nin_block(256, 384, kernel_size=3, strides=1, padding=1), nn.MaxPool2d(3, stride=2),
nn.Dropout(0.5), nin_block(384, 10, kernel_size=3, strides=1, padding=1), nn.AdaptiveAvgPool2d((1, 1)),
nn.Flatten()) # 将四维的输出转成二维的输出，其形状为(批量大小,10)

X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
    # X = layer(X)
    # print(layer.__class__.__name__,'output shape:\t', X.shape)

# 每层输出形状结果：
# Sequential output shape:        torch.Size([1, 96, 54, 54])
# MaxPool2d output shape:         torch.Size([1, 96, 26, 26])
# Sequential output shape:        torch.Size([1, 256, 26, 26])
# MaxPool2d output shape:         torch.Size([1, 256, 12, 12])
# Sequential output shape:        torch.Size([1, 384, 12, 12])
# MaxPool2d output shape:         torch.Size([1, 384, 5, 5])
# Dropout output shape:           torch.Size([1, 384, 5, 5])
# Sequential output shape:        torch.Size([1, 10, 5, 5])
# AdaptiveAvgPool2d output shape: torch.Size([1, 10, 1, 1])
# Flatten output shape:           torch.Size([1, 10])

# 训练 NiN 模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""



""" 
# GoogleNet 卷积神经网络(需要 GPU)
# GoogLeNet中基本的卷积块被称为Inception块
# Inception块由四条并行路径组成(详见图7.4.1)。前三条路径使用窗口大小为1 × 1、3 × 3和5 × 5的卷积层，从不同空间大小中提取信息
# 中间的两条路径在输入上执行1 × 1卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用3 × 3最大汇聚层，然后使用1 × 1卷积层来改变通道数
# Inception块中，通常调整的超参数是每层输出通道数

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数，c2和c3是元组，因为它们的路径包含两个卷积层
    # **kwargs：关键字参数包，接收所有未明确指定的命名参数，允许传递额外参数给父类
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        # super(...)作用 ：返回代理对象，调用父类方法；Inception: 当前子类名；self: 当前实例
        # __init__(**kwargs) 作用：调用父类 nn.Module 的构造函数，完成神经网络模块的基础初始化
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出 c1 + c2[1] + c3[1] + c4
        return torch.cat((p1, p2, p3, p4), dim=1)

# 实现 GoogleNet 网络(图7.4.2) (下层输入 == 上层输出，如：b3输入 = b2输出 -> c1 + c2[1] + c3[1] + c4)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(), nn.Conv2d(64, 192, kernel_size=3, padding=1),
     nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32), Inception(256, 128, (128, 192), (32, 96), 64),
     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64), Inception(512, 160, (112, 224), (24, 64), 64),
     Inception(512, 128, (128, 256), (24, 64), 64), Inception(512, 112, (144, 288), (32, 64), 64),
     Inception(528, 256, (160, 320), (32, 128), 128), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128), Inception(832, 384, (192, 384), (48, 128), 128),
     nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()) # AdaptiveAvgPool2d() 用法见上文解释
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 每层输出形状结果：
# Sequential output shape: torch.Size([1, 64, 24, 24])
# Sequential output shape: torch.Size([1, 192, 12, 12])
# Sequential output shape: torch.Size([1, 480, 6, 6])
# Sequential output shape: torch.Size([1, 832, 3, 3])
# Sequential output shape: torch.Size([1, 1024])
# Linear output shape:     torch.Size([1, 10])

# 训练 GoogleNet 模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""