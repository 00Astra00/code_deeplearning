import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
import torch.nn as nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


"""
# 卷积神经网络原则：
# 图像的平移不变性使得以相同方式处理局部图像，而不在乎位置
# 局部性意味着计算相应的隐藏表示只需一小部分局部图像像素
# 在图像处理中，卷积层通常比全连接层需要参数更少，但模型仍高效用
# 多个输入和输出通道使模型在每个空间位置可以获取图像的多方面特征
# 输出大小：(n_h − k_h + 1) × (n_w − k_w + 1)
# 对于任意元素x，感受野指在前向传播期间影响x计算的所有元素

def corr2d(X, K): # @save
    # 计算二维互相关运算
    h, w = K.shape # 获得卷积核形状并初始化结果矩阵形状
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X, K))

# corr2d函数实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 卷积层简单应用：通过像素变化位置检测图像不同颜色边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0 # 构造矩阵
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

# 学习卷积核参数
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
# 二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），其中批量大小和通道数为1
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
X, Y, lr = X.reshape((1, 1, 6, 8)), Y.reshape((1, 1, 6, 7)), 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核，更新参数
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    # print(f'epoch {i+1}, loss {l.sum():.3f}')
# print(conv2d.weight.data.reshape((1, 2)))


# 填充和步幅
# 填充：解决图像边界丢失信息现象，输出形状：(nh − kh + ph + 1) × (nw − kw + pw + 1)
# 步幅：大幅降低图像的宽度和高度，输出形状：⌊(nh − kh + ph + sh)/sh⌋ × ⌊(nw − kw + pw + sw)/sw⌋

# 定义一个计算卷积层的函数
# 此函数初始化卷积层权重，输出长和宽
def comp_conv2d(conv2d, X):
    # （1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

X = torch.rand(size=(8, 8))
# 这里每边都填充了1行或1列，因此总共添加2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# print(comp_conv2d(conv2d, X).shape) # 恢复到原来的形状(8,8)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
# print(comp_conv2d(conv2d, X).shape) # 恢复到原来的形状(8,8)

# 将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
# print(comp_conv2d(conv2d, X).shape) # 形状应为(4,4)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# print(comp_conv2d(conv2d, X).shape) # 形状应为(2,2)


# 多输入输出通道
# 多输入通道：例如每张图像的不同图层
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]) # shape:(2,3,3)
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# print(corr2d_multi_in(X, K)) # 两个通道的结果矩阵(2,2)对应位置相加

# 多输出通道：每个通道看作对不同特征的响应
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算，将所有结果都叠加
    # torch.stack(..., 0)：将所有输出通道的特征图沿着第0维度堆叠
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K = torch.stack((K, K + 1, K + 2), 0)
# print(K.shape, corr2d_multi_in_out(X, K))


# 1*1卷积层特殊之处在于只在通道维度变换，不改变空间维度
# 1*1卷积层通常用于调整网络层的通道数量和控制模型复杂性

# 使用全连接层实现 1*1 卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 将输入从 [c_i, h, w] 重塑为 [c_i, h*w]，将每个空间位置展平为一列
    X = X.reshape((c_i, h * w))
    # 将卷积核从 [c_o, c_i, 1, 1] 重塑为 [c_o, c_i] 移除 1×1 空间维度
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    # 将结果从 [c_o, h*w] 重塑回 [c_o, h, w]，恢复原始的空间维度
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3)) # 3个通道，3×3 空间尺寸
K = torch.normal(0, 1, (2, 3, 1, 1)) # 2个输出通道，3个输入通道
Y1 = corr2d_multi_in_out_1x1(X, K) # 使用全连接层方法实现的1×1卷积结果
Y2 = corr2d_multi_in_out(X, K) # 使用常规卷积方法计算的结果
# 证明1×1卷积确实等价于空间位置上的全连接操作
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6


# 汇聚层目的：降低卷积层对位置的敏感性；降低对空间降采样表示的敏感性
# 汇聚层不包含参数，并且池运算是确定的，汇聚层分为最大汇聚层和平均汇聚层

# 汇聚层前向传播函数
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size # 初始化汇聚层形状和结果矩阵 Y 的形状、数值
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# print(pool2d(X, (2, 2)), "\n", pool2d(X, (2, 2), 'avg'))

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
# 深度学习框架中默认步幅与汇聚窗口的大小相同，也可以手动设定具体大小
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
# print(pool2d(X))

# 汇聚层的输出通道数与输入通道数相同
X = torch.cat((X, X + 1), 1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# print(pool2d(X)) # 汇聚后输出通道的数量仍为 2
"""



""" 
# 聚集神经网络LeNet
# 由两个部分组成：卷积编码器：由两个卷积层组成；全连接层密集块：由三个全连接层组成
# 每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和平均汇聚层
# 为了将卷积块的输出传递给稠密块，须将这个四维输入转换成全连接层所期望的二维输入


# 实例化 LeNet 网络
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
nn.AvgPool2d(kernel_size=2, stride=2), nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
    # X = layer(X) # 循环遍历网络，输出每层结果
    # print(layer.__class__.__name__,'output shape: \t',X.shape)

# 实例化过程中各模块输出形状输出结果：
# Conv2d output shape: 	     torch.Size([1, 6, 28, 28])
# Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
# AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
# Conv2d output shape: 	     torch.Size([1, 16, 10, 10])
# Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
# AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
# Flatten output shape: 	 torch.Size([1, 400])
# Linear output shape: 	     torch.Size([1, 120])
# Sigmoid output shape: 	 torch.Size([1, 120])
# Linear output shape: 	     torch.Size([1, 84])
# Sigmoid output shape: 	 torch.Size([1, 84])
# Linear output shape: 	     torch.Size([1, 10])


# 模型训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 使用GPU计算模型在数据集上的精度
def evaluate_accuracy_gpu(net, data_iter, device=None): # @save
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式
    if not device: # 自动获取模型所在的设备（GPU/CPU）
        device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        # 遍历数据迭代器，将数据移动到指定设备
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
                y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 用GPU训练模型(在第六章定义)
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
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        d2l.plt.show()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""



# 习题 2,3,4
# 2：
class ImprovedLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedLeNet, self).__init__()
        # 1. 调整卷积窗口大小和输出通道数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # 2. 添加更多卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 3. 使用ReLU激活函数
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4. 调整全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # 5. 添加Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # 第二个卷积块
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # 第三个卷积块
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# 3.
def evaluate_accuracy_gpu(net, data_iter, device=None): # @save
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式
    if not device: # 自动获取模型所在的设备（GPU/CPU）
        device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        # 遍历数据迭代器，将数据移动到指定设备
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
                y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_improved_lenet():
    # 创建改进的LeNet模型
    net = ImprovedLeNet(num_classes=10)

    # 6. 调整学习率和训练细节
    lr, num_epochs = 0.001, 15  # 更小的学习率，更多的轮数

    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练模型
    device = d2l.try_gpu()
    net.to(device)

    animator = d2l.Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc'])
    d2l.plt.show()

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

        scheduler.step()
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)

        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        d2l.plt.show()

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, '
              f''f'Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    return net

# 4.
def visualize_activations(net, test_iter, class_names):
    # 可视化第一层和第二层的激活值
    net.eval()
    device = next(net.parameters()).device

    # 获取毛衣和外套的样本
    sweater_idx = class_names.index('pullover')
    coat_idx = class_names.index('coat')

    sweater_img, coat_img = None, None

    # 从测试集中找到样本
    for X, y in test_iter:
        for i in range(len(y)):
            if y[i].item() == sweater_idx and sweater_img is None:
                sweater_img = X[i:i + 1]
            if y[i].item() == coat_idx and coat_img is None:
                coat_img = X[i:i + 1]
        if sweater_img is not None and coat_img is not None:
            break

    # 移动到设备
    sweater_img = sweater_img.to(device)
    coat_img = coat_img.to(device)

    # 注册钩子来获取中间激活值
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # 注册钩子
    net.conv1.register_forward_hook(get_activation('conv1'))
    net.conv2.register_forward_hook(get_activation('conv2'))

    # 前向传播
    with torch.no_grad():
        _ = net(sweater_img)
        sweater_conv1 = activations['conv1'][0]
        sweater_conv2 = activations['conv2'][0]

        _ = net(coat_img)
        coat_conv1 = activations['conv1'][0]
        coat_conv2 = activations['conv2'][0]

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 显示原始图像
    axes[0, 0].imshow(sweater_img[0, 0].cpu(), cmap='gray')
    axes[0, 0].set_title('Sweater Original')
    axes[1, 0].imshow(coat_img[0, 0].cpu(), cmap='gray')
    axes[1, 0].set_title('Coat Original')

    # 显示第一层卷积的激活（前3个通道）
    for i in range(3):
        axes[0, i + 1].imshow(sweater_conv1[i].cpu(), cmap='hot')
        axes[0, i + 1].set_title(f'Sweater Conv1 Ch{i}')
        axes[1, i + 1].imshow(coat_conv1[i].cpu(), cmap='hot')
        axes[1, i + 1].set_title(f'Coat Conv1 Ch{i}')

    plt.tight_layout()
    plt.show()

    # 显示第二层卷积的激活
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    for i in range(3):
        axes[0, i].imshow(sweater_conv2[i].cpu(), cmap='hot')
        axes[0, i].set_title(f'Sweater Conv2 Ch{i}')
        axes[1, i].imshow(coat_conv2[i].cpu(), cmap='hot')
        axes[1, i].set_title(f'Coat Conv2 Ch{i}')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # 训练模型
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    improved_net = train_improved_lenet()

    # 使用示例
    class_names = ['t-shirt', 'trouser', 'pullover', 'dress',
    'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    visualize_activations(improved_net, test_iter, class_names)