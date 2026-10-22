import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy
import torch
import subprocess
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

# 注意：本节的代码需要多 GPU 运行，因此只需看代码即可

# 符号式编程，即代码通常在完全定义过程之后执行计算
# 步骤：1.定义计算流程；2.将流程编译成可执行程序；3.调用编译好的程序执行
# 通过torch.jit.script函数来转换模型，就可以编译和优化计算，而结果不变
# 1. net = torch.jit.script(net)  2. print(net(x)) # 结果不变
def get_net():
    # 生产网络的工厂模式
    net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
    return net

class Benchmark: # @save
    # 定义一个度量时间的类，用于测量运行时间
    def __init__(self, description='Done'):
        self.description = description
    def __enter__(self):
        self.timer = d2l.Timer()
        return self
    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

net = get_net()
x = torch.randn(size=(1, 512))
with Benchmark('无torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('有torchscript'):
    for i in range(1000): net(x)


# 异步计算
# NumPy点积在CPU执行，PyTorch矩阵乘法在GPU执行，后者速度要快得多
# 前端线程和后端线程间交互概括如下：1.前端命令后端将计算任务插入队列
# 2.然后后端从队列接收计算任务并执行；3.然后后端将计算结果返回到前端


# 自动并行
# 深度学习框架会在后端自动构建计算图。利用计算图，系统可以了解所有
# 依赖关系，并且可以选择性地并行执行多个不相互依赖的任务以提高速度
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]
x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

# 通过在测量之前需要预热设备来确保缓存的作用不影响最终的结果
# .cuda.synchronize()等待CUDA设备上所有流的核心计算完成
# 函数接受一个device参数，代表是哪个设备需要同步
run(x_gpu1)
run(x_gpu2) # 预热设备
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])
with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
torch.cuda.synchronize(devices[0])
with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
torch.cuda.synchronize(devices[1])

# 删除两任务间的synchronize，系统在两个设备上自动并行计算
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()


# 并行计算与通信
# 需要在CPU和GPU之间移动数据，或者是GPU之间
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]
with d2l.Benchmark('在GPU1上运行'):
    y = run(x_gpu1)
    torch.cuda.synchronize()
with d2l.Benchmark('复制到CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()

# to()和copy_()等允许显式non_blocking，
# 这允许在不需要同步时调用方可以绕过同步
with d2l.Benchmark('在GPU1上运行并复制到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()


# 多 GPU 训练
# 一般来说，k个GPU并行训练过程如下：
# 1.任意次训练迭代中，给定的随机小批量样本将被分成k个部分，均匀分配到GPU上
# 2.每个GPU根据分配给它的小批量子集，计算模型参数的损失和梯度
# 3.将k个GPU局部梯度聚合，获得小批量随机梯度，聚合梯度被重新分到每个GPU中
# 4.每个GPU使用这个小批量随机梯度，来更新它所维护的完整的模型参数集
# 在k个GPU上训练时，需要扩大小批量大小为k的倍数，这样每个GPU都有相同工作量
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]
# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 数据同步
# GPU训练的两个基本操作：首先，需要向多个设备分发参数并附加梯度
# 第二，需要跨多个设备对参数求和，也就是需要一个 allreduce 函数
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
# print('b1 权重:', new_params[1])
# print('b1 梯度:', new_params[1].grad)

# 现有一个向量分布在多GPU上，allreduce 函数将所有向量相加，并将结果广播给所有GPU
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

# 通过在不同设备上创建具有不同值的向量并聚合它们
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
# print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
# print('allreduce之后：\n', data[0], '\n', data[1])

# 将一个小批量数据均匀地分布在多个GPU上
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices) # 核心分配数据函数
# print('input :', data, '\nload into', devices, '\noutput:', split)

def split_batch(X, y, devices): # @save
    # 将X和y拆分到多个设备上
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))

# 多 GPU 训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [loss(lenet(X_shard, device_W), y_shard).sum() # 每个GPU上分别计算损失
    for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    for l in ls: # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 使用全尺寸的小批量

# 训练函数需要分配GPU并将所有模型参数复制到所有设备
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
        lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'f'在{str(devices)}')

# train(num_gpus=1, batch_size=256, lr=0.2) # 单个GPU上运行效果
# train(num_gpus=2, batch_size=256, lr=0.2) # 多个GPU上运行效果

# 多 GPU 的简洁实现
def resnet18(num_classes, in_channels=1): # @save
    # 稍加修改的ResNet-18模型
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(nn.Conv2d(in_channels, 64,
    kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net

# 初始化网络，获取GPU列表
net = resnet18(10)
devices = d2l.try_all_gpus()

# 训练模型
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'f'在{str(devices)}')

# train(net, num_gpus=1, batch_size=256, lr=0.1) # 单GPU训练
# train(net, num_gpus=2, batch_size=512, lr=0.2) # 多GPU训练