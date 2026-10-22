import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from torch.nn import functional as F


# 块可以描述单个层、由多个层组成的组件或整个模型本身
# nn.Sequential定义了一种特殊的Module，即在PyTorch中表示一个块的类
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# X = torch.rand(2, 20)
# net(X) # 回顾 MLP

# 每个块必须提供的基本功能：
# 1. 将输入数据作为其前向传播函数的参数
# 2. 通过前向传播函数生成输出，注意输出形状可能与输入形状不同
# 3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问
# 4. 存储和访问前向传播计算所需的参数
# 5. 根据需要初始化模型参数


# 实例化 MLP 的块(参照回顾代码)
# 定义名为 MLP 的类，继承自 nn.Module
class MLP(nn.Module):
    # 定义类的初始化方法（构造函数）
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)

    # 定义模型前向传播，即根据输入X返回模型输出
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

# X = torch.rand(2, 20)
# net = MLP()
# print(net(X))


# 实例化自建块(通用性更好)
# 1. 将块逐个追加到列表中的函数
# 2. 前向传播函数，将输入按追加块顺序传给块组成的“链条”
class MySequential(nn.Module):
    # 使用 *args 可变参数，接受任意数量模块作为参数
    def __init__(self, *args):
        super().__init__()
        # enumerate 同时获取索引和模块对象
        for idx, module in enumerate(args):
            # 这里，module 是 Module 子类的一个实例
            # _modules 中 _module 类型是 OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict 保证按照成员添加顺序遍历
        for block in self._modules.values():
            X = block(X)
        return X

# X = torch.rand(2, 20)
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))


# 并不是所有的架构都是简单的顺序架构
# 有时合并既不是上一层的结果也不是可更新参数的项，即常数参数
# 这是一个具有固定权重和控制流的特殊MLP实现
class FixedHiddenMLP(nn.Module):
    def __init__(self):
       super().__init__()
       # 不计算梯度的随机权重参数，因此其在训练期间保持不变
       self.rand_weight = torch.rand((20, 20), requires_grad=False)
       self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流：一个动态的条件循环
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# X = torch.rand(2, 20)
# net = FixedHiddenMLP()
# print(net(X))


# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64),
           nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

# X = torch.rand(2, 20)
# chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
# print(chimera(X))


# 创建参数 net1，net2 的平行块(习题)
class ParallelBlock(nn.Module):
    # 在指定维度 dim=1 上进行合并
    def __init__(self, net1, net2, dim=1):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.dim = dim

    def forward(self, X):
        out1 = self.net1(X)
        out2 = self.net2(X)
        return torch.cat([out1, out2], dim=self.dim)

X = torch.rand(2, 20)
net1 = nn.Sequential(nn.Linear(20,20), nn.ReLU())
net2 = nn.Sequential(nn.Dropout(), nn.Linear(20,20))
Parallel_Net = ParallelBlock(net1, net2)
print(Parallel_Net(X))