import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn


# 回顾单隐藏层的多层感知机 MLP
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
# print(net(X))

# 参数访问：通过索引来访问模型的任意层
# print(net[2].state_dict()) # 返回一个有序字典，包含该层所有参数
# 分别返回第三层偏置参数的类型，偏置参数的对象，偏置参数的数值
# print(type(net[2].bias), net[2].bias, net[2].bias.data)
# print(net[2].weight.grad) # 访问参数的梯度

# .named_parameters()：获取层的所有参数；print(*)：使用解包操作符打印列表所有元素
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# 定义一个生成块的函数
def block1():
    return nn.Sequential(nn.Linear(4, 8),
    nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

# 循环嵌套块的函数
def block2():
    net = nn.Sequential()
    for i in range(4): # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet(X), "\n", rgnet, "\n", rgnet[0][1][0].bias.data) # 打印相关参数和工作流程

# 内置函数初始化参数模块 nn.init
def init_normal(m):
    if type(m) == nn.Linear: # 注意是 normal_
        # 高斯分布初始化，输入初始化对象，均值，方差
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias) # 注意是 zeros_

net.apply(init_normal) # apply 自动迭代初始化参数
# print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        # constant_ 将所有参数初始化为给定常数
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])

def init_xavier(m):
    # Xavier 方法初始化参数
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier) # 使用 Xavier 初始化第一个神经网络层
net[2].apply(init_42) # 将第三个神经网络层初始化为常量值42
# print(net[0].weight.data[0], net[2].weight.data)

# 自定义初始化函数
def my_init(m):
    if type(m) == nn.Linear: # “*” 使用解包操作符打印列表所有元素
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10) # 使用均匀分布初始化权重(原地操作)
        m.weight.data *= m.weight.data.abs() >= 5 # 只保留绝对值≥5的权重，其他置为0

net.apply(my_init)
# print(net[0].weight[:2])
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
# print(net[0].weight.data[0])

# 多个层共享参数：定义稠密层，使用其参数设置另一层参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
# print(net(X)) # X = torch.rand(size=(2, 4))

# 检查参数是否相同(2,4层共享参数，应该返回 True)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100 # 修改某个位置的值
# 检查参数是否相同，由于参数共享，所以仍返回 True
print(net[2].weight.data[0] == net[4].weight.data[0])

# 延后初始化：直到模型运行时，系统才会动态的初始化参数