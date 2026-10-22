import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
from torch import nn


""" 
# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 带参数的层
class MyLinear(nn.Module):
    # 输入 in_units，units 作为参数
    def __init__(self, in_units, units):
        super().__init__()
        # nn.Parameter(): 将张量包装为Parameter对象，能被自动追踪梯度
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
# print(linear.weight)
"""


""" 
# 读写文件
# 对于 张量 ，调用 load 和 save 函数读写
x, y = torch.arange(4), torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
# print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
# print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# net.state_dict(): 获取模型的所有参数的有序字典
torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
# clone.load_state_dict(): 将加载参数用到克隆网络
clone.load_state_dict(torch.load('mlp.params'))
clone.eval() # 将模型设置为评估模式

Y_clone = clone(X)
# print(Y_clone == Y)
"""



""" 
# GPU
# 可以查询可用gpu的数量
# print(torch.cuda.device_count())

def try_gpu(i=0): # @save
    # 如果存在，则返回gpu(i)，否则返回cpu()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus(): # @save
    # 返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1, 2, 3])
# print(x.device) # 查询张量所在设备
# 对多个项进行操作，它们都必须在同一个设备上

X = torch.ones(2, 3, device=try_gpu())
# Z = X.cuda(1) # 复制 X 到 Z 上进行操作
# print(X, Z)

# 神经网络模型可以指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu()) # 如果可以，指定在 GPU 上运行
# print(net(X))
"""