import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

"""
# 损失函数：通常为平方损失函数
# 随机小批量梯度下降法适用于所有深度学习模型
# 超参数：可以调整但不在训练过程中更新的参数
# 泛化：找到一组参数能在从未见过的数据上实现较低的损失
# 矢量化：不用依赖循环，加速计算过程
n = 10000 # 数据量
a = torch.ones([n])
b = torch.ones([n])

# 定义一个计时器类 Timer
class Timer: # @save
    # 记录多次运行时间
    def __init__(self):
        # 类中的主函数 self
        self.times = []
        self.start()
    def start(self):
        # 启动计时器
        self.tik = time.time()
    def stop(self):
        # 停止计时器并将时间记录在列表中
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        # 返回平均时间
        return sum(self.times) / len(self.times)
    def sum(self):
        # 返回时间总和
        return sum(self.times)
    def cumsum(self):
        # 返回累计时间
        return np.array(self.times).cumsum().tolist()

# 测试有无矢量化的时间区别
c = torch.zeros([n])
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
# print(f'{timer.stop():.8f}s')
timer.start()
d = a + b
# print(f'{timer.stop():.8f}s')

# 正态分布计算函数：与均值和标准差有关
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
# 正态分布可视化(选看)
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)] # 均值和标准差对
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5), legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
# d2l.plt.show() # 注意这里不是 plot

# 噪声：符合正态分布，为拟合函数添加随机性
# 似然函数(极大似然估计法)：通过给定的x观测到特定y的似然的概率
# 极大似然估计法，参数w和b的最优值是使整个数据集的似然最大的值
# 但优化函数通常求最小值，因此改求最小化负对数似然 − log P(y | X)
# 线性回归模型实质上是单层神经网络(全连接层)
"""


"""
# 实现线性规划
# 生成模拟数据集
def synthetic_data(w, b, num_examples): # @save
    # 生成y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # 执行矩阵乘法 X * w
    y += torch.normal(0, 0.01, y.shape) # 添加噪声
    # 将 y 从形状 (num_examples, ) 转换为 (num_examples, 1)，确保标签是列向量
    # -1 是一个特殊的参数，这里已知总数据量 n ，且只有一列，则自动计算矩阵行的大小
    return X, y.reshape((-1, 1))

true_w , true_b = torch.tensor([2, -3.4]), 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# print(features[0], "\n", labels[0])

d2l.set_figsize() # 线性关系可视化(选看)
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# data_iter 函数，该函数接收批量大小、特征矩阵和标签向量作为输入，
# 生成大小为 batch_size 的小批量(包含一组特征和标签)，实现小批量随机梯度下降
def data_iter(batch_size, features, labels): # @save
    num_examples = len(features)
    # indices: 创建从 0 到 num_examples-1 的索引列表
    indices = list(range(num_examples))
    # shuffle 函数随机读取样本，也就是将索引列表随机打乱
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引切片，min(i + batch_size, num_examples) 确保不超出索引范围
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # yield 使函数成为生成器，每次迭代返回一个批次的数据(0->15,16->31,32->47,......)
        yield features[batch_indices], labels[batch_indices]

batch_size = 16 # 以下读取第一个批量样本
for X, y in data_iter(batch_size, features, labels):
    # print(X, '\n', y)
    break

# 模型参数初始化
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b): # @save
    # 线性回归模型
    return torch.matmul(X, w) + b

# 定义损失函数(均方损失)
def squared_loss(y_hat, y): # @save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 小批量随机梯度下降
def sgd(params, lr, batch_size): # @save
    # 禁用梯度计算：在参数更新时，不需要跟踪这些操作的梯度
    with torch.no_grad():
        for param in params:
            # param.grad: 当前参数的梯度（通过反向传播计算得到）
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练函数
lr, num_epochs = 0.10, 4
net, loss= linreg, squared_loss
for epoch in range(num_epochs):
    # 持续循环，更新参数
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size, 1)，而不是一个标量
        # l中的所有元素被加到一起，并以此计算关于[w, b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
"""


""" 
# 线性回归简洁实现
# 生成模拟数据集
true_w, true_b = torch.tensor([2, -3.4]), 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集，其中 is_train 表示是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True): # @save
    # 构造一个 PyTorch 数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 使用 iter 构造 Python 迭代器，并使用 next 从迭代器中获取第一项
# print(next(iter(data_iter)))

# nn是神经网络的缩写，Sequential 表示实例化线性网络
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# 通过 net[0] 选择第一个图层，使用 weight.data 和 bias.data 访问参数
# 还可以使用替换方法 normal_ 和 fill_ 来重写参数值
# print(net[0].weight.data.normal_(0, 0.01))
# print(net[0].bias.data.fill_(0))

# 定义损失函数(返回均方损失 MSE)
# MSELoss类，也称平方L2范数，返回所有样本损失的平均值
loss = nn.MSELoss()

# 定义优化算法
# 实例化一个 SGD 时，要指定优化的参数（通过 net.parameters() 获得）
# 还要获得优化算法所需的超参数字典，小批量随机梯度下降只需要设置 lr 值
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # 获得 trainer 参数优化对象

# 训练模型
num_epochs = 4
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        # 根据梯度更新参数
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 误差估计
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


# 独热编码：一个向量，分量和类别一样多，类别对应的分量设置为1，否则为0
# softmax：将输出映射为 0 ~ 1 之间的概率数(非负、和为1)
# 交叉熵损失(分类问题适用)：最小化负对数似然 - sigma(y * log(y_hat))
# 交叉熵可想象为 “主观概率为Q的观察者在看到根据概率P生成的数据时的预期惊异”
"""