import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
import pandas as pd
#%matplotlib inline
import numpy as np
from torch.distributions import multinomial
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from d2l import torch as d2l

""" 
# 1. torch 基础操作
X = torch.tensor([2,3,4])
x = torch.arange(12)
# 打印 形状 和 总大小(size)
print(x.shape, " ", x.numel())
x_ = x.reshape(3,4)
print(x.sum())

a = torch.ones((3,4))
b = torch.zeros((3,4))

# 基于(0,1)正态分布随机赋值
n = torch.randn(3,4)

x_1 = torch.tensor([2,5,6,8])
x_2 = torch.tensor([4,6,8,2])
print(x_1 * x_2, x_1 ** x_2)

print(torch.exp(x)) # 取e

# cat 连接矩阵，dim=0横向连接，dim=1纵向连接
x_3 = torch.cat((x_1, x_2), dim=0)

# 广播机制(将形式不符的矩阵自动匹配)
# 注意广播机制的前提是只有一个维度的形状不一样
aa = torch.arange(12).reshape((2, 2, 3))
bb = torch.arange(6).reshape((1, 2, 3))
print(aa + bb)

print(x[-1]) # -1 表示取最后元素
print(x[2:6]) # 取序列元素，左闭右开
x_[1:3,0:2] = 13 # 区块赋值

# 原地操作节省内存
B = torch.zeros_like(b)
print(id(B))
# 只有标明 B[:, :]格式才能原地操作
B[:,:] = B + x_
# 打印两次地址比较
print(id(B))

# item 函数将张量转化为标量
s = torch.tensor([7.8])
print(s, " ", s.item())
"""


"""
# 2. pandas 基础
data = pd.read_csv(file_name)

# 处理缺失值 : 插值法
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())

# 数值类型转化为张量类型
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
"""


""" 
# 3. 线性代数基础
a = torch.arange(24, dtype=torch.float32)
# print(a[3], " ", len(a))
# shape 返回每个维度长度的元素组
a_ = a.reshape(2,3,4)
print(a_.shape) # 2,3,4

A = torch.arange(20).reshape(5, 4)
# print(A.T) # 矩阵转置

# 通过分配新内存，将A的一个副本分配给B
B = A.clone()

# 两个矩阵的按元素对应位置乘法为Hadamard积(⊙)
print(A * B) # 验证Hadamard积，必须形状相同，返回矩阵

# 设置矩阵元素为浮点数
x = torch.arange(4, dtype=torch.float32)
# print(x,x.sum())

# 指定 axis=0，压缩成行；axis=1，压缩成列
A_sum = A.sum(axis = 0) # A.sum([0, 1])等价于A.sum()
print(A_sum, A_sum.shape)

# mean 方法求平均值，等价于 A.sum()/A.numel()
print(x.mean(), " ", x.sum() / x.numel())

# 求和时用 keepdims=True 保持维度不变
sum_A = A.sum(axis=1, keepdims=True)

# 如果我们想沿某个轴计算A元素的累加和，如axis=0（按行算），
# 可以调用cumsum函数，此函数不会沿任何轴降低输入张量的维度。
print(A.cumsum(axis=0))

# 点积 dot =  x.T * y 是相同位置的按元素乘积的和，返回值
b = torch.ones(24, dtype=torch.float32)
print(torch.dot(a,b), torch.sum(a * b))

# 矩阵-向量乘积 mv (m的列数 == v的长度)
m = torch.arange(20).reshape(5,4)
v = torch.tensor([1, 9, 7, 4])
print(m.shape, v.shape, torch.mv(m, v))

# 矩阵-矩阵乘法 mm
m_1 = torch.arange(24).reshape(4,6)
m_2 = torch.arange(18).reshape(6,3)
print(torch.mm(m_1, m_2))

# 范数，向量中指向量的长度；在线性代数中，向量范数是将向量映射到标量的函数f
# 符合：1.按常数因子α缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放
# 2.符合三角不等式  3.非负性 -> 下面介绍 L1，L2 和 LP范数
f = torch.arange(12, dtype=torch.float32)
# L1 范数：向量中元素的绝对值之和
f_1 = torch.abs(f).sum()
# L2 范数：向量元素平方和的平方根(欧几里何距离)
# norm 函数要求向量元素必须为浮点数
f_2 = torch.norm(f)
# 补充F范数：Frobenius范数是矩阵元素平方和的平方根
"""


""" 
# 4.微积分基础(链式求导，微分，偏导数)
# 注意：@save 表示将编写的程序保存到d2l库中

# 利用 matplotlib 编写绘图函数 
def use_svg_display(): # @save
    # 使用svg格式在Jupyter中显示绘图
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)): # @save
    # 设置matplotlib的图表大小
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): # @save
    # 设置matplotlib的轴函数
    # axes 在 Matplotlib 中指的是坐标轴对象
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid() # 显示图像

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
xscale='linear', yscale='linear',fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None): # @save
    # 绘制数据点
    if legend is None:
        legend = []

    set_figsize(figsize)
    # axes 在 Matplotlib 中指的是坐标轴对象
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return(hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
              and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla() # 清除坐标轴

    # zip 将多个序列中相同位置的元素组合成元组，返回一个迭代器。
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 绘制函数 u = f(x)及其在 x = 1处的切线 y = 2x − 3
def f(x):
    return x ** 3 - 1.0 / x

x = np.arange(0, 3, 0.1)
plot(x, [f(x), 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show()
"""


""" 
# 4.自动微分(续)
# require_grad = True 相当于分配内存储存梯度
y = torch.arange(4.0, requires_grad=True)
print(y.grad) # 默认值是 None
z = 2 * torch.dot(y, y)
print(z.backward(),y.grad)
print(y.grad == 4 * y) # 验证求导

y.grad.zero_() # 清空储存的梯度
z = y.sum().backward() # 反向梯度求导
print(y.grad)

# 对非标量调用backward需要传入一个gradient参数
y.grad.zero_()
z = y * y
z.sum().backward()
print(y.grad)

# detach 函数设置参数为常数
y.grad.zero_()
z = y * y
u = z.detach() # 将 z 设为常数
f = u * y # 求导函数
f.sum().backward()
print(y.grad == u)

# 思考题
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a) # 一定有 K 使得 d = k * a
d.backward() # 求导结果一定是 K
print(a.grad == d / a) # 即 k * a / a = k
"""


""" 
# 5.概率论基础
# 例子：投骰子问题，利用 mM 函数根据概率矩阵 fair_pro 随机抽取 total_count 个样本
fair_pro = torch.ones([6]) / 6

# counts 矩阵保存每次情况的样本数，sample((num,)) 表示进行 num 次操作
counts = multinomial.Multinomial(10, fair_pro).sample((50,))
# print(counts / 50) # counts / 50 该矩阵作为相对频率的估计值

cum_counts = counts.cumsum(dim=0)
# print(cum_counts) # 显示每个面的累加次数，dim = 0 表示在每行(0轴)累加

estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates) # cum_counts.sum 对每行求和，留下列；计算概率，反应动态变化过程

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed') # 绘制 y = 0.167 水平基准线
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()

# 联合概率：给定 P(A=a),P(B=b)，表示 P(A=a,B=b) 同时发生的概率
# 条件概率：P(B=b | A=a) 表示 A 发生的前提下 B 发生的概率
# 贝叶斯定理：P(A | B) = ( P(B | A) * P(A) ) / P(B)
# 独立性：A 与 B 无关，即 P(A | B) = P(A)
# 期望：sigma(x * P(x))，sigma 为求和函数
# 方差：衡量随机变量X与其期望值的偏置，取平方根为标准差
# V[x] = E( (x - E(x))^2 ) = E(x^2) - (E(x))^2
"""