import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from mpl_toolkits import mplot3d
from d2l import torch as d2l


""" 
# 经验风险是训练数据集的平均损失，而风险则是整个数据群的预期损失
def f(x): # 风险函数f
    return x * torch.cos(np.pi * x)
def g(x): # 经验风险函数g
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)
def annotate(text, xy, xytext): # @save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->'))

# 训练数据集的最低经验风险可能与最低风险不同
x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
# d2l.plt.show()

# 局部最小值：在x处对应的f(x)值小于在x附近任意其他点的f(x)值，那么f(x)可能是局部最小值
x = torch.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
# d2l.plt.show()

# 鞍点：梯度消失的一个原因，指函数的梯度都消失但既不是全局最小值也不是局部最小值的任何位置
x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
# d2l.plt.show()
# 假设函数Hessian矩阵将有k个特征值，可能是局部最小值、局部最大值或函数梯度为零位置处的鞍点
# 当函数在零梯度位置处的Hessian矩阵的特征值全部为正值时，我们有该函数的局部最小值
# 当函数在零梯度位置处的Hessian矩阵的特征值全部为负值时，我们有该函数的局部最大值
# 当函数在零梯度位置处的Hessian矩阵的特征值为负值和正值时，我们有该函数的一个鞍点

# 梯度消失：
x = torch.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
# d2l.plt.show()
"""



""" 
# 凸集和凸函数
# 对于任何a, b ∈ X，连接a和b的线段也位于X中，则向量空间中的一个集合X是凸的
# 意味着对所有λ ∈ [0, 1]，我们得到 λ * a + (1 − λ) * b ∈ X 当 a, b ∈ X
# 给定凸集Xi，它们的交集 ∩_i X_i 是凸的，但凸集的并集不一定是凸的，即非凸的
f = lambda x: 0.5 * x**2 # 凸函数
g = lambda x: torch.cos(np.pi * x) # 非凸函数
h = lambda x: torch.exp(0.5 * x) # 凸函数
x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
d2l.use_svg_display()
# _, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
# for ax, func in zip(axes, [f, g, h]):
    # d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
# d2l.plt.show()

# 詹森不等式：凸性定义的一种推广
# E_X * [f(X)] ≥ f (E_X[X])，即凸函数的期望不小于期望的凸函数

# 1. 相关性质：凸函数的局部极小值也是全局极小值
# 凸函数f(x) = (x − 1)^2，有局部最小值x = 1，也是全局最小值
# 2. 给定一个定义在凸集X上的凸函数f，其任意一个下水平集是凸的
# 3. f为凸函数，当且仅当任意二次可微一维函数f : R_n → R是凸的
# 任意二次可微多维函数 f，它是凸的当且仅它的 Hessian∇^2f ⪰ 0

# 凸优化的一个很好的特性是能够有效地处理约束
# minimize(x) f(x)，subject to c_i(x) ≤ 0 ，f是目标函数，c_i是约束函数
# 一种近似满足约束优化问题的方法是采用拉格朗日函数L。除满足 c_i(x) ≤ 0 之外，
# 只需将 α_i * c_i(x) 添加到目标函数 f(x)，可以确保不会严重违反约束(惩罚)
# 满足约束条件的另一种策略是投影，更泛化地说，在凸集 X 上的投影被定义为下列式子
# Proj_X'(x) = argmin(x' ∈ X') ∥x − x′∥，它是凸集 X' 中离 X 最近的点
"""



""" 
# 梯度下降
def f(x): # 目标函数
    return x * torch.cos(c * x)
def f_grad(x): # 目标函数的梯度
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f(x): # 目标函数
    return x ** 2
def f_grad(x): # 目标函数的梯度(导数)
    return 2 * x
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    # print(f'epoch 10, x: {x:f}')
    return results

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
    f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

c= torch.tensor(0.15 * np.pi)
# show_trace(gd(2, f_grad), f)

results = gd(0.2, f_grad)
# show_trace(results, f)

# show_trace(gd(0.05, f_grad), f) # lr过小
# show_trace(gd(1.1, f_grad), f) # lr过大

# 多元梯度下降
# 考虑一下 x = [x1, x2,..., xd]_T的情况，即目标函数 f 将向量映射成标量
# 梯度是由d个偏导数组成的向量：偏导数 ∂f(x)/∂x_i 代表输入x_i时f在x处的变化率
def train_2d(trainer, steps=20, f_grad=None): # @save
    # 用定制的训练机优化2D目标函数
    # s1和s2是稍后将使用的内部状态变量
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results): # @save
    # 显示优化过程中2D变量的轨迹
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
             torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def f_2d(x1, x2): # 目标函数
    return x1 ** 2 + 2 * x2 ** 2
def f_2d_grad(x1, x2): # 目标函数的梯度
    return (2 * x1, 4 * x2)
def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1 # eta == lr
# show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
"""



""" 
# 自适应法(牛顿法：函数f的泰勒展开式：)
# f(x+ϵ)=f(x)+ϵ⊤∇f(x)+1/2ϵ⊤∇2f(x)ϵ+O(∥ϵ∥^3)
# 避免繁琐，H(def) = ∇2f(x)定义为f的Hessian
# ∇f(x) + Hϵ = 0 and hence ϵ = −H^−1∇f(x)
# 作为优化问题的一部分，需要将Hessian矩阵H求逆
def f(x): # O目标函数
    return torch.cosh(c * x)
def f_grad(x): # 目标函数的梯度
    return c * torch.sinh(c * x)
def f_hess(x): # 目标函数的Hessian
    return c**2 * torch.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
    f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

c = torch.tensor(0.5)
show_trace(newton(), f)
"""