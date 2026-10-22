import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l


def f(x1, x2): # 目标函数
    return x1 ** 2 + 2 * x2 ** 2
def f_grad(x1, x2): # 目标函数的梯度
    return 2 * x1, 4 * x2
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
def constant_lr(): # lr 变化率函数
    return 1

# eta, lr = 0.1, constant_lr # 常数学习速度
# d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))

# 随机梯度下降中变量的轨迹比梯度下降中轨迹嘈杂得多
# 唯一选择：改变学习率 lr，在优化过程中动态降低学习率
# η(t) = η_i (t_i ≤ t ≤ t_i+1) 分段常数
# η(t) = η0 * e^(−λt) 指数衰减
# η(t) = η0 * (βt + 1)^(−α) 多项式衰减
def exponential_lr(): # 指数衰减
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return math.exp(-0.1 * t)

eta, t, lr = 0.1, 1, exponential_lr
# d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))

def polynomial_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t, lr = 1, polynomial_lr
# d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
# d2l.plt.show()

# 小批量随机梯度下降
# 用小批量的决策的核心是计算效率
timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
# 逐元素计算 A = BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
# print(timer.stop()) # 0.8574166297912598
# 逐列计算 A = BC
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
# print(timer.stop()) # 0.014349937438964844
# 一次性计算 A = BC
timer.start()
A = torch.mm(B, C)
# print(timer.stop()) # 0.000997304916381836

# 小批量计算
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
# print(timer.stop()) # 0.001001596450805664

# 从零开始生成小批量
# 读取数据集(已预处理)：移除均值并将方差重新缩放到每个坐标为1
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
'76e5be1548fd8222e5074cf0faae75edff8cf93f') # @save

def get_data_ch11(batch_size=10, n=1500): # @save
    # np.genfromtxt(): 从文本文件加载数据到numpy数组；delimiter='\t': 指定制表符分隔
    data = np.genfromtxt(d2l.download('airfoil'), dtype=np.float32, delimiter='\t')
    # (data - data.mean(axis=0)) / data.std(axis=0): Z-score标准化
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    # data[:n, :-1]: 前n行的所有列除了最后一列（特征）；data[:n, -1]: 前n行的最后一列（目标值）
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1 # data.shape[1] - 1: 特征数量

def sgd(params, states, hyperparams):
    # states: 优化器状态；hyperparams: 超参数字典
    for p in params:
        # p.data.sub_(): 从参数值中减去学习率*梯度
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2): # @save
    # trainer_fn: 优化器函数（如SGD、Adam等）；feature_dim: 特征维度
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 训练模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer() # n：样本计数器
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams) # 调用优化器函数更新参数
            n += X.shape[0] # 累计已处理的样本数量
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),(d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch') # 输出最终损失和每轮平均训练时间
    return timer.cumsum(), animator.Y[0] # 返回累计时间列表和损失历史

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

# 梯度下降（GD）实验：loss: 0.243, 0.055 sec/epoch
# gd_res = train_sgd(1, 1500, 10)
# 随机梯度下降（SGD）实验：loss: 0.246, 0.102 sec/epoch
# sgd_res = train_sgd(0.005, 1)
# 小批量梯度下降实验：
# mini1_res = train_sgd(.4, 100) # loss: 0.243, 0.003 sec/epoch
# mini2_res = train_sgd(.05, 10) # loss: 0.243, 0.013 sec/epoch

# d2l.set_figsize([6, 3])
# d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),'time (sec)',
#     'loss', xlim=[1e-2, 10], legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
# d2l.plt.gca().set_xscale('log')
# d2l.plt.show()

# 简洁实现随机小批量梯度下降
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4): # @save
    # 初始化模型
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss计算平方误差时不带系数1/2
                animator.add(n/X.shape[0]/len(data_iter),(d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
d2l.plt.show()