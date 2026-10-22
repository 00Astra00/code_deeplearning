import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
from d2l import torch as d2l


""" 
# 动量法：泄漏平均值取代梯度计算：v_t = β * v_t−1 + g_(t,t−1)
# 其中 β ∈ (0, 1)，这有效将瞬时梯度替换为多个“过去”梯度的平均值
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# eta = 0.4 / 0.6
# d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
# 陷入两难：小学习率会确保解不会在x_2方向发散，但要承受在x_1方向缓慢收敛
# 相反，如果学习率较高，可以确保解在x_1方向上进展很快，但在x_2方向会发散
# 动量法能够解决上面的梯度下降问题，使用v_t而不是梯度g_t可生成以下更新等式：
# 1. v_t ← β * v_t−1 + g_(t,t−1)   2. x_t ← x_t−1 − (lr)_t * v_t
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
# d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

# 从零开始实现动量法
def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
    {'lr': lr, 'momentum': momentum}, data_iter, feature_dim, num_epochs)
    # d2l.plt.show()

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# train_momentum(0.02, 0.5)

# 简洁实现
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
"""



""" 
# AdaGrad 优化法
# 将 s(i, t) 计作截至 t 时观察到功能 i 的次数
# AdaGrad算法使用 s(i, t + 1) = s(i, t) + (∂_i * f(x))^2 调整学习率
# 两个好处：首先，不需要决定梯度何时算足够大；其次，它会随梯度的大小自动变化
# 使用变量 st 来累加过去的梯度方差，如下所示：
# g_t = ∂_w * l(y_t, f(x_t, w))
# s_t = s_t−1 + (g_t)^2
# w_t = w_t−1 − ((eta) / sqrt(s_t + ϵ)) * g_t
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
# d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

# 从零开始实现 AdaGrad
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)
def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# d2l.train_ch11(adagrad, init_adagrad_states(feature_dim), {'lr': 0.1}, data_iter, feature_dim)

# 简洁实现
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
# d2l.plt.show()
"""



""" 
# RMSProp 优化法
# RMSProp算法作为将速率调度与坐标自适应学习率分离的简单修复方法
# 参数更新方式改变： s_t ← γ * s_t−1 + (1 − γ) * (g_t)^2
# 保持所有其它部分不变就产生RMSProp算法(γ > 0)

# 从零开始实现 RMSProp
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
# d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))

def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim), {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim)

# 简洁实现
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)
# d2l.plt.show()
"""



""" 
# Adadelta 优化法
# Adadelta使用两个状态变量，s_t 存储梯度二阶导数泄露平均值，∆x_t 存储模型本身中参数变化二阶导数泄露平均值
# 即：s_t = ρ * s_t−1 + (1 − ρ) * (g_t)^2，使用重新缩放的梯度 g'_t 执行更新，即 x_t = x_t−1 − g'_t
# 调整后的梯度 g'_t 按如下方式计算：g'_t = sqrt(∆x_t−1 + ϵ) / sqrt(s_t + ϵ) ⊙ g_t
# 其中 ∆x_t−1 是重新缩放梯度的平方 g_t' 的泄漏平均值,将 ∆x_0 初始化为0，然后在每个步骤中使用 g_t' 更新它，
# 即：∆x_t = ρ * ∆x_t−1 + (1 − ρ) * (g_t')^2
def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))
def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-placeupdatesvia[:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# d2l.train_ch11(adadelta, init_adadelta_states(feature_dim), {'rho': 0.9}, data_iter, feature_dim)

# 简洁实现
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
# d2l.plt.show()
"""




# Adam 算法将所有这些技术汇总到一个高效的学习算法中
# Adam算法的关键组成部分：它使用指数加权移动平均值来估算梯度的动量和二次矩，即它使用状态变量
# v_t ← β_1 * v_t−1 + (1 − β_1) * g_t，s_t ← β_2 * s_t−1 + (1 − β_2) * (g_t)^2
# 经常设置为 β_1 = 0.9 和 β_2 = 0.999。也就是说，方差估计的移动远远慢于动量估计的移动
# 准化状态变量由下式获得：vˆt = v_t / (1 − β_1t) and sˆt = s_t / (1 − β_2t)
# 使用类似于RMSProp算法的方式重新缩放梯度以获得：g_t' = (lr) * vˆt / (sqrt(sˆt) + ϵ)
# 与RMSProp不同，更新使用动量 vˆt 而不是梯度，通常 ϵ = 1e−6，最后：x_t ← x_t−1 − g_t'
def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))
def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# d2l.train_ch11(adam, init_adam_states(feature_dim), {'lr': 0.01, 't': 1}, data_iter, feature_dim)

# 简洁实现
trainer = torch.optim.Adam
# d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
# d2l.plt.show()

# Adam 基础上的 Yogi 优化
# Adam算法存在问题：在凸环境下，当st二次矩估计值爆炸时，可能无法收敛
# Adam算法更新如下：s_t ← s_t−1 + (1 − β_2) * ((g_t)^2 - s_t-1)
# 当 (g_t)^2 具有值很大的变量或更新很稀疏时，s_t 可能会太快“忘记”过去值
# 有效解决方法是 (g_t)^2 −s_t−1 -> (g_t)^2 ⊙ sgn((g_t)^2 − s_t−1)
# 更新式：s_t ← s_t−1 + (1 − β_2) * (g_t)^2 ⊙ sgn((g_t)^2 − s_t−1)
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            # Yogi优化创新点：这种自适应机制梯度较小时更新更保守
            # 当 g_t² > s_{t-1} 时，sign=1，更新方式类似Adam
            # 当 g_t² < s_{t-1} 时，sign=-1，进行衰减更新
            s[:] = s + (1 - beta2) * torch.sign(torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim), {'lr': 0.01, 't': 1}, data_iter, feature_dim)
# d2l.plt.show()
