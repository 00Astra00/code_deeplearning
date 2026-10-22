import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l


# 调度器示例
def net_fn():
    model = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
    return model

loss = nn.CrossEntropyLoss()
batch_size, device = 256, d2l.try_gpu()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 代码几乎与d2l.train_ch6定义在卷积神经网络一章LeNet一节中的相同
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        if scheduler:
             if scheduler.__module__ == lr_scheduler.__name__:
                 # UsingPyTorchIn - Builtscheduler
                 scheduler.step()
             else:
                 # Usingcustomdefinedscheduler
                 for param_group in trainer.param_groups:
                     param_group['lr'] = scheduler(epoch)
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')

lr, num_epochs = 0.3, 30
# net = net_fn()
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device)

# 每个迭代轮数（甚至在每个小批量）之后向下调整学习率
# 定义一个调度器：当调用更新次数时，返回学习率的适当值
# 学习率设置为 η = η_0 * (t + 1) ^ (−1/2)
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr
    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)

# scheduler = SquareRootScheduler(lr=0.1)
# net = net_fn()
# trainer = torch.optim.SGD(net.parameters(), lr)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)


# 普遍方法设计调度器：多项式衰减和分段常数表
# 1.单因子调度器：多项式衰减的一种替代方案是乘法衰减，为了防止学习率衰减到一个合理
# 下界之下，更新方程经常修改为 η_t+1 ← max(η_min, η_t * α)，其中 α -> (0, 1)
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr
    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(torch.arange(50), [scheduler(t) for t in range(50)])

# 2.多因子调度器：给定一组降低学习率的时间点，当 t ∈ s 时，降低 η_t+1 ← η_t * α
# 这种分段恒定学习率调度背后的直觉是，让优化持续进行，直到权重向量的分布达到一个驻点
# 此时才将学习率降低，以获得更高质量的代理来达到一个良好的局部最小值
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# 创建多步学习率调度器；milestones=[15, 30]: 在第15和30个epoch时降低学习率
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0] # 获取当前学习率
    trainer.step()                  # 优化器更新参数
    scheduler.step()                # 调度器更新步数
    return lr

d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler) for t in range(num_epochs)])

# 3.余弦调度器：依据观点：不想在一开始就太大降低学习率，最终能用非常小学习率“改进”解决方案
# 函数形式：η_t = η_T + (η_0 − η_T) / 2 * (1 + cos(π * t / T))，学习率 t ∈ [0, T]
# η_0 是初始学习率，η_T 是 T 时的目标学习率。此外对于t > T，只需将值固定到η_T而不再增加它
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        # max_update：最大更新步数，final_lr：最终学习率，max_steps=max_update-warmup_steps：余弦衰减步数
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        # 线性增加学习率：从 warmup_begin_lr 到 base_lr_orig
        # base_lr_orig - warmup_begin_lr 起始到目标学习率的总差值
        # float(epoch) / warmup_steps 计算当前预热阶段中的进度比例
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        # increase = 总差值 × 进度比例，将增加量加到起始学习率上
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        # 预热阶段学习率计算
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        # 余弦衰减阶段学习率计算
        if epoch <= self.max_update:
            # (epoch - warmup_steps) / max_steps：余弦衰减阶段进度比例
            # (1 + cos(pi * 进度比例)) / 2：将进度从[0, 1]映射到[0, 1]
            # (base_lr_orig - final_lr) * 归一化值：学习率变化的总范围，
            # 按余弦曲线比例缩放；+ self.final_lr：加上最终学习率作为基准
            self.base_lr = (self.final_lr + (self.base_lr_orig - self.final_lr) *
            (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2)
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)

# 某些情况下，初始化参数不足以得到良好解，解决方法是用预热期，
# 在此期间学习率将增加至初始最大值，然后冷却直到优化过程结束
# 预热可应用于任何调度器，预热限制非常深的网络参数的发散程度
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)