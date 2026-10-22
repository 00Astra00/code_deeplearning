import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

"""
# 定义 softmax 操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制

# 实现 softmax 回归
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 参数 W，b 初始化
# num_inputs, num_outputs = 784, 10 # 输入维度，输出维度
# W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs, requires_grad=True)

# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数(交叉熵)
y = torch.tensor([0, 2]) # 两个样本，三个类别预测
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y]) # 分别为 0.1, 0.5

def cross_entropy(y_hat, y):
    # y_hat[range(len(y_hat)), y] 循环遍历取值
    return - torch.log(y_hat[range(len(y_hat)), y])

# print(len(y_hat)) # 返回第一个维度的长度
# print(cross_entropy(y_hat, y))

# 分类精度
def accuracy(y_hat, y): # @save
    # 计算预测正确的数量，注意是数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # argmax(axis=1)：沿 axis = 1 维度取最大值索引
        y_hat = y_hat.argmax(axis=1)
    # 确保数据类型一致，‘== y’ 逐元素比较预测和真实标签
    cmp = y_hat.type(y.dtype) == y # 返回比较列表
    # 返回最终预测正确的数量
    return float(cmp.type(y.dtype).sum())

# print(accuracy(y_hat, y) / len(y))

# 实用程序类 Accumulator，用于对多个变量进行累加
class Accumulator: # @save
    # 在n个变量上累加
    def __init__(self, n):
        # 初始化包含n个0.0的列表
        self.data = [0.0] * n
    def add(self, *args):
        # 将多个值加到对应的累加器上
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        # 重置所有累加器为0.0，保持相同的长度
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        # 支持索引访问，使类行为类似列表
        return self.data[idx]

# 在动画中绘制参数(选看)
class Animator: # @save
    # 在动画中绘制数据，使训练进度可视化
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
    xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 对于任意数据迭代器 data_iter 可访问的数据集，评估任意模型 net 的精度
def evaluate_accuracy(net, data_iter): # @save
    # 检查 net 是否是PyTorch的Module类型
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad(): # 禁用梯度计算，加速运行
        for X, y in data_iter: # 将结果累加到 metric 中
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# print(evaluate_accuracy(net, test_iter))

# 定义优化函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# softmax回归的训练函数，训练一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater): # @save
    if isinstance(net, torch.nn.Module):
        net.train() # 将模型设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    # 计算梯度并更新参数
    for X, y in train_iter:
        y_hat = net(X) # 前向传播
        l = loss(y_hat, y)
        # 使用 PyTorch 内置的优化器和损失函数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward() # 计算平均损失的梯度
            updater.step()
        else: # 使用定制的优化器和损失函数
            l.sum().backward() # 计算总损失的梯度
            updater(X.shape[0]) # 手动传入batch_size
        # 训练损失总和、训练准确度总和、样本数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回平均训练损失和平均训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练模型，训练 num_epochs 个周期
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): # @save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    # 训练每个epoch，得到训练损失和准确率，在测试集上评估准确率，更新可视化图表
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 这段代码是训练结果的验证和断言检查，用于确保模型训练达到了预期的效果
    # condition, message：如条件为 False，抛出 AssertionError ，验证程序假设条件
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 预测函数
def predict_ch3(net, test_iter, n=6): # @save
    # 从测试数据迭代器中获取一个批次的数据，获取：
    # X: 图像数据形状为 (batch_size, 1, 28, 28)
    # y: 真实标签，形状为 (batch_size,)
    for X, y in test_iter:
        break
    # 将数字标签转换为文本标签
    trues = d2l.get_fashion_mnist_labels(y)
    # 获取预测标签文本，将预测的数字标签转换为文本标签
    predicts = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + predict for true, predict in zip(trues, predicts)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

# 主函数，添加主模块保护
if __name__ == '__main__':
    # 先前的非函数代码移动至后面
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs = 784, 10 # 输入维度，输出维度
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    lr, num_epochs = 0.1, 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    print(evaluate_accuracy(net, test_iter))
    predict_ch3(net, test_iter)
    d2l.plt.show()
"""


""" 
# softmax 回归简洁实现
# 加载数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# PyTorch 不会隐式调整输入形状，因此在线性层前定义展平层，来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear: # 只对线性层进行操作
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights) # 递归地对网络中的所有模块应用初始化函数
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10 # 调用高级 API 训练 softmax 回归函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
"""