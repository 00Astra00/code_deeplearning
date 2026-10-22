import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l


# 泛化性和灵活性之间的这种基本权衡被描述为偏差-方差权衡
# 神经网络并不局限于单独查看每个特征，而是学习特征之间的交互
# 暂退法在前向传播过程中，计算每一内部层的同时注入噪声(仅训练时使用)
# 标准暂退法正则化中，通过按保留的节点的分数进行规范化来消除每一层的偏差，
# 即每个中间活性值 h 以暂退概率 p 由随机变量 h′ 替换

# 定义模型和参数
num_inputs, num_outputs = 784, 10
num_hiddens1, num_hiddens2 = 256, 256
dropout1, dropout2 = 0.2, 0.5

# 从零开始实现暂退法
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # torch.rand(X.shape)：生成与输入X相同形状的均匀分布随机数
    # > dropout：创建布尔掩码，随机数大于dropout保留，否则丢弃
    mask = (torch.rand(X.shape) > dropout).float()
    # 应用掩码，随机将部分元素置零，缩放操作，保持输出期望值不变
    return mask * X / (1.0 - dropout)

# 测试掩码效果
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

# 定义 Net 类
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        # 调用父类nn.Module的初始化方法
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        # 将输入重塑为(batch_size, num_inputs)的形状
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

# 定义模型
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# d2l.plt.show()



# 简洁实现暂退法
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1),
      nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()



# 暂退法 vs 权重衰减
# 特性	 Dropout	    权重衰减
# 机制	 随机丢弃神经元	L2正则化，惩罚大权重
# 计算	 前向传播时应用	反向传播时应用
# 效果	 防止特征共适应	防止权重过大
# 超参数	 dropout概率p	权重衰减系数λ

# 同时使用的效果：
# 不是简单累加，而是互补：
# 协同作用：Dropout防止神经元共适应，权重衰减防止过拟合
# 收益递减：同时使用可能收益减少，但通常效果更好
# 需要调参：需要仔细调整p和λ的平衡