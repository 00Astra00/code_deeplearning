import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 循环神经网络是具有隐状态的神经网络
# 当前时间步隐藏变量计算：H_t = ϕ(X_t * W_xh + H_t−1 * W_hh + b_h)
# 从 H_t 和 H_t−1 关系可知，变量捕获并保留序列直到其当前时间步的历史信息，
# 这样的隐藏变量被称为隐状态。输出层的输出计算：O_t = H_t * W_hq + b_q
# 困惑度：度量语言模型的质量，即下一个词元实际选择数的调和平均数(最低为1)
# 通过序列中所有词元交叉熵损失平均值衡量性能，困惑度是交叉熵损失平均值的指数

""" 
# 从零开始实现 RNN
# 读取时光机器数据集
# batch_size, num_steps = 32, 35
# train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 独热编码：将每个索引映射为相互不同的单位向量(示例)
# print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
# X = torch.arange(10).reshape((2, 5))
# print(F.one_hot(X.T, 28).shape) # torch.Size([5, 2, 28])

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape): # 正态分布初始化参数
        return torch.randn(size=shape, device=device) * 0.01
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens)) # W_xh：输入到隐藏层权重
    W_hh = normal((num_hiddens, num_hiddens)) # W_hh：隐藏层到隐藏层权重
    b_h = torch.zeros(num_hiddens, device=device) # b_h：隐藏层偏置
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs)) # W_hq：隐藏层到输出权重
    b_q = torch.zeros(num_outputs, device=device) # b_q：输出层偏置
    # 附加梯度，用于反向传播
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# init_rnn_state函数返回初始隐藏状态(全零)
def init_rnn_state(batch_size, num_hiddens, device):
    # 最后的逗号是为了创建单元素元组(必需)，统一返回元组类型(统一接口)
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# rnn 函数定义了如何在一个时间步内计算隐状态和输出
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 本质上 state = (hidden_state,) # 单元素元组
    H, = state # 元组解包得到 hidden_state
    outputs = []
    for X in inputs: # X的形状：(批量大小，词表大小)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 返回所有时间步的输出拼接和新的隐藏状态
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: # @save
    # 从零开始实现的循环神经网络模型
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 将输入X转换为独热编码，转置维度以适应RNN输入格式
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params) # 调用前向传播函数

    def begin_state(self, batch_size, device): # 创建初始隐藏状态
        return self.init_state(batch_size, self.num_hiddens, device)

# 在循环遍历prefix中的字符时，不断将隐状态传递到下一时间步，但不生成输出
# 这被称为预热期。在预热期结束后，隐状态的值通常比刚开始的初始值更适合预测
# prefix：起始字符串；num_preds：要预测的字符数量；net：训练好的RNN模型
def predict_ch8(prefix, num_preds, net, vocab, device): # @save
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]] # 将前缀的第一个字符转换为索引
    # lambda (参数): 表达式 == def 函数名(参数): return 表达式
    # def get_input(): # 等价于......
    #     return torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 每次调用返回最新的输出作为输入；outputs[-1]：最后预测字符索引；确保形状为(批量大小=1, 序列长度=1)
    for y in prefix[1:]: # 预热期
        # 用当前隐藏状态预测下一个字符，但忽略预测结果
        _, state = net(get_input(), state)
        outputs.append(vocab[y]) # 将真实的下一个字符添加到outputs
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state)
        # y.argmax(dim=1): 在词汇维度取最大值索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 梯度裁剪
# net：神经网络模型；theta：梯度裁剪的阈值
def grad_clipping(net, theta): # @save
    if isinstance(net, nn.Module):
        # 使用 net.parameters() 获取所有参数，过滤只需要梯度的参数
        params = [p for p in net.parameters() if p.requires_grad]
    else: # 自定义模型通常将参数存储在net.params列表中
        params = net.params
    # 计算梯度向量L2范数（所有参数梯度的平方和的平方根），即√(∑ᵢ∑ⱼ(∂L/∂wᵢⱼ)²)
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta: # 如果梯度范数超过阈值，按比例缩小所有梯度
        # 如 ‖g‖ > θ，则：g ← g × (θ / ‖g‖)，即 ‖g_new‖ = θ
        for param in params:
            param.grad[:] *= theta / norm

# 训练网络一个迭代周期
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter): # @save
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else: # 分离状态，避免梯度传播到整个序列
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_() # state对于nn.GRU是个张量
            else: # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                     for s in state:
                         s.detach_()
        y = Y.T.reshape(-1) # 将标签转置并展平(批量大小×序列长度)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean() # 计算损失并取平均
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel()) # 累加总损失和总词元数
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop() # 返回困惑度和训练速度

# 训练模型
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False): # @save
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 根据情况初始化优化函数
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else: # 自定义优化函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device) # 文本生成函数
    # 训练和预测
    for epoch in range(num_epochs):
        # 计算困惑度和训练速度
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

if __name__=='__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    X, num_hiddens = torch.arange(10).reshape((2, 5)), 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())  # 根据批量大小创建零状态
    Y, new_state = net(X.to(d2l.try_gpu()), state)  # 输入数据，得到输出和新状态

    # print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

    num_epochs, lr = 500, 1
    # train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu()) # 不使用随机采样，下面相反
    # train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)
"""



""" 
# 简洁实现 RNN
class RNNModel(nn.Module): # @save
    # 循环神经网络模型
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # self.num_directions：处理单向 / 双向RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1 # 处理单向RNN
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2 # 处理双向RNN
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # inputs形状: (batch_size, num_steps)
        # 独热编码后: (num_steps, batch_size, vocab_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # Y形状: (num_steps, batch_size, num_hiddens * num_directions)
        Y, state = self.rnn(X, state)
        # 重塑: (num_steps * batch_size, num_hiddens * num_directions)
        # 输出: (num_steps * batch_size, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM): # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else: # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

if __name__=='__main__':
    batch_size, num_steps, num_hiddens = 32, 35, 512
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    state = torch.zeros((1, batch_size, num_hiddens))

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    # Y形状: (num_steps, batch_size, num_hiddens)
    # state_new形状: (num_layers, batch_size, num_hiddens)
    Y, state_new = rnn_layer(X, state)
    # print(Y.shape, state_new.shape)

    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    num_epochs, lr = 100, 1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()
"""