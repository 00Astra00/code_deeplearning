import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import re
import torch
import random
from torch import nn
import collections
from d2l import torch as d2l


""" 
# RNN:循环神经网络，用于处理序列信息；处理序列信息的方法如下：
# 一，序列 x_t−1,..., x_1是不必要的，只需满足长度τ时间跨度，即观测序列x_t−1,..., x_t−τ
# 当下获得的最直接的好处就是参数的数量总是不变的，至少在t > τ时如此，这种模型被称为自回归模型
# 二，保留对过去观测总结h_t，同时更新xˆt和h_t。xˆt = P(x_t | h_t)，h_t = g(h_t−1, x_t−1)
# 由于h_t从未被观测到，这类模型也被称为隐变量自回归模型

# 使用正弦函数和一些可加性噪声来生成序列数据
T = 1000 # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) # sin() + 高斯噪声
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show()

# 将这个序列转换为模型的 特征－标签 对
tau = 4 # 使用前4个时间点预测第5个时间点
# 创建特征矩阵：每行包含连续的4个时间点
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 提供 T - tau 个样本的 tau 个原始数据
    features[:, i] = x[i: T - tau + i]
# 标签是每个窗口的下一个时间点
labels = x[tau:].reshape((-1, 1))

# 简化示例：T=10, tau=4
# x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]  # 10个点
# T, tau = 10, 4
# features = torch.zeros((6, 4))  # 10-4=6个样本
# 循环结束后features矩阵为：
# 第0列: [x0, x1, x2, x3, x4, x5]  ← x[0:6]
# 第1列: [x1, x2, x3, x4, x5, x6]  ← x[1:7]
# 第2列: [x2, x3, x4, x5, x6, x7]  ← x[2:8]
# 第3列: [x3, x4, x5, x6, x7, x8]  ← x[3:9]
# 最终的features矩阵：
# [[x0, x1, x2, x3],  # 样本1：用x0-x3预测x4
#  [x1, x2, x3, x4],  # 样本2：用x1-x4预测x5
#  [x2, x3, x4, x5],  # 样本3：用x2-x5预测x6
#  [x3, x4, x5, x6],  # 样本4：用x3-x6预测x7
#  [x4, x5, x6, x7],  # 样本5：用x4-x7预测x8
#  [x5, x6, x7, x8]]  # 样本6：用x5-x8预测x9
# 对应的labels：[x4, x5, x6, x7, x8, x9]

batch_size, n_train = 16, 600 # 只有前 n_train 个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 构建基于序列的模型(MLP)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
          nn.ReLU(),nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失(注意：MSELoss计算平方误差时不带系数1/2)
loss = nn.MSELoss(reduction='none')

# 训练序列模型
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, 'f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测序列模型
# 单步预测(预测下一步)
onestep_preds = net(features)
d2l.plot([time, time[tau:]],[x.detach().numpy(), onestep_preds.detach().numpy()],
'time','x', legend=['data', '1-step preds'], xlim=[1, 1000],figsize=(6, 3))
# d2l.plt.show()

# K步预测
# 长1000的全零张量，存储多步预测的结果
multistep_preds = torch.zeros(T)
# 将前604个位置（600+4）用原始数据x的真实值填充
# 这是因为我们需要至少4个已知点才能开始预测第5个点
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    # multistep_preds[i - tau:i]：取当前点前4个位置的预测值
    # .reshape((1, -1))：将形状从(4,)改为(1, 4)，符合模型输入要求
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

# .detach()：断开梯度追踪；.numpy()：转换为NumPy数组
# 由于错误的累积，经过几个预测步骤之后，预测的结果很快就会衰减到一个常数
d2l.plot([time, time[tau:], time[n_train + tau:]],[x.detach().numpy(), onestep_preds.detach().numpy(),multistep_preds[n_train + tau:].detach().numpy()],
         'time','x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show()

# 以下代码用于显示K步预测的困难(选看)
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],[features[:, (tau + i - 1)].detach().numpy() for i in steps],
         'time', 'x',legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],figsize=(6, 3))
# d2l.plt.show()
"""



""" 
# 文本预处理
# 1. 将文本作为字符串加载到内存中；2. 将字符串拆分为词元（如单词和字符）
# 3. 建立一个词表，将拆分的词元映射到数字索引；4. 将文本转换为数字索引序列

# 下面的函数将数据集读取到由多条文本行组成的列表中，其中每条文本行都是一个字符串
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine(): # @save
    # 将时间机器数据集加载到文本行的列表中
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines() # 读取文件的所有行，返回字符串列表
    # re.sub(pattern, replacement, string)：正则表达式替换，将所有非字母字符替换为空格
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# tokenize 函数将文本行列表作为输入，列表中每个元素是一个文本序列，文本序列被拆分成一个词元列表
# 词元（token）是文本的基本单位。最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串
def tokenize(lines, token='word'): # @save
    # 将文本行拆分为单词或字符词元，默认为 'word'(单词级别)
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        # list(line)：将字符串转换为字符列表
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 词表：用来将字符串类型的词元映射到从0开始的数字索引中
# 合并训练集中的所有文档，对唯一词元进行统计，称为语料
# 然后根据每个唯一词元的出现频率，为其分配一个数字索引

# 统计词元的频率
def count_corpus(tokens):  # @save
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
        # 等价的标准循环写法：
        # flattened_tokens = []
        # for line in tokens:          # 第一轮：line = ['hello', 'world']
        #     for token in line:       # 第一轮内层：token = 'hello', 然后 'world'
        #         flattened_tokens.append(token)
    # Counter对象，包含每个词元的出现次数
    return collections.Counter(tokens)

# 文本词表类
class Vocab: # @save
    # min_freq：最小频率阈值，低于此频率的词元被忽略
    # reserved_tokens：保留的特殊标记(<pad>, <bos>, <eos>)
    # '<pad>'(填充标记)  '<bos>'(序列开始)  '<eos>'(序列结束)
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序，获得 Counter对象
        counter = count_corpus(tokens)
        # counter.items()：转为[('the', 500), ...]；key=lambda x: x[1]：按第二个元素排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 创建列表，建立索引数字 → 词元字符串映射关系，返回一个有序的词元列表
        # ['<unk>'] + reserved_tokens：创建包含未知词元标记的列表 + 列表拼接
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 创建字典，建立词元字符串 → 索引数字的逆向映射关系，token: idx 为 字典[token] = idx
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break # freq < min_freq 的忽略
            # 添加不重复出现的索引，词元对到列表中
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        # 返回词表大小，即唯一词元数量
        return len(self.idx_to_token)

    # 将词元转换为索引，支持批量处理
    def __getitem__(self, tokens):
        # 检查 tokens 是单个词元（字符串）
        if not isinstance(tokens, (list, tuple)):
            # .get(key, default)：如果key存在返回对应值，否则返回default
            return self.token_to_idx.get(tokens, self.unk)
        # 检查 tokens 是列表或元组类型，则递归处理 tokens
        return [self.__getitem__(token) for token in tokens]

    # 将索引转换为词元，支持批量处理
    def to_tokens(self, indices):
        # 检查indices是单个索引（整数）
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 检查 indices 是列表或元组类型，则循环遍历处理 indices
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        # 返回词频的统计信息
        return self._token_freqs

# 返回时光机器数据集的词元索引列表和词表
def load_corpus_time_machine(max_tokens=-1): # @save
    # max_tokens 最大词元数量限制，-1表示不限制
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') # 字符级别
    vocab = Vocab(tokens)
    # 数据集中每个文本行不一定是句子或段落，所以将所有文本行展平到列表中
    corpus = [vocab[token] for line in tokens for token in line]
    # corpus = []                   # 等价代码
    # for line in tokens:           # 遍历每一行
    #     for token in line:        # 遍历行中的每个字符
    #         index = vocab[token]  # 将字符转换为索引
    #         corpus.append(index)  # 添加到语料库
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# corpus：整数列表，直接作为模型输入
# vocab：词表对象，字符到索引的映射关系
corpus, vocab = load_corpus_time_machine()
# print(len(corpus), len(vocab))
"""


"""
# 如果P(x_t+1 | x_t,..,x_1) = P(x_t+1 | x_t)，序列分布满足一阶马尔可夫性质
# 阶数越高，对应的依赖关系就越长。这种性质推导出了许多可以应用于序列建模的近似公式：
# P(x1, x2, x3, x4) = P(x1)P(x2)P(x3)P(x4),
# P(x1, x2, x3, x4) = P(x1)P(x2 | x1)P(x3 | x2)P(x4 | x3),
# P(x1, x2, x3, x4) = P(x1)P(x2 | x1)P(x3 | x1, x2)P(x4 | x2, x3).
# 涉及一个、两个和三个变量的概率公式分别被称为一元语法、二元语法和三元语法模型

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或段落，因此把所有文本行拼接
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
# print(vocab.token_freqs[:10])
# 流行的词词通常被称为停用词，可以被过滤掉
freqs = [freq for token, freq in vocab.token_freqs] # 画出词频图(一元语法)
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
# 词频迅速衰减，满足齐普夫定律，即第i个最常用单词的频率n_i为：
# log(n_i) = -α*log(i) + c，其中α是刻画分布的指数，c是常数

# 检查二元语法是否满足相同的规律
# zip(corpus[:-1], corpus[1:])：创建相邻词对(左闭右开)
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])

# 检查三元语法是否满足相同的规律
# zip(corpus[:-2], corpus[1:-1], corpus[2:])：创建连续三个词的组合(左闭右开)
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
# print(trigram_vocab.token_freqs[:10])

# 画图直观对比
bigram_freqs, trigram_freqs = [freq for token, freq in bigram_vocab.token_freqs], [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])
# d2l.plt.show()


# 实现 随机采样 和 顺序分区
# 从数据中随机生成一个小批量，num_steps 是每个子序列中预定义的时间步数(序列长度)
def seq_data_iter_random(corpus, batch_size, num_steps): # @save
    # 引入随机性，避免总是从固定位置开始采样
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 得到各个长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    # 计算总共可以生成多少个完整批量
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取批量索引：initial_indices[i: i + batch_size]
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        # 生成输入X：每个子序列；生成标签Y：每个子序列向右移动1位（预测下一个词）
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y) # yield 多次返回
    # my_seq = list(range(35))  # [0,1,2,...,34]
    # 步骤1：随机偏移
    # corpus = my_seq[random.randint(0,4):]  # 假设从索引2开始
    # corpus = [2,3,4,...,34] (长度33)
    # 步骤2：计算子序列
    # num_subseqs = (33-1) // 5 = 32 // 5 = 6
    # initial_indices = [0,5,10,15,20,25]  # 在corpus中的位置
    # random.shuffle(initial_indices)  # 假设变为 [15,5,25,0,20,10]
    # 步骤3：批量生成
    # 第1个批量 (i=0)：
    # initial_indices_per_batch = [15, 5]  # 取前2个索引
    # X = [data(15), data(5)] = [[15,16,17,18,19], [5,6,7,8,9]]
    # Y = [data(16), data(6)] = [[16,17,18,19,20], [6,7,8,9,10]]
    # 第2个批量，第3个批量类似
    # 输出结果示例(第1个批量)
    # X:  tensor([[15, 16, 17, 18, 19],[ 5,  6,  7,  8,  9]])
    # Y:  tensor([[16, 17, 18, 19, 20],[ 6,  7,  8,  9, 10]])

# my_seq = list(range(35))
# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    # print('X: ', X, '\nY:', Y)


# 迭代过程中，保证两个相邻的小批量中的子序列在原始序列上也是相邻的
# 这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序，称为顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps): # @save
    # 使用顺序分区生成一个小批量子序列，从随机偏移量开始划分序列
    offset = random.randint(0, num_steps) # 引入随机性，避免从固定位置开始
    # 计算可用数量，-1：为标签留出空间（Y比X右移1位）
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # Xs：输入序列(从offset开始，取num_tokens个元素)；Ys：标签序列(Xs右移1位)
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # 重塑逻辑：将一维序列转换为二维矩阵；行数：batch_size；列数：自动计算
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算每个批量可以分成多少个子序列
    # Xs.shape[1]：重塑后张量的列数（时间步总数）
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X, Y = Xs[:, i: i + num_steps], Ys[:, i: i + num_steps]
        yield X, Y # 生成批量数据，yield 多次返回
    # 假设数据：corpus = list(range(35))（0到34）
    # 参数：batch_size=2, num_steps=5
    # 步骤1：随机偏移和令牌计算
    # offset = random.randint(0,5)  # 假设offset=1
    # num_tokens = ((35-1-1) // 2) * 2 = 32
    # 步骤2：创建Xs和Ys
    # Xs = corpus[1:1+32]     # [1,2,3,...,32]
    # Ys = corpus[1+1:1+32+1] # [2,3,4,...,33]
    # 步骤3：重塑形状 # batch_size=2
    # Xs, Ys = Xs.reshape(2, -1), Ys.reshape(2, -1)
    # Xs: [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    #      [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
    # Ys: [[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
    #      [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
    # 步骤4：生成批量数据
    # 第1个批量 (i=0)：# 第0-4列
    # X, Y = Xs[:, 0:5], Ys[:, 0:5]
    # X: [[1,2,3,4,5],[17,18,19,20,21]]
    # Y: [[2,3,4,5,6],[18,19,20,21,22]]
    # 第2个批量，第3个批量类似


# 加载序列数据的迭代器
class SeqDataLoader: # @save
    # use_random_iter：是否使用随机迭代器；max_tokens：最大词元数量限制
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter: # 使用随机抽样迭代器
            self.data_iter_fn = d2l.seq_data_iter_random
        else: # 使用顺序分区迭代器
            self.data_iter_fn = d2l.seq_data_iter_sequential
        # self.corpus：数字索引序列；self.vocab：词表对象(用于索引 ↔ 词元转换)
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self): # 实现迭代器接口
        # __iter__方法：使类成为可迭代对象，当对实例使用for循环时自动调用，返回数据迭代器
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000): # @save
    # 返回时光机器数据集的迭代器和词表
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
"""