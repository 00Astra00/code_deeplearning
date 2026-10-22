import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import torch
from torch import nn
from d2l import torch as d2l


# 下载 aclImdb 数据集
d2l.DATA_HUB['aclImdb'] = ( # @save
'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
'01ada507287d82875905620988597833ad4e0903')
data_dir = d2l.download_extract('aclImdb', 'aclImdb')

def read_imdb(data_dir, is_train): # @save
    # 读取IMDb评论数据集文本序列和标签
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

# train_data = read_imdb(data_dir, is_train=True)
# print('训练集数目：', len(train_data[0])) # 训练集数目：25000

# 预处理数据集
# train_tokens = d2l.tokenize(train_data[0], token='word')
# vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

# num_steps = 500 # 序列长度
# train_features = torch.tensor([d2l.truncate_pad(
# vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
# print(train_features.shape) # torch.Size([25000, 500])

# train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

# 整合代码
def load_data_imdb(batch_size, num_steps=500): # @save
    # 返回数据迭代器和IMDb评论数据集的词表
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    # d2l.truncate_pad(ids, num_steps, pad_id)：序列长度 > num_steps：截断；序列长度 < num_steps：用<pad>填充
    train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)
    return train_iter, test_iter, vocab

# 基于 RNN 情感分析
batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # bidirectional=True,获取双向循环神经网络；每个时间步的输出维度：2 * num_hiddens（前向+后向）
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs(批量大小时间步数)，LSTM要求序列维度在第一维
        # 词嵌入：输出形状(seq_len, batch_size, embed_size)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # outputs：所有时间步隐藏状态，形状(seq_len, batch_size, 2*num_hiddens)
        outputs, _ = self.encoder(embeddings) # LSTM前向传播；_：最终隐藏状态
        # 连结初始和最终时间步隐状态，作为全连接层的输入，形状(批量大小，4*隐藏单元数)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding) # 分类输出：形状(batch_size, 2)
        return outs

devices = d2l.try_all_gpus()
embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)

glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
print(embeds.shape) # torch.Size([49346, 100])

net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

def predict_sentiment(net, vocab, sequence): # @save
    # 预测文本序列的情感
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is so bad')

# 基于 CNN 情感分析
# 一维卷积运算函数实现
def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y

def corr1d_multi_in(X, K):
    # 遍历'X'和'K'的第0维（通道维），然后加在一起
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
print(corr1d_multi_in(X, K)) # (2,8,14,20,26,32)

# 使用汇聚层从序列表示中提取最大值，作为跨时间步的最重要特征
# textCNN中使用的最大时间汇聚层的工作原理类似于一维全局汇聚
# 对每个通道在不同时间步存储值的多通道输入，每个通道输出是该通道最大值

# textCNN模型将输入转换为输出，如下所示：
# 1.定义多个一维卷积核，分别对输入执行卷积运算，不同宽度的卷积核可以捕获不同数目的相邻词元间的局部特征
# 2.所有输出通道执行最大时间汇聚层，将所有标量汇聚输出连结为向量；3.使用全连接层将连结后向量转为输出类别
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练，这种方式结合学习到的表示和预训练的静态表示
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            # 多尺度卷积层：输入通道：2 * embed_size
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，即每个嵌入层的输出形状（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1) # 根据一维卷积层输入格式，重新排列张量，将通道作为第2维
        # 多尺度特征提取过程（对每个卷积层）：cat(..., dim=1)：沿通道维连接所有卷积层输出
        # conv(embeddings)：一维卷积，形状(batch_size, out_channels, new_seq_len)
        # self.pool(...)：全局平均池化，形状(batch_size, out_channels, 1)
        # squeeze(..., dim=-1)：移除最后一个维度，形状(batch_size, out_channels)
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
        for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

devices = d2l.try_all_gpus()
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
       nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False

lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

d2l.predict_sentiment(net, vocab, 'this movie is so great')
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
