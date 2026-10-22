import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
from torch import nn
from d2l import torch as d2l


"""
# 可微注意力机制
class AttentionDecoder(d2l.Decoder): # @save
    # 带有注意力机制解码器的基本接口
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    @property
    def attention_weights(self):
        raise NotImplementedError

# 首先初始化解码器的状态，需要下面的输入：
# 1. 编码器在所有时间步的最终层隐状态，将作为注意力的键和值
# 2. 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态
# 3. 编码器有效长度（排除在注意力池中填充词元）
# 在每个解码时间步骤中，解码器上一个时间步的最终层隐状态将用作查询
# 因此，注意力输出和输入嵌入都连结为循环神经网络解码器的输入
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 加性注意力机制：用于计算编码器输出与解码器状态的注意力，键、查询、值的维度都是num_hiddens
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs：所有时间步隐藏状态形状(batch_size，num_steps，num_hiddens)
        # hidden_state：最终隐藏状态形状(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        # 返回包含编码器输出、隐藏状态和有效长度的三元组
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X: # 遍历每个时间步的输入
            # 使用解码器上一时间步的最终层隐藏状态作为查询
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 计算注意力上下文：query: 解码器当前状态
            # enc_outputs 同时作为键和值，context 形状(batch_size,1,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 连接输入：将注意力上下文与当前输入词嵌入连接
            # x 输出形状为：(batch_size, 1, embed_size + num_hiddens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为 (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

# 修复编码问题，替换原来的数据加载函数
def load_data_nmt(batch_size, num_steps, num_examples=600):
    # 返回翻译数据集的迭代器和词表
    def read_data_nmt_fixed(): # 修复读取函数
        data_dir = d2l.download_extract('fra-eng')
        with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
            return f.read()

    text = read_data_nmt_fixed()
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))

    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# 接下来，使用包含7个时间步的4个序列输入的小批量测试 Bahdanau 注意力解码器
# encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2).eval()
# decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2).eval()
# X = torch.zeros((4, 7), dtype=torch.long) # (batch_size, num_steps)
# state = decoder.init_state(encoder(X), None)
# output, state = decoder(X, state) # 返回输出和更新后的状态
# state[0],  # enc_outputs: (7, 4, 16) - 编码器所有时间步输出
# state[1],  # hidden_state: 包含2个张量的元组，每个形状(4, 16)
# print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
# 输出结果(torch.Size([4, 7, 10]), 3, torch.Size([4, 7, 16]), 2, torch.Size([4, 16]))

# 训练模型
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 将几个英语句子翻译成法语并计算它们的BLEU分数
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    # 输出生成翻译结果和解码过程中的注意力权重序列
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
    net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

# dec_attention_weight_seq：形状[时间步数, (batch, num_heads, query_len, key_len)]
# 单头注意力中每个step形状为(1,1,query_len,key_len)；step[0]: 取第一个批次；step[0][0]: 取第一个注意力头
# step[0][0][0]:取第一个查询位置对应的所有键的注意力权重；结果：得到一个列表，包含每个解码时间步的注意力权重向量
# torch.cat(..., 0):在第0维度连接所有时间步的注意力权重；.reshape((...)):重塑为4D张量适应show_heatmaps函数
# (1,1,num_decoder_steps,num_steps):第一个1：显示的行数；第二个1：显示的列数；num_decoder_steps：查询位置；num_steps：键位置
attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((1, 1, -1, num_steps))
# :len(engs[-1].split()) + 1: 切片到第4个位置（索引0-3）；+1是因为通常包含起始符<bos>
# x轴表示键位置(编码器时间步，对应源语言单词)；y轴表示查询位置(解码器时间步，对应目标语言单词)
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(), xlabel='Key positions', ylabel='Query positions')
# d2l.plt.show()
"""



""" 
# 多头注意力机制
# 多头注意力：用独立学习得到的h组不同的线性投影来变换查询、键和值。将h组变换后的查询、键和值并行送到注意力汇聚中
# 最后将h个注意力汇聚的输出拼接在一起，并通过另一个可以学习的线性投影进行变换，以产生最终输出。这被称为多头注意力
# 每个注意力头 h_i(i = 1,..., h) 的计算方法为：h_i = f(W_i(q) * q, W_i(k) * k, W_i(v) * v) ∈ R_pv
class MultiHeadAttention(nn.Module): # @save
    # 多头注意力
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 变换后形状:(batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # valid_lens 形状:(batch_size，)或(batch_size，查询的个数)
        if valid_lens is not None:
            # 将有效长度复制num_heads次，因为每个注意力头都需要相同的掩码
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
            output = self.attention(queries, keys, values, valid_lens)
            # output_concat的形状:(batch_size，查询的个数，num_hiddens)
            output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# 多头分割（在transpose_qkv函数中）：
# 假设 num_hiddens = 64, num_heads = 8
# 投影后: (batch_size, num_queries, 64)
# 重塑: (batch_size, num_queries, 8, 8)  # 8个头，每个头8维
# 转置: (batch_size, 8, num_queries, 8)  # 将头维度提前
# 重塑: (batch_size*8, num_queries, 8)   # 合并批次和头维度
def transpose_qkv(X, num_heads): # @save
    # 为了多注意力头的并行计算而变换形状
    # 1.输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 1.输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 2.输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 3.最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 输入: (batch_size*num_heads, num_queries, num_hiddens/num_heads)
# 重塑: (batch_size, num_heads, num_queries, num_hiddens/num_heads)
# 转置: (batch_size, num_queries, num_heads, num_hiddens/num_heads)
# 重塑: (batch_size, num_queries, num_hiddens)  # 合并头维度
def transpose_output(X, num_heads): # @save
    # 逆转transpose_qkv函数的操作
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

num_hiddens, num_heads, batch_size, num_queries, num_kvpairs, valid_lens = 100, 5, 2, 4, 6, torch.tensor([3, 2])
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5).eval()
X, Y = torch.ones((batch_size, num_queries, num_hiddens)), torch.ones((batch_size, num_kvpairs, num_hiddens))
# print(attention(X, Y, Y, valid_lens).shape) # 多头注意力输出(batch_size，num_queries，num_hiddens) -> (2, 4, 100)
"""



""" 
# 自注意力
# 查询、键和值来自同一组输入，称为自注意力：y_i = f(x_i, (x_1, x_1), . . . ,(x_n, x_n)) ∈ R_d
# 卷积神经网络 O(k*n*d^2) 和自注意力 O(n*d^2) 都拥有并行计算的优势，而且自注意力的最大路径长度最短 O(1)
# 位置编码：为了使用序列的顺序信息，通过在输入表示中添加位置编码来注入绝对的或相对的位置信息
# 位置编码使用相同形状的位置嵌入矩阵 P 输出 X + P，第i行、2j列和2j + 1列上的元素为(10.6.2)
class PositionalEncoding(nn.Module): # @save
    # 位置编码
    def __init__(self, num_hiddens, dropout, max_len=1000):
        # num_hiddens: 编码维度；max_len: 支持的最大序列长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 P 并初始化为零
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow
        (10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        self.P[:, :, 0::2] = torch.sin(X) # 偶数维度：sin(i / 10000^(2j/d))
        self.P[:, :, 1::2] = torch.cos(X) # 奇数维度：cos(i / 10000^(2j/d))

    def forward(self, X):
        # P[:, :X.shape[1], :]:只取实际序列长度的位置编码
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0).eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :] # 获取位置编码矩阵P：(1, 60, 32)
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
      figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
# d2l.plt.show()

# 绝对位置信息
# 为了明白沿着编码维度单调降低的频率与绝对位置信息的关系，打印0, 1,..., 7的二进制表示形式
# 二进制表示中，较高比特位的交替频率低于较低比特位，只是位置编码通过使用三角函数在编码维度上降低频率
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
# d2l.plt.show()

# 相对位置信息
# 相对位置信息：对于任何确定的位置偏移δ，位置i + δ处的位置编码可以线性投影位置i处的位置编码来表示
"""