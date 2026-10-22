import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# Transformer编码器由多个相同层叠加，每层有两子层，分别是多头自注意力汇聚和基于位置前馈网络
# 在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出，每个子层都采用残差连接
# 对于序列中任何位置输入x∈R_d，满足sublayer(x)∈R_d，使残差连接满足x+sublayer(x)∈R_d
# 在残差连接的加法计算后，接着应用层规范化。最后输入序列对应每个位置，编码器都输出一个d维向量

# Transformer解码器也由多个相同层叠加而成，并且层中使用残差连接和层规范化。除编码器中
# 描述的两个子层之外，解码器还在两个子层之间插入了第三个子层，称为编码器－解码器注意力层
# 在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出
# 在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但解码器中每个位置只能考虑
# 该位置之前所有位置。这种掩蔽注意力保留了自回归属性，确保预测仅依赖于已生成的输出词元

# 基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机
class PositionWiseFFN(nn.Module): # @save
    # 基于位置的前馈网络
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# ffn = PositionWiseFFN(4, 4, 8).eval()
# print(ffn(torch.ones((2, 3, 4)))[0]) # 输出 ffn 第一层的结果

# 层规范化和批量规范化目标相同，但基于特征维度进行
# 以下代码对比不同维度的层规范化和批量规范化的效果
# ln, bn = nn.LayerNorm(2), nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算X的均值和方差并输出结果
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module): # @save
    # 残差连接后进行层规范化
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        # X: 原始输入, Y: 子层输出
        return self.ln(self.dropout(Y) + X)

# add_norm = AddNorm([3, 4], 0.5).eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

# Transformer 编码器部分
class EncoderBlock(nn.Module): # @save
    # Transformer编码器块
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # LayerNorm(X + Dropout(attention_output))
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # LayerNorm(Y + Dropout(ffn_output))
        return self.addnorm2(Y, self.ffn(Y))

# X: 形状(2, 100, 24) -> 2个批次，100个时间步，24维特征
# X, valid_lens = torch.ones((2, 100, 24)), torch.tensor([3, 2])
# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5).eval()
# print(encoder_blk(X, valid_lens).shape) # 输出形状不变，torch.Size([2, 100, 24])

class TransformerEncoder(d2l.Encoder): # @save
    # Transformer编码器
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), EncoderBlock(key_size, query_size, value_size,
            num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 嵌入向量的模长通常较大，缩放后与位置编码尺度匹配
        # 位置编码值[-1, 1]，因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens) # 逐层处理：每个编码器块处理输入并更新X
            self.attention_weights[i] = blk.attention.attention.attention_weights # 记录每层多头注意力权重
        return X

# 创建一个两层的Transformer编码器，Transformer编码器输出的形状是（批量大小，时间步数目，num_hiddens）
# encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5).eval()
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape) # torch.Size([2, 100, 24])

# Transformer 解码器部分
# 为保留自回归属性，掩蔽自注意力设定参数dec_valid_lens，使查询只会与已生成词元位置进行注意力计算
class DecoderBlock(nn.Module):
    # 解码器中第i个块
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state: 包含三部分信息的元组：state[0]: 编码器输出 enc_outputs
        # state[1]: 编码器有效长度 enc_valid_lens；state[2]: 解码器状态列表
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，state[2][self.i]为None
        if state[2][self.i] is None:
            key_values = X # 训练时使用整个输入序列
        # 预测阶段，输出序列通过词元一个接一个解码，state[2][self.i]包含直到第i个块解码的输出表示
        else: # 逐步累积已经生成的词元
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        # 因果掩码生成
        if self.training: # 训练阶段
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps)，其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else: # 预测阶段
            dec_valid_lens = None
        # 掩蔽自注意力输出
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        # 等价于 Y = LayerNorm(X + Dropout(X2)) # 残差连接 + 层归一化
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        # 等价于 Z = LayerNorm(Y + Dropout(Y2)) # 残差连接 + 层归一化
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24,48, 8, 0.5, 0).eval()
# X = torch.ones((2, 100, 24))
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape) # (2, 100, 24)

class TransformerDecoder(d2l.AttentionDecoder):
    # 完整的 Transformer 解码器
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size, value_size,
            num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 初始化注意力权重存储：创建 2 * num_layers 的二维列表
        # [0][i]: 第i层的解码器自注意力权重；[1][i]: 第i层的编码器-解码器注意力权重
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state) # 每个解码器块处理输入并更新状态
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state # state: 更新后的状态（包含解码器缓存）

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

if __name__=='__main__':
    # 实例化 Transformer 架构并训练模型
    num_hiddens, num_layers, dropout, batch_size, num_steps, lr = 32, 2, 0.1, 64, 10, 0.005
    ffn_num_input, ffn_num_hiddens, num_heads, num_epochs, device = 32, 64, 4, 200, d2l.try_gpu()
    key_size, query_size, value_size, norm_shape = 32, 32, 32, [32]
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)

    # 为了正常运行而修改的函数
    def train_seq2seq_custom(net, data_iter, lr, num_epochs, tgt_vocab, device):
        """自定义序列到序列训练函数，处理网络输出解包问题"""
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])
        net.apply(xavier_init_weights)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
        net.train()
        for epoch in range(num_epochs):
            metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
            for batch in data_iter:
                optimizer.zero_grad()
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
                # 处理网络输出，避免解包错误
                output = net(X, dec_input, X_valid_len)
                if isinstance(output, tuple):
                    Y_hat = output[0]  # 如果是元组，取第一个元素作为预测
                else:
                    Y_hat = output  # 如果不是元组，直接使用输出
                l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1))
                l.sum().backward()
                d2l.grad_clipping(net, 1)
                num_tokens = Y_valid_len.sum()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l.sum(), num_tokens)
            if (epoch + 1) % 10 == 0:
                print(f'epoch {epoch + 1}, loss {metric[0] / metric[1]:.3f}')

    # 使用自定义的训练函数，避免解包错误
    train_seq2seq_custom(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 翻译句子并计算 BLEU 分数
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    # print(f'{eng} => {translation}, ',f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

    # 编码器自注意力权重的形状为（编码器层数，注意力头数，num_steps或查询的数目，num_steps或“键－值”对的数目）
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
    # print(enc_attention_weights.shape) # torch.Size([2, 4, 10, 10])
    d2l.show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions',ylabel='Query positions',
                      titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

    # 解码器的自注意力权重和“编码器－解码器”的注意力权重都有相同的查询：即以序列开始词元 BOS 打头，再与后续输出的词元共同组成序列
    dec_attention_weights_2d = [head[0].tolist()for step in dec_attention_weight_seq for attn in step for blk in attn for head in blk]
    # for step in dec_attention_weight_seq:   # 遍历每个时间步
    #     for attn in step:                   # 遍历每种注意力类型
    #         for blk in attn:                # 遍历每个解码器块
    #             for head in blk:            # 遍历每个注意力头
    #                 head[0].tolist()        # 取第一个批次，转换为列表
    # 结果：得到一个扁平的2D列表，包含所有时间步、注意力类型、块和头的注意力权重
    dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
    dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4) # 维度重排列和分离
    # permute(...)：[批次,类型,层,头,步] -> [类型,层,头,批次,步]
    # 然后分离为两个张量，分别对应解码器自注意力和编码器-解码器注意力
    # print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)

    d2l.show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],xlabel='Key positions',
                      ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    d2l.show_heatmaps(dec_inter_attention_weights, xlabel='Key positions', ylabel='Query positions',
                      titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    # d2l.plt.show()