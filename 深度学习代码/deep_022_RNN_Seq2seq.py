import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


# 两个主要架构:编码器：接受一个长度可变序列作为输入，并转换为固定形状编码状态
# 解码器：将固定形状的编码状态映射到长度可变的序列。以上这些被称为编码器-解码器
class Encoder(nn.Module): # @save
    # 编码器-解码器架构的基本编码器接口
    def __init__(self, **kwargs):
        # 使用 **kwargs 接受任意关键字参数传递给父类
        super(Encoder, self).__init__(**kwargs)
    def forward(self, X, *args):
        # 注意 NotImplementedError 表示抛出未实现错误，
        # 要求子类必须重写此方法，说明编码器和解码器必须结合使用
        raise NotImplementedError

# 解码器在每个时间步都会将输入和编码后的状态映射成当前时间步的输出词元
class Decoder(nn.Module): # @save
    # 编码器-解码器架构的基本解码器接口
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    # 用于将编码器的输出转换为解码器的初始状态
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module): # @save
    # 编码器-解码器架构的基类
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args) # 编码器处理输入序列
        dec_state = self.decoder.init_state(enc_outputs, *args) # 用编码器输出初始化解码器状态
        return self.decoder(dec_X, dec_state) # 解码器基于初始状态生成输出序列

# 在时间步t，循环神经网络将词元 x_t 的输入特征向量 x_t 转换为 h_t = f(x_t, h_t−1)，
# 然后编码器通过函数q，将所有时间步的隐状态转换为上下文变量：c = q(h_1, . . . , h_T)
# 使用嵌入层来获得输入序列中每个词元的特征向量，其中行数=输入词表大小，列数=特征向量维度

class Seq2SeqEncoder(d2l.Encoder): # @save
    # 用于序列到序列学习的循环神经网络编码器
    # embed_size: 词嵌入维度；num_layers: RNN层数
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 创建嵌入层：将词索引转换为密集向量；创建GRU循环神经网络
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出 X 形状(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 使用 permute(1, 0, 2) 调整维度顺序
        # 因为PyTorch的 RNN 要求序列长度在第一个维度
        X = X.permute(1, 0, 2)
        # output: 每个时间步的隐藏状态；state: 最后时间步的隐藏状态
        output, state = self.rnn(X) # 如果未提及状态，则默认为0
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# encoder.eval() # 设置模型为评估模式：关闭 dropout 等训练时的随机性，确保前向传播的一致性
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = encoder(X)
# print(output.shape, state.sharp) # (7,4,16) / (2,4,16)

# 对于每个时间步 t′，解码器输出 y_t′ 的概率取决于先前的输出子序列
# y_1,..., y_t′−1 和上下文变量 c，即P(y_t′ | y_1,..., y_t′−1, c)
# 在输出序列上的时间步t′，隐状态s_t′表示为：s_t' = g(y_t′−1, c, s_t′−1)
class Seq2SeqDecoder(d2l.Decoder):
    # 用于序列到序列学习的循环神经网络解码器
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size) # 与编码器相同的嵌入层
        # rnn输入大小: embed_size + num_hiddens，因为解码器要同时看到：当前输入词的嵌入和编码器的上下文信息
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # dense: 全连接层，将隐藏状态映射回词汇表概率分布
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] # enc_outputs[1] 取编码器的最终隐藏状态 state

    def forward(self, X, state):
        # 输出 X 的形状：(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播 context，使其具有与 X 相同的 num_steps
        # state[-1]: 取编码器最后一层的隐藏状态
        # .repeat(X.shape[0], 1, 1): 沿时间步维度复制
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        # output的形状:(num_steps,batch_size,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        output, state = self.rnn(X_and_context, state)
        # 调整回 (batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

# decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval() # 设置模型为评估模式：关闭 dropout 等训练时的随机性，确保前向传播的一致性
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape) # (4,7,10) / (2,4,16)

# sequence_mask 函数通过零值化屏蔽不相关项
def sequence_mask(X, valid_len, value=0): # @save
    maxlen = X.size(1) # 获取序列的最大长度
    # [None, :] 变为(1, maxlen)，在第一个维度增加一个维度
    # [:, None] 变为(batch_size,1)，第二个维度增加一个维度
    # 广播后：每个批次样本的位置索引与对应的有效长度比较
    # 结果生成形状为 (batch_size, maxlen) 的布尔掩码
    mask = torch.arange((maxlen), dtype=torch.float32,
        device=X.device)[None, :] < valid_len[:, None]
    # 取反，将所有应该屏蔽的位置设置为指定值(默认0)
    X[~mask] = value
    return X

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))
# 输出结果：tensor([[1, 0, 0], [4, 5, 0]])
# X = torch.ones(2, 3, 4) # 使用指定非零值 -1 填充
# print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss): # @save
    # 带遮蔽的softmax交叉熵损失函数
    # pred 形状：(batch_size,num_steps,vocab_size)
    # label 形状：(batch_size,num_steps)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none' # 'none' 表示不进行汇总，返回每个位置的损失值
        # pred.permute(0,2,1) 变为 (batch_size, vocab_size, num_steps)
        # 因为PyTorch的CrossEntropyLoss要求通道维度在第二维
        # 计算未加权的损失：调用父类的交叉熵损失计算，返回形状(batch_size, num_steps)，即每个位置一个损失值
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 用掩码加权，屏蔽位置的损失变为0；沿序列维度求平均；返回形状(batch_size,)，即每个样本的平均损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# loss = MaskedSoftmaxCELoss() # 输出结果：tensor([2.3026, 1.1513, 0.0000])
# print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))
# pred = torch.ones(3, 4, 10)    # 所有预测值相同
# label = torch.ones((3, 4))       # 所有标签为1
# valid_len = torch.tensor([4, 2, 0]) # 有效长度

# 步骤分解：创建权重掩码：
# weights = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
# weights = sequence_mask(weights, [4,2,0])
# # 结果: [[1,1,1,1], [1,1,0,0], [0,0,0,0]]
# 由于所有预测值相同，每个词相等的概率：0.1
# 交叉熵损失：-log(0.1) = 2.3026
# unweighted_loss: 所有位置都是2.3026
# [[2.3026, 2.3026, 2.3026, 2.3026],
#  [2.3026, 2.3026, 2.3026, 2.3026],
#  [2.3026, 2.3026, 2.3026, 2.3026]]
# 加权和平均：样本1: (2.3026×4) / 4 = 2.3026
# 样本2: (2.3026×2) / 2 = 1.1513 （注意：平均时只除有效长度2）
# 样本3: (0) / 1 = 0.0000 （有效长度为0，但避免除0）
# [2.3026, 1.1513, 0.0000] 对应3个样本的最终损失

# 训练函数
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device): # @save
    # 训练序列到序列模型；tgt_vocab: 目标词汇表
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # GRU需要遍历所有权重参数进行初始化
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # apply(): 递归地对所有子模块应用初始化函数
    net.apply(xavier_init_weights).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train() # 设置模型为训练模式
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2) # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 为每个样本创建开始符号 <bos>；Y.shape[0] 是批次大小；结果形状：(batch_size, 1)
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            # 关键步骤，构建解码器输入：Y[:, :-1]: 目标序列去掉最后一个词元
            # 拼接结果：[<bos>, y1, y2, ..., yn-1]，形状：(batch_size, tgt_len)
            # 教师强制原理：真实目标序列 Y: [y1, y2, y3, ..., yn, <eos>]
            # 解码器输入[<bos>,y1,y2,...,yn-1]；模型预测[y1,y2,y3,...,yn,<eos>]
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
            # 预测形状(batch_size,tgt_len,vocab_size)，每个位置输出词汇表大小概率分布
            Y_hat = net(X, dec_input, X_valid_len)
            # Y_valid_len: 用于屏蔽填充位置
            # l 形状(batch_size,)，每个样本的平均损失
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1) # 梯度裁剪
            num_tokens = Y_valid_len.sum() # 计算当前批次的有效词元总数
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    d2l.plt.show()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} 'f'tokens/sec on {str(device)}')


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

# 预测部分(序列到序列模型的预测)
# 每个解码器当前时间步的输入都将来自于前一时间步的预测词元(一个接着一个词元的方式预测输出序列)
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False): # @save
    # src_sentence: 源语言句子；save_attention_weights: 是否保存注意力权重（用于可视化）
    net.eval() # 在预测时将net设置为评估模式
    # 将句子转为小写并按空格分割成词元列表，然后将源句子转换为词元ID序列
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # torch.unsqueeze(..., dim=0)在维度0添加批量轴，形状变化[num_steps] -> [1, num_steps]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 准备解码器的起始输入：torch.unsqueeze(..., dim=0)添加批量轴,形状：[1, 1] (批量大小=1，序列长度=1)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], [] # 储存输出序列和注意力权重序列
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性词元，作为解码器在下一时间步输入
        dec_X = Y.argmax(dim=2) # 在词汇表维度取最大值索引
        # 提取预测结果：squeeze(dim=0)移除批量维度 [1, 1] -> [1]
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重(稍后讨论)
        if save_attention_weights:
            # 如果启用注意力权重保存，将当前步的注意力权重添加到序列中
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    # 返回结果：将预测的词元ID序列转换回文本字符串
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k): # @save
    # 计算BLEU；k: 考虑的最大n-gram长度
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算长度惩罚因子(BP)：惩罚过短的预测，鼓励生成适当长度的序列
    # 1 - len_label/len_pred: 如果预测序列比标签长，这个值为正
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1): # 循环计算1-gram到k-gram的精度
        # num_matches: 统计匹配的n-gram数量
        # label_subs: 使用默认字典统计标签中每个n-gram的出现次数
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            # 遍历标签序列中的所有可能的n-gram
            # 将n-gram用空格连接成字符串作为键，计数加1
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            # 遍历预测序列中的所有可能的n-gram
            # 如果当前预测的n-gram在标签中存在且计数>0
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1 # 匹配数加1
                # 将标签中对应n-gram计数减1，确保每个n-gram在标签中最多被匹配一次
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # 更新总分数：其中，num_matches/(len_pred - n + 1):n-gram精度
        # math.pow(0.5, n):给更高阶n-gram更小的权重；将所有n-gram的加权精度相乘
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__=='__main__':
    embed_size, num_hiddens, num_layers, dropout = 128, 256, 2, 0.1
    batch_size, num_steps, lr, num_epochs, device = 128, 10, 0.004, 300, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)  # 修改了函数

    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .'] # 英文测试句子列表
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .'] # 法文参考翻译列表
    for eng, fra in zip(engs, fras): # zip(engs, fras) 创建配对：('go .', 'va !'),...
        # 返回：translation(预测的法文), attention_weight_seq(注意力权重)
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')