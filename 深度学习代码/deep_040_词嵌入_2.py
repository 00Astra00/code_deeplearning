import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import torch
import collections
from torch import nn
from d2l import torch as d2l


# 预训练word2vec
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size, num_noise_words)

# 嵌入层的输入是词元的索引。对于任何词元索引i，其向量表示可以从嵌入层中的权重矩阵的第i行获得
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, 'f'dtype={embed.weight.dtype})')

# 两个变量(center、contexts_and_negatives)通过嵌入层从词元索引转换成向量，然后批量矩阵相乘
# 返回形状（批量大小，1，max_len）的输出。输出中的每个元素是中心词向量和上下文或噪声词向量的点积
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    # 双嵌入层设计：embed_v：中心词嵌入层；embed_u：上下文词嵌入层
    # center：中心词索引contexts_and_negatives：上下文和负样本索引
    v, u = embed_v(center), embed_u(contexts_and_negatives)
    # 批量矩阵乘法：每个元素是中心词向量与上下文/负样本向量的点积
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

print(skip_gram(torch.ones((2, 1), dtype=torch.long),
torch.ones((2, 4), dtype=torch.long), embed, embed).shape)

class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
              inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
      nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))

def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            # 关键损失计算：pred.reshape(label.shape)：调整预测值形状匹配标签
            # / mask.sum(axis=1)：除以有效位置数（归一化）；* mask.shape[1]：乘以最大长度（恢复原始尺度）
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, 'f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)

# 使用训练好模型中词向量的余弦相似度来从词表中找到与输入单词语义最相似的单词
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data # 嵌入层权重矩阵
    x = W[vocab[query_token]] # 查询词的向量
    # 余弦相似度计算：增加1e-9以获得数值稳定性
    # mv(W, x)：得到所有词与查询词点积；sum(W * W, dim=1)：每个词向量L2范数平方
    # sum(x * x)：查询词向量的L2范数平方；公式：cos = (W·x) / (||W|| * ||x||)
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32') # 找到最相似的k+1个词
    for i in topk[1:]: # 删除输入词，打印相似词和相似度
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

print(get_similar_tokens('chip', 3, net[0]))




# 全局向量的词嵌入（GloVe）
# 在整个语料库中，所有以w_i为中心词的上下文词形成一个词索引的多重集C_i，元素的实例数称为其重数
# 多重集C_i中元素j的重数为x_ij，即词w_j(上下文词)和词w_i(中心词)在语料库同一上下文窗口中的全局共现计数
# 跳元模型的损失函数 (14.5.2)，x_i表示上下文窗口中所有上下文词数量，其中w_i作为中心词出现，相当于|C_i|
# 设 p_ij 为用于生成上下文词w_j的条件概率 x_ij / x_i。给定中心词w_i，(14.5.2) 可以重写为 (14.5.3)
# −sigma(p_ij * log(q_ij))计算全局语料统计条件分布p_ij和模型预测的条件分布q_ij交叉熵，损失按x_i加权

# GloVe模型基于平方损失对跳元模型做三个修改：
# 1.使用变量p′_ij=x_ij和q′_ij=exp(u_j.T * v_i)，取两者对数，平方损失项(log(p′_ij) − log(q′_ij))^2
# 2.每个词wi添加标量参数：中心词偏置bi和上下文词偏置ci；3.权重函数h(x_ij)替换每个损失项权重，h在[0,1]递增
# 训练GloVe降低损失函数：sigma(i∈V, sigma(j∈V, h(x_ij)*(u_j.T * v_i + b_i + c_j - log(x_ij))^2))
# 当词w_i出现在词w_j上下文窗口时，词w_j也出现在词w_i的上下文窗口(xij = xji)，即GloVe拟合对称概率log(x_ij)
# 因此，在GloVe模型中，任意词的中心词向量和上下文词向量在数学上是等价的


# 子词嵌入
# 为使用形态(单词形成和词汇关系)信息，fastText模型提出子词嵌入方法，其中子词是一个字符n‐gram
# fastText可以被认为是子词级跳元模型，而非学习词级向量表示，其中每个中心词由其子词级向量之和表示
# 由于相似结构的词之间共享来自子词的参数，罕见词甚至词表外的词在fastText中可能获得更好的向量表示

# 在 fastText 中，所有提取子词必须是指定长度，因此词表大小不能预定义
# 为在固定大小词表中允许可变长度子词，可用一种称为字节对编码的压缩算法来提取子词
# 字节对编码执行训练数据集统计分析，以发现单词内公共符号，如任意长度的连续字符。从长度为1的符号开始，
# 字节对编码迭代合并最频繁连续符号对以产生新的更长符号(不考虑跨边界)。最后使用类似子词符号来切分单词

# 首先，将符号词表初始化为所有英文小写字符、特殊的词尾符号'_'和特殊的未知符号'[UNK]'
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]']

# 特殊符号'_'被附加到每个词的尾部，以便容易地从输出符号序列恢复单词序列。由于仅从单个字符和特殊符号
# 的词开始合并处理，所以在每个词（词典token_freqs的键）内的每对连续字符间插入空格(词中符号间分隔符)
token_freqs = {}
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
print(token_freqs)

def get_max_freq_pair(token_freqs):
    # 返回词内最频繁的连续符号对
    pairs = collections.defaultdict(int) # 创建默认值为0的字典
    for token, freq in token_freqs.items(): # 遍历词表中的每个词及其频率
        symbols = token.split() # 将词拆分为符号列表
        for i in range(len(symbols) - 1): # 遍历词中的每个相邻符号对
            # “pairs”的键是两个连续符号的元组，该步骤以符号对为键
            pairs[symbols[i], symbols[i + 1]] += freq # 累加该词的出现频率
    return max(pairs, key=pairs.get) # 具有最大值的“pairs”键

def merge_symbols(max_freq_pair, token_freqs, symbols):
    # 字节对编码合并最频繁的连续符号对以产生新符号
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token] # 只改变词的表示形式，不改变频率
    return new_token_freqs

num_merges = 10
for i in range(num_merges): # 对词典键迭代执行字节对编码算法
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'合并# {i+1}:',max_freq_pair)
# print(symbols) # 多出'ta','tal','tall','fa','fas','fast','er','er_','tall_','fast_'
# print(list(token_freqs.keys())) # ['fast_', 'fast er_', 'tall_', 'tall er_']

def segment_BPE(tokens, symbols):
    # 尝试将单词从输入参数symbols分成可能最长子词
    outputs = []
    for token in tokens:
        # 当前处理位置开始 / 结束索引
        start, end = 0, len(token)
        cur_output = [] # 存储当前词的分词结果
        # 具有符号中可能最长子字的词元段
        # 算法的核心部分，使用贪心最长匹配策略：
        # 从最长可能子词开始：end = len(token)
        while start < len(token) and start < end:
            # 找到匹配：如果 token[start:end] 在符号表中
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end # 移动指针：跳到下一个位置
                end = len(token) # 重置查找：从最长开始
            else:
                end -= 1 # 逐步缩短：如果不匹配，end -= 1
        if start < len(token): # 说明有无法匹配的字符
            cur_output.append('[UNK]') # 表示未知部分
        outputs.append(' '.join(cur_output))
    return outputs

tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
