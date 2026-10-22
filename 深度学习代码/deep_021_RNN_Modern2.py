import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l


""" 
# 深层循环神经网络
# 第 l 个隐藏层：H_t(l) = ϕ_l(H_t(l-1) * W_xh(l) + H_t-1(l) * W_hh(l) + b_h(l)
# 最终输出层：O_t = H_t(l) * W_hq + b_q，其中隐藏层数目 l 和隐藏单元数目 h 都是超参数
# 简洁实现(唯一的区别是指定了层的数量)
batch_size, num_steps, num_epochs, lr = 32, 35, 200, 2
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs, device = vocab_size, d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab)).to(device)
# d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
# d2l.plt.show()

# 双向循环神经网络
# 双向循环神经网络添加了反向传递信息的隐藏层
# 双向循环神经网络主要用于序列编码和给定双向上下文的观测估计
# 前向隐状态：H_t(->) = ϕ(X_t * W_xh(f) + H_t-1(->) * W_hh(f) + b_h(f))
# 后向隐状态：H_t(<-) = ϕ(X_t * W_xh(b) + H_t-1(<-) * W_hh(b) + b_h(b))
# 输出层：O_t = H_t * W_hq + b_q

# 以下代码是双向循环神经网络的错误应用
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers, num_epochs, lr = len(vocab), 256, 2, 500, 1
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab)).to(device)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# d2l.plt.show()
"""



""" 
# 机器翻译
# 下载和预处理数据集
# Tatoeba项目的“英－法”数据集，数据集中的每一行都是文本序列对
# 序列对由原文序列和翻译后文本序列组成，英语是源语言，法语是目标语
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5') # @save
def read_data_nmt(): # @save
    # 载入“英语－法语”数据集
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
# print(raw_text[:75])

def preprocess_nmt(text): # @save
    # 预处理“英语－法语”数据集
    def no_space(char, prev_char):
        # 检查两个条件：前一个字符 prev_char 不是空格
        # 当前字符 char 是否是标点符号 , . ! ? 中的一个
        # 如果都满足，返回 True，表示需要在这个标点前添加空格
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格，使用小写字母替换大写字母
    # \u202f：替换窄不间断空格为普通空格；\xa0：替换不间断空格为普通空格
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格，enumerate(text) 获取字符的索引 i 和字符本身 char
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out) # 将处理后的字符列表用空字符串连接，重新组合成字符串

text = preprocess_nmt(raw_text)
# print(text[:80])

# tokenize_nmt 函数返回两个词元列表：source 和 target
def tokenize_nmt(text, num_examples=None): # @save
    # 词元化“英语－法语”数据数据集
    # num_examples: 可选参数，限制处理的样本数量
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2: # 检查分割后的部分数量是否为2（确保有源语言和目标语言）
            source.append(parts[0].split(' ')) # 将源语言文本按空格分割成词元列表
            target.append(parts[1].split(' ')) # 将目标语言文本按空格分割成词元列表
    return source, target

source, target = tokenize_nmt(text)
# print(source[:6], target[:6])

# 绘制每个文本序列所包含的词元数量的直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist): # @save
    d2l.set_figsize()
    # 分别计算 xlist 和 ylist 中每个序列的长度；patches: 获取直方图图形对象，用于后续样式设置
    _, _, patches = d2l.plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    # patches[1]: 获取 ylist 的直方图
    for patch in patches[1].patches:
        patch.set_hatch('/') # 为每个柱状图设置斜线填充图案
    d2l.plt.legend(legend)
    # d2l.plt.show()

# show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)

# 构建词表
# reserved_tokens=[...]: 保留的特殊标记；<pad>: 填充；<bos>: 序列开始；<eos>: 序列结束
src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(len(src_vocab)) # 10012

# 加载数据集
# 为了提高计算效率，通过截断和填充实现一次只处理一个小批量的文本序列
def truncate_pad(line, num_steps, padding_token): # @save
    # 截断或填充文本序列
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

def build_array_nmt(lines, vocab, num_steps): # @save
    # 将机器翻译的文本序列转换成小批量
    lines = [vocab[l] for l in lines] # 将文本序列转换为词元ID序列
    lines = [l + [vocab['<eos>']] for l in lines] # 在每个序列末尾添加结束标记 <eos>
    # 对每个序列进行截断或填充，使其长度统一为 num_steps，将结果转换为 PyTorch 张量
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # 创建布尔掩码，True表示非填充位，计算每个序列有效长度(非填充标记数量)
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# 训练模型
def load_data_nmt(batch_size, num_steps, num_examples=600): # @save
    # 返回翻译数据集的迭代器和源语言、目标语言两个词表
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    # 分别构建源语言和目标语言的词表
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 将机器翻译的文本序列转换成小批量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 组合数据并创建数据迭代器
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# 读出“英语－法语”数据集中的第一个小批量数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32), "\n", 'X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32), "\n", 'Y的有效长度:', Y_valid_len)
    break
"""