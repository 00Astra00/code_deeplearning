import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import random
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


""" 
# 上下文敏感：词元x的上下文敏感是函数f(x, c(x))，其取决于x及其上下文c(x)
# 尽管ELMo显著改进了各种自然语言处理任务的解决方案，但每个解决方案仍然依赖于一个特定于任务的架构
# GPT模型为上下文敏感表示设计通用任务无关模型(由于语言模型自回归特性，GPT只能向前看（从左到右）)
# BERT（来自Transformers的双向编码器表示）结合了这两个方面的优点。它对上下文进行双向编码，并且对
# 大多自然语言处理任务只需最少架构改变。通过预训练Transformer编码器，基于其双向上下文表示任何词元
# BERT表示输入到添加输出层中，根据任务性质更改架构；预训练T....编码器参数微调，额外输出层从头训练

# BERT输入序列明确地表示单个文本和文本对：
# 当输入为单个文本时，BERT输入序列是特殊类别词元“<cls>”、文本序列的标记、以及特殊分隔词元“<sep>”的连结
# 当输入为文本对时，输入序列是“<cls>”、第一个文本序列标记、“<sep>”、第二个文本序列标记、及“<sep>”连结

# 将一个或两个句子作为输入，然后返回BERT输入序列的标记及其相应的片段索引
def get_tokens_and_segments(tokens_a, tokens_b=None): # @save
    # 获取输入序列的词元及其片段索引
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


# 确保正确使用 EncoderBlock 类
class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values的形状: (batch_size, 查询或者"键－值"对的个数, num_hiddens)
        # valid_lens的形状: (batch_size,) 或 (batch_size, 查询的个数)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    # 输入X的形状:(batch_size，查询或者"键－值"对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者"键－值"对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者"键－值"对的个数，num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads, 查询或者"键－值"对的个数，num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    # 逆转transpose_qkv函数的操作
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


# BERT输入序列嵌入是词元嵌入、片段嵌入和位置嵌入和
# 以下BERTEncoder类使用片段嵌入和可学习的位置嵌入
class BERTEncoder(nn.Module): # @save
    # BERT编码器
    def __init__(self, vocab_size, num_hiddens, norm_shape,
        ffn_num_input,ffn_num_hiddens, num_heads, num_layers, dropout,
        max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        # ffn_num_input, ffn_num_hiddens: 前馈网络的输入和隐藏层维度
        # num_heads: 注意力头数；num_layers: Transformer层数
        super(BERTEncoder, self).__init__(**kwargs)
        # 创建词元嵌入层，将词元ID映射为向量；创建片段嵌入层，区分两个句子（0和1）
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            # 循环创建多个Transformer编码器层，每层都是EncoderBlock，包含多头注意力和前馈网络
            self.blks.add_module(f"{i}", EncoderBlock(key_size, query_size, value_size,
            num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 创建可学习的位置嵌入参数，形状为(1, max_len, num_hiddens)，可以广播到批量大小
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # tokens: 词元ID，形状(batch_size, seq_len)
        # segments: 片段ID，形状(batch_size, seq_len)
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # 加上位置嵌入,[:, :X.shape[1], :] 只取前seq_len个位置
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)

# 使用输入tokens的BERTEncoder的前向推断返回编码结果，其中每个词元由向量表示，
# 其长度由超参数num_hiddens定义，称为Transformer编码器的隐藏大小（隐藏单元数）
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print(encoded_X.shape) # torch.Size(2, 8, 768)

# 预训练包括以下两个任务：掩蔽语言模型和下一句预测

# 1.掩蔽语言模型：为双向编码上下文表示每个词元，BERT随机掩蔽词元并使用来自双向上下文词元以自监督方式预测掩蔽词元
# 预测掩蔽词元而不使用标签作弊(用特殊“<mask>”替换输入序列中词元)。为避免预训练和微调不匹配，如果为预测而屏蔽词元，
# 则在输入中将其替换为：80%时间为特殊的“<mask>“词；10%时间为随机词元；10%时间内为不变的标签词元
class MaskLM(nn.Module): # @save
    # BERT的掩蔽语言模型任务
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                   nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1] # 获取每个样本预测位置数量
        pred_positions = pred_positions.reshape(-1) # 将预测位置展平为一维张量
        batch_size = X.shape[0] # 创建批次索引
        batch_idx = torch.arange(0, batch_size)
        # 重复批次索引，为每个预测位置创建对应的批次索引
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions] # 提取被掩蔽位置的表示
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X) # 通过MLP预测被掩蔽的词元
        return mlm_Y_hat

# mlm_positions定义为在encoded_X的任一输入序列中预测的3个指示
# mlm前向推断返回encoded_X所有掩蔽位置mlm_positions处预测结果
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
print(mlm_Y_hat.shape) # torch.Size([2, 3, 10000])

mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
print(mlm_l.shape) # torch.Size([6])

# 为帮助理解两个文本序列之间的关系，BERT在预训练中考虑二元分类任务——下一句预测。在为预训练生成句子对时，
# 有一半的时间它们确实是标签为“真”的连续句子；在另一半的时间里，第二个句子从语料库中随机抽取，标记为“假”
# NextSentencePred类使用单隐藏层的多层感知机来预测第二个句子是否是BERT输入序列中第一个句子的下一个句子
class NextSentencePred(nn.Module): # @save
    # BERT的下一句预测任务
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batch_size,num_hiddens)
        return self.output(X)

encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batch_size，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
print(nsp_Y_hat.shape) # torch.Size([2, 2])
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
print(nsp_l.shape) # torch.Size([2])

# 整合代码
# 预训练BERT时，最终损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合
class BERTModel(nn.Module): # @save
    # BERT模型
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
        ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000,
        key_size=768, query_size=768, value_size=768, hid_in_features=768,
        mlm_in_features=768, nsp_in_features=768):
        # hid_in_features, mlm_in_features, nsp_in_features: 各任务的输入维度
        super(BERTModel, self).__init__()
        # BERT编码器：处理输入序列，生成上下文相关的词元表示
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
        ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout,
        max_len=max_len, key_size=key_size, query_size=query_size, value_size=value_size)
        # 隐藏层变换：将[CLS]标记的表示转换为适合NSP任务的维度
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        # MLM任务：掩蔽语言模型；NSP任务：下一句预测
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        # 只在训练时提供pred_positions（被掩蔽的位置）
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else: # 推理时mlm_Y_hat为None
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器隐藏层，0是“<cls>”标记的索引
        # encoded_X[:, 0, :]: 提取每个序列的第一个位置（[CLS]标记）
        # 形状(batch_size, num_hiddens)，隐藏层变换后输入NSP分类器
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        # 返回：编码后的序列表示；MLM预测结果（训练时）；NSP预测结果
        return encoded_X, mlm_Y_hat, nsp_Y_hat
"""



""" 
# 预训练BERT的数据集
# 无需知道细节，只需知道有即可
d2l.DATA_HUB['wikitext-2'] = ( # @save
'https://s3.amazonaws.com/research.metamind.io/wikitext/'
'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir): # @save
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # 大写字母转换为小写字母
        paragraphs = [line.strip().lower().split(' . ')
        for line in lines if len(line.split(' . ')) >= 2]
        random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs): # @save
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len): # @save
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
        paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab): # @save
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
             masked_token = '<mask>'
        else:
             # 10%的时间：保持词不变
             if random.random() < 0.5:
                 masked_token = tokens[mlm_pred_position]
             # 10%的时间：用随机词替换该词
             else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab): # @save
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
    candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
    tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples, max_len, vocab): # @save
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
        max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) +
        [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
        max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset): # @save
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
            # 获取遮蔽语言模型任务的数据
            examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) +
            (segments, is_next)) for tokens, segments, is_next in examples]
            (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights,
             self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab) # 填充输入
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
        self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx], self.nsp_labels[idx])
    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len): # @save
    # 加载WikiText-2数据集
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
    pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape, nsp_y.shape)
    break
print(len(vocab)) # 20256




# 预训练BERT
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128], ffn_num_input=128,
ffn_num_hiddens=256, num_heads=2, num_layers=2, dropout=0.2, key_size=128, query_size=128,
value_size=128, hid_in_features=128, mlm_in_features=128, nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X,
valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y): # @save
    # 辅助函数：给定训练样本，计算遮蔽语言模型和下一句子预测任务的损失
    # 模型前向传播：忽略返回的encoded_X（编码表示）；获取mlm_Y_hat: MLM预测结果 和 nsp_Y_hat: NSP预测结果
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X)
    # MLM损失计算：mlm_Y_hat.reshape(...): 预测结果重塑为(num_predictions, vocab_size)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1) # 权重矩阵，只有被掩蔽的位置权重为1，其他为0
    # MLM损失归一化：对加权损失求和；除以被掩蔽位置的数量；+ 1e-8防止除零
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    nsp_l = loss(nsp_Y_hat, nsp_y) # 计算下一句子预测任务的损失
    l = mlm_l + nsp_l # 两个预训练任务的总损失
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm', 'nsp'])
    metric = d2l.Accumulator(4) # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    num_steps_reached = False # 终止标志：控制训练循环
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            # 调用之前定义的损失计算函数，返回：mlm_l: MLM损失；nsp_l: NSP损失；l: 总损失
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
            segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            # 累加MLM损失，NSP损失，处理的样本数（tokens_X.shape[0]）
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            # 绘制当前平均MLM损失(metric[0]/metric[3])和 NSP损失(metric[1]/metric[3])
            animator.add(step + 1,(metric[0] / metric[3], metric[1] / metric[3]))
            step += 1 # 步数增加
            if step == num_steps:
                num_steps_reached = True
                break
    print(f'MLM loss {metric[0] / metric[3]:.3f}, 'f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on 'f'{str(devices)}')

train_bert(train_iter, net, loss, len(vocab), devices, 50)

def get_bert_encoding(net, tokens_a, tokens_b=None):
    # 返回tokens_a和tokens_b中所有词元的BERT表示
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :] # [CLS]标记的表示（用于分类任务）
encoded_text_crane = encoded_text[:, 2, :] # "crane"词元的表示（位置2）
print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])

tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just', 'left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :] # 双句对的[CLS]表示
encoded_pair_crane = encoded_pair[:, 2, :] # 第一个句子中的"crane"词元表示
print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])
""" 