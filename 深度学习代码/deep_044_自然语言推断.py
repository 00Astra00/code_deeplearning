import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import re
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


# 自然语言推断：主要研究假设是否可以从前提中推断出来，其中两者都是文本序列
# 即自然语言推断决定了一对文本序列之间的逻辑关系。这类关系通常分为三种类型：
# 蕴涵：假设可以从前提中推断出来；矛盾：假设的否定可以从前提中推断出来；中性：所有其他情况

d2l.DATA_HUB['SNLI'] = ( # @save
'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
'9fcde07509c7e87ec61c640c1b2753d9041758e4')
# data_dir = d2l.download_extract('SNLI')

def read_snli(data_dir, is_train): # @save
    # 将SNLI数据集解析为前提、假设和标签
    def extract_text(s):
        # 删除不会使用的信息
        s = re.sub('\(', '', s) # 删除左括号
        s = re.sub('\)', '', s) # 删除右括号
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\s{2,}', ' ', s)
        return s.strip() # 去除首尾空白字符
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
    if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    # 数据提取逻辑：row[0]：标签；row[1]：前提；row[2]：假设
    # if row[0] in label_set：只保留有效标签的数据
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

# train_data = read_snli(data_dir, is_train=True)
# test_data = read_snli(data_dir, is_train=False)
# for data in [train_data, test_data]:
#     # 遍历所有标签，统计每个类别（0,1,2）的出现次数
#     print([[row for row in data[2]].count(i) for i in range(3)])
# train_data:[183416, 183187, 182764]；test_data:[3368, 3237, 3219]

class SNLIDataset(torch.utils.data.Dataset): # @save
    # 用于加载SNLI数据集的自定义数据集
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        # 保存序列长度参数，对前提和假设文本进行分词
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            # 如果没有提供词汇表：基于所有前提和假设的词元构建新词汇表
            self.vocab = d2l.Vocab(all_premise_tokens +             all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else: # 如果提供词汇表：使用现有词汇表
            self.vocab = vocab
        # 对前提和假设进行填充处理
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')
    def _pad(self, lines):
        # self.vocab[line]：将词元列表转换为ID列表
        return torch.tensor([d2l.truncate_pad(self.vocab[line],
        self.num_steps, self.vocab['<pad>']) for line in lines])
    def __getitem__(self, idx):
        # 获取单个样本：输入：(前提ID序列, 假设ID序列)；输出：对应的标签
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
    def __len__(self):
        return len(self.premises)

def load_data_snli(batch_size, num_steps=50): # @save
    # 下载SNLI数据集并返回数据迭代器和词表
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    # train_set：训练数据创建数据集；test_set：测试数据创建数据集，共享训练集词汇表
    # 这确保了训练和测试使用相同的词到ID映射，避免测试集中出现训练集未见的词导致的问题
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab

# 基于注意力的自然语言推断：三个联合训练的步骤组成：对齐、比较和汇总
# 1.注意：第一步是将一个文本序列中的词元与另一个序列中的每个词元对齐
# A(a1,...,am)和B(b1,..., bn)表示前提和假设，其中ai,bj是词向量
# 对于软对齐，注意力权重e_ij ∈ R计算为：e_ij = f(ai).T * f(bj)
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

# 将注意力权重规范化，计算假设中所有词元向量的加权平均值，
# 将假设与前提中索引i(j同理)词元软对齐(15.5.2, 15.5.3)
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B):
        # A/B的形状：(批量大小，序列A/B的词元数，embed_size)
        # f_A/B的形状：(批量大小，序列A/B的词元数，num_hiddens)
        f_A, f_B = self.f(A), self.f(B)
        # e的形状：(批量大小，序列A的词元数，序列B的词元数)
        # 结果e[i, j, k]表示前提第i个词元与假设第k个词元相似度
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # (β计算)对每个前提词元，在假设序列维度上做softmax
        # 每行表示一个前提词元对所有假设词元的注意力分布
        # torch.bmm()：用注意力权重对假设词向量加权求和
        # 结果beta意义：每个前提词元对应的软对齐假设表示
        # beta的形状(批量大小，序列A的词元数，embed_size)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # (α计算)对每个假设词元，在前提序列维度上做softmax
        # torch.bmm()：用注意力权重对前提词向量加权求和
        # 结果alpha意义：每个假设词元对应的软对齐前提表示
        # alpha的形状(批量大小，序列B的词元数，embed_size)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        # β：假设软对齐到前提的结果；α：前提软对齐到假设的结果
        return beta, alpha

# 2.比较：将一个序列中的词元与与该词元软对齐的另一个序列进行比较
# 比较步骤中，将来自一个序列的词元的连结和来自另一序列的对齐的词元送入函数g(MLP)：
# v_A,i = g([ai, βi]), i = 1,...,m，所有假设中词元与前提中词元i软对齐，再与词元i的比较
# v_B,j = g([bj, αj]), j = 1,...,n，所有前提中词元与假设中词元j软对齐，再与词元j的比较
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B, beta, alpha):
        # 前提比较 (V_A)：torch.cat([A, beta], dim=2)：将原始前提词向量A与软对齐假设表示β连接
        # 假设比较 (V_B)：torch.cat([B, alpha], dim=2)：将原始假设词向量B与软对齐前提表示α连接
        V_A = self.g(torch.cat([A, beta], dim=2)) # 输出：每个前提词元的比较表示
        V_B = self.g(torch.cat([B, alpha], dim=2)) # 输出：每个假设词元的比较表示
        return V_A, V_B

# 3.聚合：聚合比较信息以推断逻辑关系。首先求和这两组比较向量，接下来，
# 将两个求和结果连结提供给函数h(MLP)，获得逻辑关系分类结果：y^ = h([vA, vB])
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)
    def forward(self, V_A, V_B):
        # 序列级聚合：结果：每个序列的全局表示向量
        # V_A.sum(dim=1)：对前提的所有比较向量沿序列维度求和
        # V_B.sum(dim=1)：对假设的所有比较向量沿序列维度求和
        V_A, V_B = V_A.sum(dim=1), V_B.sum(dim=1)
        # 将两个求和结果的连结送到多层感知机中
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat

# 整合代码
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises) # premises：前提词索引序列
        B = self.embedding(hypotheses) # hypotheses：假设词索引序列
        # beta：假设软对齐到前提(batch_size, seq_len_A, embed_size)
        # alpha：前提软对齐到假设(batch_size, seq_len_B, embed_size)
        beta, alpha = self.attend(A, B) # 注意力对齐
        # V_A：前提比较向量(batch_size, seq_len_A, num_hiddens)
        # V_B：假设比较向量(batch_size, seq_len_B, num_hiddens)
        V_A, V_B = self.compare(A, B, beta, alpha) # 比较表示
        Y_hat = self.aggregate(V_A, V_B) # 分类输出三个类别logits
        return Y_hat

# 训练和评估模型
batch_size, num_steps, lr, num_epochs = 256, 50, 0.001, 4
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

def predict_snli(net, vocab, premise, hypothesis): # @save
    # 预测前提和假设之间的逻辑关系
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    # 输入：(premises, hypotheses) 元组；输出：形状 (1, 3) 的logits
    label = torch.argmax(net([premise.reshape((1, -1)), hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'

print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.']))



""" 
# 习题3
# 标注标准：
# 0.0-0.2: 完全不相关
# 0.2-0.4: 弱相关
# 0.4-0.6: 部分相关
# 0.6-0.8: 强相关
# 0.8-1.0: 完全等价

import torch
import torch.nn as nn
import torch.nn.functional as F
class SemanticSimilarityModel(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, num_heads=8, dropout=0.1):
        super(SemanticSimilarityModel, self).__init__()
        # 词嵌入层（可使用预训练词向量）
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 双向编码器（捕捉上下文信息）
        self.encoder = nn.LSTM(embed_size, hidden_size // 2, batch_first=True, bidirectional=True)
        # 多粒度注意力机制
        self.word_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.phrase_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        # 交互匹配层
        self.interaction_layer = InteractionModule(hidden_size)
        # 相似度预测层
        self.similarity_predictor = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间的相似度分数
        )
    def forward(self, sentence1, sentence2):
        # 词嵌入
        emb1 = self.embedding(sentence1)  # (batch, len1, embed_size)
        emb2 = self.embedding(sentence2)  # (batch, len2, embed_size)
        # 上下文编码
        encoded1, _ = self.encoder(emb1)  # (batch, len1, hidden_size)
        encoded2, _ = self.encoder(emb2)  # (batch, len2, hidden_size)
        # 词级别注意力交互
        word_sim_features = self.word_level_interaction(encoded1, encoded2)
        # 句子级别表示
        sent1_rep = self.get_sentence_representation(encoded1)
        sent2_rep = self.get_sentence_representation(encoded2)
        # 特征融合
        combined_features = torch.cat([
            sent1_rep, sent2_rep, word_sim_features,
            torch.abs(sent1_rep - sent2_rep),  # 差异特征
            sent1_rep * sent2_rep  # 交互特征
        ], dim=1)
        # 相似度预测
        similarity_score = self.similarity_predictor(combined_features)
        return similarity_score.squeeze()

    def word_level_interaction(self, encoded1, encoded2):
        # 词级别注意力交互
        # 交叉注意力
        attn_weights = torch.bmm(encoded1, encoded2.transpose(1, 2))  # (batch, len1, len2)
        attn_weights = F.softmax(attn_weights, dim=-1)
        # 注意力加权表示
        attended_2_to_1 = torch.bmm(attn_weights, encoded2)  # (batch, len1, hidden)
        attended_1_to_2 = torch.bmm(attn_weights.transpose(1, 2), encoded1)  # (batch, len2, hidden)
        # 交互特征计算
        interaction_1 = torch.cat([encoded1, attended_2_to_1,
        encoded1 - attended_2_to_1, encoded1 * attended_2_to_1], dim=2)
        interaction_2 = torch.cat([encoded2, attended_1_to_2,
        encoded2 - attended_1_to_2, encoded2 * attended_1_to_2], dim=2)
        # 池化得到句子级交互特征
        max_pool_1 = F.adaptive_max_pool1d(interaction_1.transpose(1, 2), 1).squeeze()
        max_pool_2 = F.adaptive_max_pool1d(interaction_2.transpose(1, 2), 1).squeeze()
        return torch.cat([max_pool_1, max_pool_2], dim=1)
    def get_sentence_representation(self, encoded):
        # 获取句子级别表示
        # 多头自注意力
        self_attn = self.word_attention(encoded, encoded, encoded)
        # 多层池化
        max_pool = F.adaptive_max_pool1d(encoded.transpose(1, 2), 1).squeeze()
        avg_pool = F.adaptive_avg_pool1d(encoded.transpose(1, 2), 1).squeeze()
        self_attn_pool = F.adaptive_max_pool1d(self_attn.transpose(1, 2), 1).squeeze()
        return torch.cat([max_pool, avg_pool, self_attn_pool], dim=1)

class MultiHeadAttention(nn.Module):
    # 简化版多头注意力机制
    def __init__(self, hidden_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
    def forward(self, query, key, value):
        batch_size = query.size(0)
        # 线性变换
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        # 多头分割
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 加权求和
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size)
        return self.out_linear(context)

# 使用示例
def train_similarity_model():
    model = SemanticSimilarityModel(vocab_size=10000)
    criterion = nn.MSELoss()  # 回归任务使用均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (sent1, sent2, true_similarity) in enumerate(train_loader):
            pred_similarity = model(sent1, sent2)
            loss = criterion(pred_similarity, true_similarity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""
