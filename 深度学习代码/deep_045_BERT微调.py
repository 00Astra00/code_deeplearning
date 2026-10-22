import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import multiprocessing
import os
import torch
import inspect
from torch import nn
from d2l import torch as d2l


# 序列级和词元级应用微调BERT
# 在序列层次，介绍单文本分类任务和文本对分类/回归任务，将文本输入BERT表示转为输出标签
# 在词元级别，介绍新应用(文本标注和问答)，并说明BERT如何表示它们的输入并转换为输出标签
# 微调期间，不同应用间BERT所需“最小架构更改”是额外全连接层。在下游应用的监督学习期间，
# 额外层的参数是从零开始学习的，而预训练BERT模型中的所有参数都是微调的

# 单文本分类定义：单个文本序列作为输入，输出分类结果，语言可接受性语料库(数据集)，判断句子是否符合语法
# BERT输入序列表示单个文本，特殊分类标记“<cls>”用于序列分类，“<sep>”标记单个文本结束或分隔成对文本
# 单文本分类中，特殊分类标记“<cls>”的BERT表示对整个输入文本序列的信息进行编码作为输入单个文本的表示，
# 它被送入到由全连接（稠密）层组成的小多层感知机中，以输出所有离散标签值的分布

# 文本对分类或回归定义：一对文本作为输入，输出连续值，语义文本相似度是“文本对回归”任务，评估句子语义相似度
# 语义文本相似度基准数据集中，句子对相似度得分是从0（无语义重叠）到5（语义等价）的分数区间，目标是预测分数
# 文本对分类的BERT微调在输入表示上有所不同。对于文本对回归任务，可应用细微更改，输出连续标签值和使用均方损失

# 词元级别任务语义标注：每个词元都分配一个标签。在文本标注任务中，词性标注为每个单词分配词性标记(如形容词)
# 唯一的区别在于，在文本标注中，输入文本的每个词元的BERT表示被送到相同的额外全连接层中，以输出词元的标签

# 词元级别任务问答：反映阅读理解能力。斯坦福问答数据集由阅读段落和问题组成，其中每个问题答案是段落中的一段文本
# BERT输入中，问题和段落作为第一和第二个文本序列。为预测文本片段开始位置，相同的额外全连接层把位置i的词元BERT
# 表示转成标量分数s_i。文中词元分数通过softmax转成概率分布，为文中每个词元位置i分配文本片段开始概率p_i(预测
# 文本片段结束相同，额外全连接层参数与预测开始位置参数无关)。预测结束时，位置i词元由相同全连接层变成标量分数e_i
# 问答训练目标：最大化真实值开始和结束位置对数似然。预测片段时计算位置i到j有效片段分数s_i+e_j，输出分数最高跨度

# 代码有误，无法运行

# 自然语言推断：微调BERT
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip', '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip', 'c72329e68a732bef0452e4b96a1c341c8910f81f')

def load_pretrained_model(pretrained_model, num_hiddens,
    ffn_num_hiddens, num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    vocab = d2l.Vocab() # 定义空词表以加载预定义词表
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256], ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
    num_heads=4, num_layers=2, dropout=0.2, max_len=max_len, key_size=256, query_size=256, value_size=256,
    hid_in_features=256, mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab

devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model('bert.small', num_hiddens=256,
ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)

# 定制数据集类SNLIBERTDataset。在每个样本中，前提和假设形成一对文本序列，并打包成BERT输入序列片段索引区分
# 输入序列前提和假设。利用预定义BERT输入序列最大长度，持续移除输入文本对中较长文本最后标记，直到满足max_len
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        # for sentences in dataset[:2]：处理前提和假设
        # 最终得到：[[p_tokens1, h_tokens1], [p_tokens2, h_tokens2], ...]
        all_premise_hypothesis_tokens = [[p_tokens, h_tokens] for p_tokens, h_tokens in zip(
              *[d2l.tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]])]
        self.labels, self.vocab, self.max_len = torch.tensor(dataset[2]), vocab, max_len
        (self.all_token_ids, self.all_segments,self.valid_lens)         = (self._preprocess(all_premise_hypothesis_tokens)) # 调用并行预处理方法
        print('read ' + str(len(self.all_token_ids)) + ' examples')
    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4) # 使用4个进程并行处理
        # pool.map：将数据分发给工作进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), torch.tensor(valid_lens))
    def _mp_worker(self, premise_hypothesis_tokens):
        # 功能：多进程工作函数，处理单个前提-假设对
        # premise_hypothesis_tokens 包含前提和假设词元列表的元组
        p_tokens, h_tokens = premise_hypothesis_tokens
        # 调用截断方法：确保前提和假设的总长度不超过BERT最大长度限制
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        # 词索引转换和填充：self.vocab[tokens]：将词元列表转换为索引列表
        # [self.vocab['<pad>']] * (self.max_len - len(tokens))：创建填充部分
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments, valid_len = segments + [0] * (self.max_len - len(segments)), len(tokens)
        return token_ids, segments, valid_len
    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 功能：智能截断前提和假设，确保总长度不超过限制
        # "-3" 是为BERT输入中的'<CLS>'、和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            # 平衡截断策略：有助于保持两个序列的信息平衡
            # 比较前提和假设的长度，从较长序列移除最后一个词元
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx]), self.labels[idx]
    def __len__(self):
        return len(self.all_token_ids)

batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)

# 自然语言推断微调BERT只需额外多层感知机将特殊“<cls>”词元BERT表示进行转换，
# 该词元同时编码前提和假设的信息为自然语言推断的三个输出：蕴涵、矛盾和中性
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        # 新增线性分类层：唯一从头训练的组件
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        # BERT编码：输出形状：(batch_size, seq_len, hidden_size)
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        # encoded_X[:, 0, :]：提取每个序列第一个位置([CLS])
        # [CLS]标记在BERT预训练中专门设计用于汇总整个序列信息
        # self.output()：通过新增分类层，输出形状(batch_size, 3)
        return self.output(self.hidden(encoded_X[:, 0, :]))

net = BERTClassifier(bert)
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)