import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 注：以下文字说明对应的知识点难度较大

# 将每个单词或子词视为单个词元时，可以在大型语料库上使用word2vec、GloVe或子词嵌入模型预先训练每个词元的词元
# 预训练后，每个词元的表示可以是一个向量，其后保持不变。许多较新的预训练模型使相同词元的表示适应于不同的上下文
# 词向量是用于表示单词意义的向量，并且还可以被认为是单词的特征向量或表示。将单词映射到实向量的技术称为词嵌入
# 独热向量通常不是一个好的选择，原因是独热向量不能准确表达不同词之间的相似度，比如我们经常使用的“余弦相似度”

# word2vec工具包含两个模型：跳元模型和连续词袋。对于语义上有意义的表示，训练依赖于条件概率
# 跳元模型假设一个词可以用来在文本序列中生成其周围的单词。更具体地说，对于词典中索引为i的任何词，
# 分别用v_i ∈ R_d和u_i ∈ R_d表示用作中心词和上下文词时的两个向量。给定中心词w_c(词典索引c)，
# 生成任何上下文词w_o(词典索引o)的条件概率可以通过对向量点积的softmax操作来建模：P(wo | wc) (14.1.4)
# 文本序列(长度T)时间步t的词为w_t。对于上下文窗口m，似然函数是给定中心词情况下生成所有上下文词概率 (14.1.5)
# 跳元模型参数是词表中每个词的中心词向量和上下文词向量。通过最大化似然函数(最小化损失函数)学习参数：(14.1.6)
# 为计算随机梯度，要获得(通过微分)对数条件概率关于中心词向量和上下文词向量的梯度 (14.1.8)

# 连续词袋与跳元模型类似，但不同的是连续词袋模型假设中心词是基于其在文本序列中的周围上下文词生成的
# 由于存在多个上下文词，因此在计算条件概率时对这些上下文词向量进行平均。即对于字典中索引i的任意词，
# 用vi ∈ R_d和ui ∈ R_d表示用作上下文词和中心词两个向量。给定上下文词w_oi生成中心词w_c的条件概率 (14.1.10)
# 文本序列(长度T)时间步t的词为w_t。对于上下文窗口m，似然函数是给定上下文词情况下生成所有中心词的概率 (14.1.12)
# 连续词袋模型的最大似然估计等价于最小化损失函数 (14.1.13)，通过 (14.1.11) 式的化简可得到 (14.1.14) 的式子
# 通过微分，可以获得其关于任意上下文词向量v_oi（i = 1, . . . , 2m）的梯度 (14.1.15)


# 为降低上述计算复杂度(在词典上求和梯度的成本)，本节将介绍两种近似训练方法：负采样和分层softmax
# 负采样修改原目标函数。给定中心词w_c的上下文窗口，任意上下文词w_o来自上下文窗口认为是下式概率事件 (14.2.1)
# 文本序列(长度T)时间步t的词为w_t，上下文窗口为m，考虑最大化联合概率(只考虑那些正样本的事件)：(14.2.3)
# 为了使目标函数更有意义，负采样添加从预定义分布中采样的负样本。负采样将 (14.2.3)中的联合概率重写 (14.2.4)
# 通过事件S, N1,..., NK近似条件概率：(14.2.5)，(14.2.5)中关于条件概率的对数损失为：(14.2.6)
# 每个训练步的梯度计算成本与词表大小无关(线性依赖于K)。超参数K设较小值时，负采样的每个训练步处的梯度计算成本较小

# 层序 Softmax 使用二叉树，每个叶节点表示词表V中的一个词。L(w)表示二叉树中字w从根节点到叶节点路径上的节点数
# 设n(w, j)为路径上的j_th节点，其上下文字向量为u_n(w,j)。分层softmax将(14.1.4)中条件概率近似为 (14.2.7)
# 其中函数 σ 在(14.2.2)中定义，leftChild(n) 是节点 n 的左子节点：如果x为真，[[x]] = 1；否则[[x]] = −1
# 二叉树结构，L(wo)−1 与 O(log2|V|) 是一个数量级。词表V很大时，使用分层softmax每个训练步计算代价显著降低

import math
import os
import random
import torch
import collections
from torch import nn
from d2l import torch as d2l

# 读取数据集Penn Tree Bank (用于预训练词嵌入模型)
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
'319d85e578af0cdc590547f26231e4e31cdf1e42') # @save

def read_ptb(): # @save
    # 将PTB数据集加载到文本行的列表中
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb() # 'sentences数: 42069'
# print(f'sentences数: {len(sentences)}')

# 构建词表，其中出现次数少于10次的任何单词都将由“<unk>”词元替换
vocab = d2l.Vocab(sentences, min_freq=10)
print(f'vocab size: {len(vocab)}')

# 实现缺失的 count_corpus 函数
def count_corpus(tokens):
    # 统计词元的频率
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 下采样：当训练词嵌入模型时，对高频单词进行下采样(数据集中每个词w_i将有概率被丢弃)：
# p(w_i) = max(1 - sqrt(t / f(w_i)), 0)。f(w_i)是w_i的词数与数据集中的总词数的比率，t是超参数
# 可看到只有当相对比率 f(w_i) > t 时，高频词w_i才能被丢弃，且该词的相对比率越高，被丢弃的概率就越大
def subsample(sentences, vocab): # @save
    # 下采样高频词，排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = count_corpus(sentences) # 统计每个词的出现频率，计算总词数
    num_tokens = sum(counter.values())
    def keep(token): # 如果在下采样期间保留词元，则返回True
        # counter[token] / num_tokens：词频 f(w_i)
        # 保留概率：p = max(1 - sqrt(t / f(w_i)), 0)
        # 实现为：random.uniform(0, 1) < sqrt(t / f(w_i))
        return(random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))
    # 应用下采样，只保留概率满足条件的词，并返回下采样后的语料和词频统计
    return ([[token for token in line if keep(token)] for line in sentences], counter)

subsampled, counter = subsample(sentences, vocab)
d2l.show_list_len_pair_hist(['origin', 'subsampled'],
'# tokens per sentence','count', sentences, subsampled)

def compare_counts(token):
    return (f'"{token}"的数量：'
    f'之前={sum([l.count(token) for l in sentences])}, '
    f'之后={sum([l.count(token) for l in subsampled])}')
# print(compare_counts('the')) # "the"的数量：之前=50770, 之后=2056
# print(compare_counts('join')) # "join"的数量：之前=45, 之后=45
corpus = [vocab[line] for line in subsampled] # 将词转换为索引
print(corpus[:3]) # [[], [2115, 274, 406], [140, 3, 5277, 3054, 1580]]


# 中心词和上下文词的提取：get_centers_and_contexts函数从corpus中提取所有中心词及其上下文词
# 对于任一中心词，与其距离不超过采样上下文窗口大小(1到max_window_size间的整数)的词为其上下文词
def get_centers_and_contexts(corpus, max_window_size): # @save
    # 返回跳元模型中的中心词和上下文词
    centers, contexts = [], []
    # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
    for line in corpus:
        # 跳过长度小于2句子
        if len(line) < 2:
            continue
        centers += line # 每个词都作为中心词
        for i in range(len(line)): # 上下文窗口中间i
            # 遍历句子中每个位置，随机选择窗口大小(1~max_window_size)
            window_size = random.randint(1, max_window_size)
            # 计算上下文词索引范围：左边界max(0,i-window_size)；右边界min(len(line),i+1+window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            indices.remove(i) # 从上下文词中排除中心词
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# 创建人工数据集
tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('数据集', tiny_dataset) # 数据集 [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心词', center, '的上下文词是', context) # 模拟提取上下文词的过程

all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
print(f'中心词-上下文词对数量: {sum([len(contexts) for contexts in all_contexts])}')


# 负采样：为根据预定义分布对噪声词进行采样，定义RandomGenerator类
class RandomGenerator: # @save
    # 根据n个采样权重在{1,...,n}中随机抽取
    def __init__(self, sampling_weights): # 功能：高效的加权随机采样器
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights # 每个元素的采样权重
        self.candidates = [] # 缓存的采样结果
        self.i = 0 # i：当前缓存位置

    def draw(self): # 高效采样策略
        # 当缓存用尽时，一次性生成10000个样本，避免频繁调用随机函数，提高效率
        if self.i == len(self.candidates): # 缓存k个随机采样结果
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1] # 逐个返回缓存中的样本

# @save
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]

def get_negatives(all_contexts, vocab, counter, K): # @save
    # 返回负采样中的噪声词，索引为1、2、...（索引0是词表中排除的未知标记）
    # 关键公式：权重 = 词频^0.75；counter[vocab.to_tokens(i)]：将索引转换为词，获取词频
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights) # 初始化结果列表和随机生成器
    for contexts in all_contexts: # 对每个上下文集合生成负样本
        negatives = [] # 需要 len(contexts) * K 个负样本
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# 执行负采样，K=5（每个上下文词对应5个负样本）
all_negatives = get_negatives(all_contexts, vocab, counter, 5)


# 小批量加载训练实例
# 小批量中，i_th个样本包括中心词及其ni个上下文词和mi个噪声词。由于上下文窗口大小不同，ni + mi对于不同的i是不同的
# 对于每个样本，在contexts_negatives个变量中将上下文词和噪声词连结起来并填充零，直到长为maxi ni + mi(max_len)
def batchify(data): # @save
    # 返回带有负采样的跳元模型的小批量样本
    # 找到批次中 上下文词数 + 负样本数 的最大值，用于后续的零填充
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        # 遍历批次中的每个样本，收集中心词
        cur_len = len(context) + len(negative)
        centers += [center]
        # 连接上下文词和负样本，用0填充到最大长度
        contexts_negatives +=         [context + negative + [0] * (max_len - cur_len)]
        # 掩码矩阵：有效位标为1，填充位标为0，损失计算时忽略填充位置
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        # 标签矩阵：上下文词标记为1（正样本），负样本和填充位置标记为0
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
           contexts_negatives), torch.tensor(masks), torch.tensor(labels))

x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))
names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)

# 整合代码
def load_data_ptb(batch_size, max_window_size, num_noise_words): # @save
    # 下载PTB数据集，然后将其加载到内存中
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index): # 按索引返回单个样本
            return (self.centers[index], self.contexts[index], self.negatives[index])

        def __len__(self): # 返回数据集大小
            return len(self.centers)

    # 创建数据加载器；collate_fn=batchify：使用自定义的批次处理函数
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size,
        shuffle=True, collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab

data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
        break
