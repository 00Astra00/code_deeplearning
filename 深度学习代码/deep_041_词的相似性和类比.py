import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import torch
from torch import nn
from d2l import torch as d2l


d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
'0b8703943ccdb6eb788e6f091b8946e82231bc4d') # @save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a') # @save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
'b5116e234e9eb9076672cfeabf5469f3eec904fa') # @save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
'c1816da3821ae9f43899be655002f6c723e91b88') # @save

class TokenEmbedding: # @save
    # GloVe嵌入
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0 # 未知词的索引（对应'<unk>'）
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                # 每行格式：word v1 v2 ... vn（词 + n维向量）
                # elems[0]：词;elems[1:]：词向量，转换为浮点数
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        # 重载 [] 操作符，支持批量获取词向量；输入：词列表，输出：对应的词向量张量
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

glove_6b50d = TokenEmbedding('glove.6b.50d') # len(glove_6b50d)) -> 400001
print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])

# 词相似：根据词向量间的余弦相似性为输入词查找语义相似的词
def knn(W, x, k):
    # 余弦相似度计算：增加1e-9获得数值稳定性，防止除零错误
    # torch.mv(W, x)：矩阵-向量乘法，计算所有词与x的点积
    # torch.sum(W * W, axis=1)：每个词向量的L2范数平方
    cos = torch.mv(W, x.reshape(-1,)) / (torch.sqrt(torch.sum(W * W, axis=1)
          + 1e-9) * torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k) # torch.topk：返回最大k个值和对应的索引
    return topk, [cos[int(i)] for i in topk] # topk索引和对应的余弦相似度

# 使用TokenEmbedding的实例embed中预训练好的词向量来搜索相似的词
def get_similar_tokens(query_token, k, embed):
    # 查找与查询词最相似的k个词；topk[1:]：跳过第一个（查询词自身）
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]): # 排除输入词
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')

print(get_similar_tokens('chip', 3, glove_6b50d))
print(get_similar_tokens('baby', 3, glove_6b50d))
print(get_similar_tokens('beautiful', 3, glove_6b50d))

# 词类比：单词类比a : b :: c : d，给出前三个词a、b和c，找到d。vec(w)表示词w向量，
# 为完成这个类比，我们将找到一个词，其向量与vec(c) + vec(b) − vec(a)的结果最相似

def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])] # 删除未知词

print(get_analogy('man', 'woman', 'son', glove_6b50d))
print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
print(get_analogy('bad', 'worst', 'big', glove_6b50d))
print(get_analogy('do', 'did', 'go', glove_6b50d))
