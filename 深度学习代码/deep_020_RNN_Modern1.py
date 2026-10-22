import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from d2l import torch as d2l


""" 
# 门控循环单元(GRU)
# 门控循环单元：有专门的机制确定何时更新隐状态和重置隐状态
# 重置门：R_t = σ(X_t * W_xr + H_t−1 * W_hr + b_r)
# 更新门：Z_t = σ(X_t * W_xz + H_t−1 * W_hz + b_z)
# 候选隐状态：H_t' = tanh(X_t * W_xh + (R_t ⊙ H_t−1) W_hh + b_h)
# 最终隐状态：H_t = Z_t ⊙ H_t−1 + (1 − Z_t) ⊙ H_t'
# 重置门有助于捕获序列中的短期依赖关系；更新门有助于捕获序列中的长期依赖关系

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    W_xz, W_hz, b_z = three() # 更新门参数
    W_xr, W_hr, b_r = three() # 重置门参数
    W_xh, W_hh, b_h = three() # 候选隐状态参数
    W_hq = normal((num_hiddens, num_outputs)) # 输出层权重
    b_q = torch.zeros(num_outputs, device=device) # 输出层偏置
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params: # 附加梯度
        param.requires_grad_(True)
    return params

# 初始化参数权重
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 门控循环单元模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: # 遍历输入序列的每个时间步
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

vocab_size, num_hiddens, device, num_epochs, lr = len(vocab), 256, d2l.try_gpu(), 500, 1
# model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简洁实现 GRU
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab)).to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
"""



""" 
# 长短期记忆网络 LSTM
# 输出条目(输出门)；数据读入(输入门)；重置单元(遗忘门)
# 输入门：I_t = σ(X_t * W_xi + H_t−1 * W_hi + b_i)
# 遗忘门：F_t = σ(X_t * W_xf + H_t−1 * W_hf + b_f)
# 输出门：O_t = σ(X_t * W_xo + H_t−1 * W_ho + b_o)
# 候选记忆元：C_t‘ = tanh(X_t * W_xc + H_t−1 * W_hc + b_c)
# 最终记忆元：C_t = F_t ⊙ C_t−1 + I_t ⊙ C_t'
# 隐状态：H_t = O_t ⊙ tanh(C_t)

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    W_xi, W_hi, b_i = three() # 输入门参数
    W_xf, W_hf, b_f = three() # 遗忘门参数
    W_xo, W_ho, b_o = three() # 输出门参数
    W_xc, W_hc, b_c = three() # 候选记忆元参数
    W_hq = normal((num_hiddens, num_outputs)) # 输出层权重
    b_q = torch.zeros(num_outputs, device=device) # 输出层偏置
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params: # 附加梯度
        param.requires_grad_(True)
    return params

# 初始化隐藏层参数
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

# 定义模型
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state # (hidden_state, cell_state)： hidden_state: 短期记忆； cell_state: 长期记忆
    outputs = []
    for X in inputs: # 遍历输入序列的每个时间步
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# 训练和预测
vocab_size, num_hiddens, device, num_epochs, lr = len(vocab), 256, d2l.try_gpu(), 500, 1
# model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简洁实现
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab)).to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
"""



""" 
# 习题2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
from tqdm import tqdm

class WordLevelDataset(Dataset):
    # 单词级文本数据集
    def __init__(self, text_sequences, target_sequences):
        self.sequences = text_sequences
        self.targets = target_sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class WordLevelLSTM(nn.Module):
    # 单词级LSTM语言模型
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super(WordLevelLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 单词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD token的索引
        )
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # (batch, seq_len, features)
        )
        # 输出层
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 权重初始化
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, state=None):
        # x形状: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        # 嵌入查找
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        # LSTM处理
        lstm_out, state = self.lstm(embeddings, state)  # (batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        # 输出预测
        output = self.output(lstm_out)  # (batch_size, seq_len, vocab_size)
        output = output.reshape(batch_size * seq_len, -1)  # 展平用于损失计算
        return output, state

    def init_state(self, batch_size, device):
        # 初始化LSTM状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class WordLevelProcessor:
    # 单词级文本处理器
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = None
        self.reverse_vocab = None

    def preprocess_text(self, text):
        # 文本预处理
        text = text.lower() # 转换为小写
        # 简单的标点符号处理
        text = re.sub(r'([.!?])', r' \1', text)  # 在标点前加空格
        text = re.sub(r'[^a-zA-Z.!?]+', ' ', text)  # 移除非字母字符
        return text

    def build_vocab(self, texts):
        # 构建词汇表
        all_words = []
        for text in texts:
            words = self.preprocess_text(text).split()
            all_words.extend(words)
        word_counts = Counter(all_words) # 统计词频
        # 构建词汇表
        vocab = {
            '<PAD>': 0,  # 填充标记
            '<UNK>': 1,  # 未知词标记
            '<SOS>': 2,  # 序列开始标记
            '<EOS>': 3   # 序列结束标记
        }
        # 添加最常见的词
        most_common = word_counts.most_common(self.vocab_size - len(vocab))
        for word, count in most_common:
            vocab[word] = len(vocab)
        self.vocab = vocab
        self.reverse_vocab = {idx: word for word, idx in vocab.items()}
        return vocab

    def text_to_sequences(self, texts, seq_length=20):
        # 将文本转换为序列
        sequences = []
        targets = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            # 添加序列标记
            words = ['<SOS>'] + words + ['<EOS>']
            # 转换为索引
            word_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
            # 创建训练样本
            for i in range(0, len(word_indices) - seq_length):
                seq = word_indices[i:i + seq_length]
                target = word_indices[i + 1:i + seq_length + 1]
                sequences.append(seq)
                targets.append(target)
        return torch.tensor(sequences), torch.tensor(targets)

class WordLSTMTrainer:
    # 单词级LSTM训练器
    def __init__(self, model, train_loader, val_loader, vocab, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD token

    def train_epoch(self, optimizer, clip_grad=5.0):
        # 训练一个epoch
        self.model.train()
        total_loss = 0
        total_tokens = 0
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, targets = data.to(self.device), targets.to(self.device)
            batch_size, seq_len = data.shape
            state = self.model.init_state(batch_size, self.device) # 初始化状态
            optimizer.zero_grad() # 前向传播
            output, state = self.model(data, state)
            loss = self.criterion(output, targets.view(-1)) # 计算损失
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad) # 梯度裁剪
            optimizer.step() # 参数更新
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
        return total_loss / total_tokens

    def validate(self):
        # 验证
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                batch_size, seq_len = data.shape
                state = self.model.init_state(batch_size, self.device)
                output, state = self.model(data, state)
                loss = self.criterion(output, targets.view(-1))
                total_loss += loss.item() * batch_size * seq_len
                total_tokens += batch_size * seq_len
        return total_loss / total_tokens

    def train(self, num_epochs, learning_rate, patience=5):
        # 完整训练过程
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss = self.train_epoch(optimizer) # 训练
            train_losses.append(train_loss)
            val_loss = self.validate() # 验证
            val_losses.append(val_loss)
            scheduler.step(val_loss) # 学习率调整
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_word_lstm.pth')
            else:
                epochs_no_improve += 1
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Perplexity: {math.exp(val_loss):.2f}')
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        return train_losses, val_losses

class WordGenerator:
    # 单词级文本生成器
    def __init__(self, model, vocab, reverse_vocab, device):
        self.model = model
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.device = device
        self.model.eval()

    def generate_text(self, start_words, max_words=50, temperature=1.0, top_k=10):
        # 生成文本
        # 预处理起始词
        if isinstance(start_words, str):
            start_words = start_words.lower().split()
        generated = start_words.copy()
        state = self.model.init_state(1, self.device)
        for word in start_words: # 预热期
            word_idx = self.vocab.get(word, self.vocab['<UNK>'])
            x = torch.tensor([[word_idx]], device=self.device)
            _, state = self.model(x, state)
        for _ in range(max_words - len(start_words)): # 生成期
            last_word = generated[-1]
            word_idx = self.vocab.get(last_word, self.vocab['<UNK>'])
            x = torch.tensor([[word_idx]], device=self.device)
            logits, state = self.model(x, state)
            # 应用温度采样和top-k筛选
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_word_idx = torch.multinomial(probs, 1).item()
            # 跳过特殊标记
            if next_word_idx in [self.vocab['<PAD>'], self.vocab['<UNK>']]:
                continue
            next_word = self.reverse_vocab.get(next_word_idx, '<UNK>')
            generated.append(next_word)
            # 句子结束检测
            if next_word in ['.', '!', '?'] and len(generated) > 10:
                break
        return ' '.join(generated)

    def generate_multiple_samples(self, start_words, num_samples=3, **kwargs):
        # 生成多个样本
        samples = []
        for i in range(num_samples):
            # 稍微改变温度增加多样性
            temp = kwargs.get('temperature', 1.0) * (1 + 0.1 * i)
            sample = self.generate_text(start_words, temperature=temp, **kwargs)
            samples.append(sample)
        return samples

# 使用示例
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}') # 设置设备
    # 示例文本数据（这里使用时光机器数据集）
    # 在实际使用中，你可以加载任何文本数据
    sample_texts = [
        "The Time Machine is a science fiction novella by H. G. Wells.",
        "It is generally credited with the popularization of the concept of time travel.",
        "The work is often credited with introducing the concept of time travel to fiction.",]
    # 文本处理器
    processor = WordLevelProcessor(vocab_size=8000)
    vocab = processor.build_vocab(sample_texts)
    print(f'Vocabulary size: {len(vocab)}')
    # 准备数据
    sequences, targets = processor.text_to_sequences(sample_texts, seq_length=15)
    print(f'Training sequences: {sequences.shape}')
    # 创建数据集和数据加载器
    dataset = WordLevelDataset(sequences, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # 创建模型
    model = WordLevelLSTM(vocab_size=len(vocab), embedding_dim=200, hidden_size=256, num_layers=2, dropout=0.3).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    # 训练模型
    trainer = WordLSTMTrainer(model, train_loader, val_loader, vocab, device)
    train_losses, val_losses = trainer.train(num_epochs=50, learning_rate=0.001, patience=5)
    # 文本生成
    generator = WordGenerator(model, vocab, processor.reverse_vocab, device)
    # 生成示例
    start_phrases = ["the time", "in the future", "the machine", "he said"]
    for phrase in start_phrases:
        print(f"\nStarting with: '{phrase}'")
        generated = generator.generate_text(phrase, max_words=20, temperature=0.8,top_k=5)
        print(f"Generated: {generated}")

if __name__ == "__main__":
    import math
    main()
"""

