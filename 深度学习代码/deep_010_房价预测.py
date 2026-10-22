import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 下载数据集 #@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    # 下载一个DATA_HUB中的文件，返回本地文件名
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 检查文件是否存在且哈希匹配
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 每次读取1MB
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存

    # 下载文件
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)

    # 验证下载的文件
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    if sha1.hexdigest() != sha1_hash:
        raise Exception(f'下载文件哈希不匹配: {fname}')

    return fname


def download_extract(name, folder=None):  # @save
    # 下载并解压zip/tar文件
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)

    if ext == '.zip':
        with zipfile.ZipFile(fname, 'r') as fp:
            fp.extractall(base_dir)
    elif ext in ('.tar', '.gz'):
        with tarfile.open(fname, 'r') as fp:
            fp.extractall(base_dir)
    else:
        assert False, '只有zip/tar文件可以被解压缩'

    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    # 下载DATA_HUB中的所有文件
    for name in DATA_HUB:
        download(name)


# 下载数据集并整合
DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# print(f"训练数据形状: {train_data.shape}")
# print(f"测试数据形状: {test_data.shape}")


# 数据预处理
# 删除无用的 ID 列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 将特征重新缩放到零均值和单位方差来标准化数据(x = (x - u(均值)) / o(方差))
# 选择所有数值型特征，获取这些数值特征的列名，识别出所有需要标准化的数值特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对每个数值列应用标准化公式，消除不同特征之间的量纲差异，使模型训练更稳定
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，均值消失，原本将所有缺失值替换成相应特征的平均值，现在可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散值
# 获取所有非数值型特征
object_features = all_features.dtypes[all_features.dtypes == 'object'].index
# 对于每个非数值型特征，检查是否有缺失值并处理
for feature in object_features:
    # 将缺失值替换为字符串 'missing'，这样get_dummies会将其作为单独类别
    all_features[feature] = all_features[feature].fillna('missing')
# pd.get_dummies()：将分类变量转换为独热编码，dummy_na=False因为已处理缺失值
all_features = pd.get_dummies(all_features, dummy_na=False)
all_features = all_features.astype(float) # 确保所有数据都是数值类型
# print(all_features.shape) # 打印处理后的特征矩阵形状
n_train = train_data.shape[0] # 获取训练数据的样本数量（行数）
# [:n_train]：选择前n_train行作为训练特征，.values：将pandas DataFrame转换为numpy数组
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# train_data.SalePrice.values：获取训练数据的目标变量（房价）值，并保持与特征 张量 相同的样本维度
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 训练函数
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

# 房价关心的是相对数量，而不是绝对数量，因此用价格预测的对数来衡量差异
# torch.clamp() 是PyTorch中的函数，用于将 张量 中的元素限制在指定范围内
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # 将预测值限制在[1, ∞)范围内，避免取对数时出现负数或0
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    # 将单元素张量转换为Python数值
    return rmse.item()

# Adam优化器的主要吸引力在于它对初始学习率不那么敏感
def train(net, train_features, train_labels, test_features,
    test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], [] # 建立训练和测试损失数据列表
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法(输入参数，学习率，以及权重衰减方法)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate,weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None: # 注意判断，不然 train_and_pred 会出错
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # slice()：创建切片对象，选择当前折的数据
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i: # 当前折索引j等于指定的验证折索引i，则将这一折作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None: # 第一个遇到的非验证折，直接将其赋值给训练集
            X_train, y_train = X_part, y_part
        else: # 其他非验证折，沿第0维度拼接数据，将多个折的数据合并为训练集
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# 基于K折交叉验证训练函数，返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data, net = get_k_fold_data(k, i, X_train, y_train), get_net()
        # train(net, *data, ...)：训练模型，*data将元组解包为四个参数，返回训练和验证误差的历史记录列表
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1] # train_ls[-1]：获取最后一个epoch的训练误差
        valid_l_sum += valid_ls[-1] # valid_ls[-1]：获取最后一个epoch的验证误差
        if i == 0: # 只在第一折（i=0）时绘制训练曲线，可视化训练过程，观察是否过拟合
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch',
                ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
            # d2l.plt.show()
        # print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 'f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k # 返回训练和验证误差的平均值

# 使用所有数据对其进行训练，并应用于测试集
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    # 使用所有训练数据训练模型。None, None：测试特征和测试标签设为None，表示不使用验证集
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 对测试集进行预测，生成测试集的预测结果
    # .detach()：从计算图中分离，避免梯度计算
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# 超参数搜索函数
def hyperparameter_tuning(X_train, y_train):
    # 定义要搜索的超参数范围
    learning_rates = [1, 3, 5, 10]
    weight_decays = [0, 0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    num_epochs, k = 100, 5

    best_score = float('inf')
    best_params = {}

    # 网格搜索超参数
    for lr in learning_rates:
        for wd in weight_decays:
            for bs in batch_sizes:
                print(f"测试超参数: lr={lr}, wd={wd}, bs={bs}")

                # 使用K折交叉验证评估
                train_score, valid_score = k_fold(k, X_train, y_train, num_epochs, lr, wd, bs)
                print(f"平均验证log rmse: {valid_score:.4f}")

                # 选择验证误差最小的超参数
                if valid_score < best_score:
                    best_score = valid_score
                    best_params = {
                        'learning_rate': lr,
                        'weight_decay': wd,
                        'batch_size': bs,
                        'num_epochs': num_epochs
                    }

    print(f"\n最佳超参数: {best_params}")
    print(f"最佳验证分数: {best_score:.4f}")
    return best_params


if __name__ == '__main__':
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 10, 0.01, 32
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    # best_params = hyperparameter_tuning(train_features, train_labels) # 执行超参数调优