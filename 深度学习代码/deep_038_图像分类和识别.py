import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


# 图像分类
# 下载数据集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
'2068874e4b9a9f0fb07ebe0ad2b29754449ccacd') # @save
# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = False
if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '/root/data/data/cifar-10/'

def read_csv_labels(fname): # @save
    # 读取fname来给标签字典返回一个文件名
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))

    
# 后来添加
def copyfile(filename, target_dir): # @save
    # 将文件复制到目标目录
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(filename))
    shutil.copy(filename, target_path)
    # print(f"复制 {os.path.basename(filename)} 到 {target_dir}")

"""
def copyfile(filename, target_dir): # @save
    # 将文件复制到目标目录
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
"""

# 后来添加
def check_reorganized_structure(data_dir):
    base_dir = os.path.join(data_dir, 'train_valid_test')
    for split in ['train', 'valid', 'train_valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            if split == 'test':
                unknown_dir = os.path.join(split_dir, 'unknown')
                if os.path.exists(unknown_dir):
                    file_count = len(os.listdir(unknown_dir))
                    print(f"  {split}/unknown: {file_count} 个文件")
            else:
                total_files = 0
                for class_dir in os.listdir(split_dir):
                    class_path = os.path.join(split_dir, class_dir)
                    if os.path.isdir(class_path):
                        file_count = len(os.listdir(class_path))
                        total_files += file_count
                        print(f"    {class_dir}: {file_count} 个文件")
                print(f"  {split}: 总共 {total_files} 个文件")
        else:
            print(f"  {split}: 目录不存在")

# 后来添加
def reorg_train_valid(data_dir, labels, valid_ratio): # @save
    # 将验证集从原始的训练集中拆分出来
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    
    # 创建目标目录
    for split in ['train', 'valid', 'train_valid']:
        for label in set(labels.values()):
            os.makedirs(os.path.join(data_dir, 'train_valid_test', split, label), exist_ok=True)
    
    train_dir = os.path.join(data_dir, 'train')
    train_files = os.listdir(train_dir)
    
    copied_count = 0
    skipped_count = 0
    
    for train_file in train_files:
        # 关键修复：直接使用完整的文件名（带扩展名）来匹配标签
        if train_file in labels:
            file_id = train_file
        else:
            # 如果直接匹配失败，尝试其他可能的格式
            file_base = os.path.splitext(train_file)[0]  # 去掉扩展名
            possible_names = [
                train_file,  # 原文件名
                file_base + '.png',  # 添加.png扩展名
                file_base + '.jpg',  # 添加.jpg扩展名
                str(int(file_base)).zfill(5) + '.png',  # 转换为5位数字格式
                file_base,  # 不带扩展名
            ]
            
            file_id = None
            for name in possible_names:
                if name in labels:
                    file_id = name
                    break
            
            if file_id is None:
                skipped_count += 1
                if skipped_count <= 5:  # 只打印前5个跳过文件的示例
                    print(f"跳过文件 {train_file}，在标签中找不到对应项")
                continue
        
        label = labels[file_id]
        fname = os.path.join(train_dir, train_file)
        
        # 确保源文件存在
        if not os.path.exists(fname):
            print(f"警告: 源文件 {fname} 不存在")
            skipped_count += 1
            continue
        
        # 复制到train_valid目录（所有文件都复制）
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        
        # 复制到train或valid目录（根据验证集比例）
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
        
        copied_count += 1
    
    print(f"成功复制 {copied_count} 个文件，跳过 {skipped_count} 个文件")
    
    # 检查每个目录的文件数量
    for split in ['train', 'valid', 'train_valid']:
        split_dir = os.path.join(data_dir, 'train_valid_test', split)
        total_files = 0
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                file_count = len(os.listdir(label_dir))
                total_files += file_count
                if file_count == 0:
                    print(f"警告: {split}/{label} 目录为空")
        print(f"{split} 目录总文件数: {total_files}")
    
    return n_valid_per_label

"""
def reorg_train_valid(data_dir, labels, valid_ratio): # @save
    # 将验证集从原始的训练集中拆分出来
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label
"""

# 后来添加
def reorg_test(data_dir): # @save
    # 在预测期间整理测试集，以方便读取
    # 创建测试目录
    test_target_dir = os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')
    os.makedirs(test_target_dir, exist_ok=True)
    
    # 复制测试文件
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file), test_target_dir)

"""
def reorg_test(data_dir): # @save
    # 在预测期间整理测试集，以方便读取
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
"""

# 后来添加
def reorg_cifar10_data(data_dir, valid_ratio):
    # 确保目标目录存在
    os.makedirs(os.path.join(data_dir, 'train_valid_test'), exist_ok=True)
    
    labels_path = os.path.join(data_dir, 'trainLabels.csv')
    if not os.path.exists(labels_path):
        print(f"错误: 标签文件 {labels_path} 不存在")
        return
        
    labels = read_csv_labels(labels_path)
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))
    
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

"""
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
"""

# 后来添加
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
check_reorganized_structure(data_dir)

"""
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
"""

# 图像增广
# 在高度和宽度上将图像放大到40像素的正方形，随机裁剪出一个高度和宽度均为40像素的正方形图像，
# 生成一个面积为原始图像面积0.64～1倍的小正方形，然后将其缩放为高度和宽度均为32像素的正方形
transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize(40),
torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])

# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
os.path.join(data_dir, 'train_valid_test', folder),
transform=transform_train) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
os.path.join(data_dir, 'train_valid_test', folder),
transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size,
shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 定义损失函数
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")

# 训练模型
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, 'f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(devices)}')

# 模型训练和验证
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 后来添加
# 分类并提交结果
net = get_net()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
preds = []
net.eval()

with torch.no_grad():
    for i, (X, _) in enumerate(test_iter):
        if i % 100 == 0:  # 每100个batch打印进度
            print(f"处理第 {i} 个batch...")
        X = X.to(devices[0])
        y_hat = net(X)
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

print(f"完成预测，共 {len(preds)} 个预测结果")

# 获取测试文件的正确ID
test_ids = []
test_dir = os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')
test_files = sorted(os.listdir(test_dir))

for test_file in test_files:
    # 获取文件名（不带扩展名）作为ID
    file_id = os.path.splitext(test_file)[0]
    test_ids.append(file_id)

print(f"测试文件数量: {len(test_ids)}")

# 确保预测数量与测试文件数量匹配
if len(preds) != len(test_ids):
    print(f"警告: 预测数量 {len(preds)} 与测试文件数量 {len(test_ids)} 不匹配")
    # 如果数量不匹配，截取或填充到正确数量
    min_len = min(len(preds), len(test_ids))
    preds = preds[:min_len]
    test_ids = test_ids[:min_len]
    print(f"调整后: 预测数量 {len(preds)}, 测试文件数量 {len(test_ids)}")

# 创建提交DataFrame
df = pd.DataFrame({'id': test_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])

# 按照ID排序
df = df.sort_values('id')

# 保存提交文件
df.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(df)} 行数据")

"""
# 分类并提交结果
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)

# 在这场Kaggle竞赛中使用完整的CIFAR‐10数据集。将超参数设为batch_size = 128，num_epochs = 100，
# lr = 0.1，lr_period = 50，lr_decay = 0.1。看看在这场比赛中能达到什么准确度和排名。能进一步改进吗？
"""




"""
# 品种识别
# 下载数据集
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
'0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d') # @save
# 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 图像增广
# 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间
# 然后，缩放图像以创建224x224的新图像，并随机更改亮度，对比度和饱和度
transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(
224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)), torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# 从图像中心裁切224x224大小的图片
transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
os.path.join(data_dir, 'train_valid_test', folder),
transform=transform_train) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
os.path.join(data_dir, 'train_valid_test', folder),
transform=transform_test) for folder in ['valid', 'test']]
train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size,
shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 微调预训练模型
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000,
    256), nn.ReLU(), nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

# 训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 只训练小型自定义输出网络
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
    if param.requires_grad), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(devices)}')

# 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 分类并提交结果
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')

# 1. 试试使用完整Kaggle比赛数据集，增加batch_size（批量大小）和num_epochs（迭代轮数），或者设计
# 其它超参数为lr = 0.01，lr_period = 10，和lr_decay = 0.1时，能取得什么结果？
# 2. 如果使用更深的预训练模型，会得到更好的结果吗？如何调整超参数？能进一步改善结果吗？
"""