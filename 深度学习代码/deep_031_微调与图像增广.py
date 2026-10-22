import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from d2l import torch as d2l


""" 
# 1.图像增广：在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而扩大训练集规模
# 此外，应用图像增广的原因：随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力

# 设置图片路径
image_path = r"../zhou/picture.jpg"
img = d2l.Image.open(image_path).convert('RGB')

# 函数在输入图像img上多次运行图像增广方法aug并显示所有结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    # d2l.plt.show()

# 翻转和裁剪
# 左右翻转图像通常不会改变对象的类别，此外还有上下翻转
# apply(img, torchvision.transforms.RandomHorizontalFlip())
# apply(img, torchvision.transforms.RandomVerticalFlip())

# 图像随机裁剪，使物体以不同比例出现在图像不同位置，降低模型对目标位置敏感性
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# 改变颜色：改变图像颜色的四个方面：亮度、对比度、饱和度和色调
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# 综合多种图像增广方法
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
"""



""" 
# 使用图像增广训练模型(数据集下载太慢，没有运行过)
# all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

# 基于多 GPU 和图像增广的模型训练
def train_batch_ch13(net, X, y, loss, trainer, devices): # @save
    # 用多GPU进行小批量训练
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()): # @save
    # 用多GPU进行模型训练
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc 'f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on 'f'{str(devices)}')

batch_size, devices = 256, d2l.try_all_gpus()
net = d2l.resnet18(10, 3)
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

train_with_data_aug(train_augs, test_augs, net)
"""



""" 
# 2.微调：迁移学习常见技巧，包括四个步骤：
# 1.在源数据集上预训练神经网络模型，即源模型
# 2.创建新的神经网络模型，即目标模型，复制源模型上输出层除外模型及其参数
#  假定这些模型参数包含从源数据集中学到的知识，这些知识将适用于目标数据集
#  假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层
# 3.向目标模型添加输出层，输出数是目标数据集中的类别数，随机初始化该层模型参数
# 4.在目标数据集上训练目标模型，输出层从头训练，其他层参数将根据源模型参数微调
# 注意：目标数据集比源数据集小得多时，微调有助于提高模型的泛化能力

# 下载数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5') # @save
data_dir = d2l.download_extract('hotdog')

# 使用ImageFolder加载训练集和测试集，自动根据文件夹结构分类
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 训练数据增强：随机裁剪到224*224、随机水平翻转、转tensor、标准化
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor(),normalize])
# 测试数据预处理：调整大小到256*256、中心裁剪到224*224、转tensor、标准化
test_augs = torchvision.transforms.Compose([torchvision.transforms.Resize([256, 256]),
torchvision.transforms.CenterCrop(224),torchvision.transforms.ToTensor(),normalize])

# 定义和初始化模型
# 加载在ImageNet上预训练好的ResNet-18模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 预训练源模型包含多特征层和一个输出层fc，划分目的是促进对除输出层以外所有层的模型参数进行微调
# print(pretrained_net.fc) # Linear(in_features=512, out_features=1000, bias=True)

# 再次加载预训练模型用于微调
finetune_net = torchvision.models.resnet18(pretrained=True)
# 关键步骤：替换最后的全连接层，将1000类输出改为2类（热狗/非热狗）
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    # 微调训练函数：网络、学习率、批次大小、训练轮数、是否参数分组
    # 如果param_group=True，输出层中的模型参数将使用十倍的学习率
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train'), transform=train_augs), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        # 关键技巧：将参数分为两组，排除最后一层(fc)的参数
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        # 关键技巧：对新添加的输出层使用10倍的学习率，预训练层使用较小学习率
        trainer = torch.optim.SGD([{'params': params_1x},{'params': net.fc.parameters(),'lr': learning_rate * 10}],
        lr=learning_rate, weight_decay=0.001)
    else: # 不分组则定义正常训练优化方式
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

train_fine_tuning(finetune_net, 5e-5) # 微调预训练模型，使用较小的学习率
scratch_net = torchvision.models.resnet18() # 创建从头训练的对比模型（不加载预训练权重）
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
# 从头训练模型，使用较大学习率5e-4，且不对参数分组
train_fine_tuning(scratch_net, 5e-4, param_group=False)
"""