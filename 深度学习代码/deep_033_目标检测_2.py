import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l



# 多尺度目标检测
# 当使用较小锚框检测较小物体时，采样更多区域，而对于较大物体，采样较少区域

# display_anchors函数定义如下：在特征图上生成锚框，每个单位作为锚框的中心
# 由于锚框(x,y)轴坐标值被除以特征图宽度和高度，介于0和1之间，表示锚框相对位置
# 给定特征图宽度和高度，函数将均匀对任何输入图像fmap_h行和fmap_w列像素采样
# 以这些均匀采样的像素为中心，将会生成大小为s且宽高比（ratios）不同的锚框
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)

img = d2l.plt.imread('/root/data/zhou/img.png')
h, w = img.shape[:2] # (561, 728)
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

# 特征图在相同空间位置的c个单元在输入图像上的感受野相同：表征同一感受野内的输入图像信息
# 因此,可以将特征图在同一空间位置的c个单元变换为使用此空间位置生成的a个锚框类别和偏移量
# 当不同层的特征图在输入图像上分别拥有不同大小的感受野时，它们可以用于检测不同大小的目标




# 目标检测数据集
d2l.DATA_HUB['banana-detection'] = (d2l.DATA_URL + # @save
'banana-detection.zip', '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True): # @save
    # 读取香蕉检测数据集中的图像和标签
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir,
        'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        # target包含（类别，左上角x，左上角y，右下角x，右下角y），所有图像都是香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset): # @save
    # 一个用于加载香蕉检测数据集的自定义数据集
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
        is_train else f' validation examples'))
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size): # @save
    # 加载香蕉检测数据集
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter

# 数组中的第一个元素是边界框中对象的类别，其中‐1表示用于填充的非法边界框
# 数组的其余四个元素是边界框左上角和右下角的（x，y）坐标值（值域在0～1之间）
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])




# 单发多框检测(SSD)：通过多尺度特征块，单发多框检测生成不同大小的锚框，
# 并预测边界框类别和偏移量来检测大小不同的目标，因此是多尺度目标检测模型

# 类别预测层：使用一个保持输入高和宽的卷积层，输出和输入在特征图宽和高上的空间坐标对应
# 考虑输出和输入同一空间坐标：输出特征图坐标通道包含以输入特征图坐标为中心生成的所有锚框类别预测
# 输出通道数为a(q + 1)，其中索引为i(q + 1) + j的通道代表了索引为i的锚框有关类别索引为j的预测
# 下面定义一个类别预测层，通过参数num_anchors和num_classes分别指定了a和q
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
# 边界框预测层：与类别预测层的设计类似，不同的是，需要为每个锚框预测4个偏移量，而不是q + 1个类别
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
# 连结多尺度的预测
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape) # (torch.Size([2, 55, 20, 20]), torch.Size([2, 33, 10, 10]))

# 尽管Y1和Y2在具有不同的大小，仍然可在同一个小批量的两个不同尺度上连接这两个预测输出
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 高和宽减半块：扩大每个单元在其输出特征图中的感受野
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape) # (2, 10, 10, 10)

# 基本网络块：用于从输入图像中抽取特征
def base_net():
    blk = [] # 使用下采样块逐步减小特征图尺寸
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape) # (2, 64, 32, 32)

# 完整的模型：五个模块组成，每个块生成的特征图既用于生成锚框，又用于预测锚框的类别和偏移量
# 这五个模块分别是：1.基本网络块，2~4.高和宽减半块，5.使用全局最大池将高度和宽度都降到1
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 前向传播输出包括：CNN特征图Y；当前尺度下根据Y生成的锚框；预测的锚框类别和偏移量
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # idx_to_in_channels：定义5个模块的输入通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 动态创建15个子模块（5×3）：blk_0 到 blk_4：特征提取模块
            # cls_0 到 cls_4：类别预测器；bbox_0 到 bbox_4：边界框预测器
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))
    def forward(self, X):
        # 初始化3个列表，用于存储5个模块的输出
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 循环处理5个模块：getattr(self, f'blk_{i}')：动态获取第i个特征提取模块
            # 返回4个值：X：更新后的特征图(传递下模块)；anchors[i]：第i个模块生成的锚框
            # cls_preds[i]：第i个模块的类别预测；bbox_preds[i]：第i个模块的边界框预测
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'),
            sizes[i], ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1) # 将所有5个模块生成的锚框在维度1上拼接
        cls_preds = concat_preds(cls_preds) # 合并所有模块的类别预测，重塑为合适的形状
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds) # 合并所有模块的边界框预测
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
print('output anchors:', anchors.shape) # output anchors: torch.Size([1, 5444, 4])
print('output class preds:', cls_preds.shape) # output class preds: torch.Size([32, 5444, 2])
print('output bbox preds:', bbox_preds.shape) # output bbox preds: torch.Size([32, 21776])

# TinySSD模型训练
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 目标检测有两种类型损失：第一种有关锚框类别的损失：复用图像分类问题里使用的交叉熵损失函数来计算
# 第二种有关正类锚框偏移量损失：预测偏移量是回归问题，使用L1范数损失，即预测值和真实值之差绝对值
# 掩码变量bbox_masks令负类和填充锚框不参与损失计算。最后将锚框类别和偏移量损失相加，获得最终损失函数

# reduction='none'：不自动求平均，保留每个样本的损失
cls_loss = nn.CrossEntropyLoss(reduction='none')
# bbox_loss = nn.L1Loss(reduction='none')
bbox_loss = nn.SmoothL1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # cls_preds：类别预测，形状 (batch_size, num_anchors, num_classes+1)
    # cls_labels：类别标签，形状 (batch_size, num_anchors)
    # bbox_preds：边界框预测，形状 (batch_size, num_anchors×4)
    # bbox_labels：边界框标签，形状 (batch_size, num_anchors×4)
    # bbox_masks：边界框掩码，形状 (batch_size, num_anchors×4)
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # 类别损失计算：cls_preds.reshape(-1, num_classes) 重塑为 (batch_size×num_anchors, num_classes)
    # 计算交叉熵损失，得到 (batch_size×num_anchors,) 形状；.reshape(batch_size, -1)：恢复为 (batch_size, num_anchors)
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # 边界框损失计算：bbox_preds * bbox_masks：只保留正样本的预测值，bbox_labels * bbox_masks：只保留正标签的预测值
    # 计算L1损失，形状 (batch_size, num_anchors×4)；.mean(dim=1)：对每个样本的所有偏移量求平均，得到 (batch_size,)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox # 总损失：类别损失 + 边界框损失：返回形状：(batch_size,)，每个样本一个总损失值

def cls_eval(cls_preds, cls_labels):
    # 类别预测结果放在最后维，argmax需指定最后维；cls_preds.argmax(dim=-1)：获取每个锚框预测类别
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 边界框回归评估：bbox_labels - bbox_preds：计算预测误差
    # * bbox_masks：只考虑正样本；.sum()：累加所有误差；最后返回总L1误差值
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 创建累加器，用于统计4个指标：
    # 正确分类的锚框数量；总锚框数量
    # 边界框L1误差总和；正样本锚框数量
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为锚框生成训练标签：bbox_masks：边界框掩码（正样本为1，负样本为0）
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
        bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    # cls_err：类别错误率 = 1 - (正确分类数 / 总锚框数)
    # bbox_mae：边界框平均绝对误差 = L1误差总和 / 正样本数
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on 'f'{str(device)}')

# 预测目标
# multibox_detection函数根据锚框及其预测偏移量得到预测边界框
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # 关键处理：F.softmax(cls_preds, dim=2)：在类别维度应用softmax，将原始输出转换为概率
    # (batch_size,num_anchors,num_classes)->(batch_size,num_classes,num_anchors)
    # 这是 multibox_detection 函数要求的输入格式
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 多框检测处理，包括：非极大值抑制（NMS）；置信度阈值过滤；背景类别标记
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # 过滤背景检测：row[0] != -1：只保留类别ID不是-1的检测（即非背景）
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx] # 返回过滤后的检测结果

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1]) # 获取置信度分数
        if score < threshold: # 跳过低于阈值的检测
            continue
        h, w = img.shape[0:2] # 坐标转换：获取图像的高度和宽度
        # row[2:6]：提取归一化的边界框坐标 [xmin, ymin, xmax, ymax]
        # * torch.tensor((w, h, w, h))：将归一化坐标转换为实际像素坐标
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

output = predict(X)
display(img, output.cpu(), threshold=0.9)

# 习题1
# 1.平滑L1损失 (Smooth L1 Loss)
# f(x) = (σx)²/2, if |x| < 1/σ²
#        |x| - 0.5/σ², otherwise
# σ ：控制平滑区域的超参数
# 当 σ 很大时：接近L1损失
# 当 σ 较小时：在零点附近更平滑
# 三种sigma值比较：sigmas = [10, 1, 0.5]
# σ=10：几乎等同于L1损失，在零点有尖锐的拐点
# σ=1：适度的平滑，平衡了L1和L2的优点
# σ=0.5：非常平滑，在零点附近使用平方函数
# 优势：在零点可导：解决了L1损失在零点不可导的问题
# 对异常值不敏感：相比L2损失，对大误差的惩罚更温和
# 训练稳定性：梯度更加平滑，有助于训练收敛
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

d2l.set_figsize()
x = torch.arange(-2, 2, 0.1)
sigmas, lines = [10, 1, 0.5], ['-', '--', '-.']
for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend()

# 2.焦点损失 (Focal Loss)
# FL(p_j) = -α(1 - p_j)^γ log(p_j)
# 参数说明：p_j：真实类别的预测概率
# γ (gamma)：调节因子，控制难易样本的权重
# α (alpha)：类别平衡因子（示例中简化为1）
# 三种gamma值的比较：gammas = [0, 1, 5]
# γ=0：等同于标准交叉熵损失
# γ=1：适度降低简单样本的权重
# γ=5：显著降低简单样本的权重，专注于困难样本
# 焦点损失的作用机制
# 对于高置信度正确分类（简单样本）：
# p_j 接近1 → (1-p_j)^γ 接近0 → 损失很小
# 模型不太关注已经分类正确的样本
# 对于低置信度或错误分类（困难样本）：
# p_j 较小 → (1-p_j)^γ 接近1 → 损失较大
# 模型更加关注难以分类的样本
# 在SSD中应用优势：解决类别不平衡
# SSD中负样本（背景）远多于正样本
# 焦点损失自动降低简单负样本的权重
# 提升对难分类物体的检测能力
# 改善小物体、遮挡物体的检测效果
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend()

class ImprovedSSDLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, sigma=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

    def focal_loss(self, cls_preds, cls_labels):
        # 焦点损失实现
        p = torch.softmax(cls_preds, dim=-1)
        p_j = p.gather(1, cls_labels.unsqueeze(1)).squeeze(1)
        loss = -self.alpha * (1 - p_j) ** self.gamma * torch.log(p_j + 1e-8)
        return loss

    def smooth_l1_loss(self, bbox_preds, bbox_labels, bbox_masks):
        # 平滑L1损失实现
        diff = bbox_preds - bbox_labels
        abs_diff = torch.abs(diff)
        mask = abs_diff < (1 / (self.sigma ** 2))
        loss = torch.where(mask, 0.5 * (self.sigma * diff) ** 2, abs_diff - 0.5 / (self.sigma ** 2))
        return (loss * bbox_masks).sum()

    def forward(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls_loss = self.focal_loss(cls_preds, cls_labels).mean()
        bbox_loss = self.smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks)
        return cls_loss + bbox_loss

# 习题2
# 1.输入图像尺寸调整
def adaptive_resize(image, target_size=512, min_size=300):
    # 自适应调整图像尺寸
    h, w = image.shape[1:3]

    # 如果图像较小，放大到目标尺寸
    if max(h, w) < min_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                                      mode='bilinear', align_corners=False)
        return resized_image.squeeze(0), scale
    else:
        return image, 1.0

# 2.负锚框采样策略改进
def balance_negative_anchors(anchors, cls_labels, negative_ratio=0.5):
    # 通过调整负锚框尺寸来平衡类别分布
    positive_mask = cls_labels > 0  # 正样本掩码
    negative_mask = cls_labels == 0  # 负样本掩码

    # 对负锚框进行下采样：将宽高减半
    negative_anchors = anchors[negative_mask]
    if len(negative_anchors) > 0:
        # 计算锚框中心点和尺寸
        center_anchors = d2l.box_corner_to_center(negative_anchors)
        # 将宽高减半
        center_anchors[:, 2:] = center_anchors[:, 2:] * 0.5
        # 转换回角点格式
        downsized_anchors = d2l.box_center_to_corner(center_anchors)

        # 替换原始负锚框
        anchors[negative_mask] = downsized_anchors

    return anchors


# 在线难例挖掘 (OHEM)
def online_hard_example_mining(cls_loss, cls_labels, negative_ratio=3):
    # 选择最难负样本进行训练
    positive_mask = cls_labels > 0
    negative_mask = cls_labels == 0

    # 保留所有正样本
    positive_loss = cls_loss[positive_mask]

    # 选择负样本中损失最大的前k个
    negative_loss = cls_loss[negative_mask]
    if len(negative_loss) > 0:
        # 按负:正比例选择
        num_neg = min(len(negative_loss), positive_mask.sum() * negative_ratio)
        _, indices = torch.topk(negative_loss, num_neg)
        hard_negative_loss = negative_loss[indices]
    else:
        hard_negative_loss = torch.tensor([0.0], device=cls_loss.device)

    # 合并损失
    balanced_loss = torch.cat([positive_loss, hard_negative_loss])
    return balanced_loss.mean()

# 3.损失函数权重调整
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Focal Loss for dense object detection

        Args:
            gamma (float): 调节因子，控制难易样本权重
            alpha (float): 类别平衡因子，可以是一个float或list/tensor
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测概率或logits，形状 (N, C) 或 (N, C, H, W)
            targets: 真实标签，形状 (N,) 或 (N, H, W)
        """
        # 如果输入是logits，先计算softmax
        if inputs.dim() > 2:
            # 对于密集预测任务，如目标检测
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
            inputs = inputs.view(-1, inputs.size(-1))  # (N*H*W, C)
            targets = targets.view(-1)  # (N*H*W,)

        # 计算softmax概率
        probs = F.softmax(inputs, dim=-1)

        # 获取真实类别的预测概率
        class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算基础交叉熵
        log_p = torch.log(class_probs + 1e-8)

        # 计算focal weight: (1 - p_t)^gamma
        focal_weight = (1 - class_probs) ** self.gamma

        # 应用alpha平衡
        if isinstance(self.alpha, (list, tuple, torch.Tensor)):
            # 如果alpha是每个类别的权重
            alpha = self.alpha.gather(0, targets)
            loss = -alpha * focal_weight * log_p
        else:
            # 如果alpha是标量，主要用于平衡正负样本
            loss = -self.alpha * focal_weight * log_p

        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
class BalancedSSDLoss(nn.Module):
    def __init__(self, cls_weight=1.0, bbox_weight=1.0,
                 negative_ratio=3, use_focal_loss=True):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.negative_ratio = negative_ratio

        if use_focal_loss:
            self.cls_loss = FocalLoss(gamma=2, alpha=0.25)
        else:
            self.cls_loss = nn.CrossEntropyLoss(reduction='none')

        self.bbox_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size = cls_preds.shape[0]

        # 分类损失
        cls_loss = self.cls_loss(cls_preds, cls_labels)

        # 应用在线难例挖掘
        cls_loss = self.ohem_loss(cls_loss, cls_labels)

        # 边界框损失（只计算正样本）
        bbox_loss = self.bbox_loss(bbox_preds * bbox_masks,
                                   bbox_labels * bbox_masks)
        bbox_loss = bbox_loss.sum() / (bbox_masks.sum() + 1e-8)

        # 加权总损失
        total_loss = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss

        return total_loss

    def ohem_loss(self, cls_loss, cls_labels):
        # 在线难例挖掘
        batch_loss = 0
        for i in range(cls_loss.shape[0]):
            sample_loss = cls_loss[i]
            sample_labels = cls_labels[i]

            positive_mask = sample_labels > 0
            negative_mask = sample_labels == 0

            # 正样本损失
            pos_loss = sample_loss[positive_mask]

            # 负样本：选择最难的
            neg_loss = sample_loss[negative_mask]
            if len(neg_loss) > 0:
                num_neg = min(len(neg_loss), positive_mask.sum() * self.negative_ratio)
                _, indices = torch.topk(neg_loss, num_neg)
                hard_neg_loss = neg_loss[indices]
            else:
                hard_neg_loss = torch.tensor([0.0], device=cls_loss.device)

            # 合并
            if len(pos_loss) > 0 or len(hard_neg_loss) > 0:
                sample_total_loss = torch.cat([pos_loss, hard_neg_loss]).mean()
            else:
                sample_total_loss = torch.tensor(0.0, device=cls_loss.device)

            batch_loss += sample_total_loss

        return batch_loss / cls_loss.shape[0]