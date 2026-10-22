import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import torch
import torchvision
from d2l import torch as d2l


# 下载数据集
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                     '4e443f8a2eca6b1dac8a6c57641b67dd40621a49') # @save
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True): # @save
    # 读取所有VOC图像并标注
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
        features, labels = [], []
        for i, fname in enumerate(images):
            features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
            labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
imgs = train_features[0:5] + train_labels[0:5]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, 5)

# 列举RGB颜色值和类名
# @save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
[0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
[192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
[64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
[128, 192, 0], [0, 64, 128]]
# @save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
'horse', 'motorbike', 'person','potted plant', 'sheep', 'sofa', 'train',
'tv/monitor']

def voc_colormap2label(): # @save
    # 构建从RGB到VOC类别索引的映射
    # 创建一个大小为256³的零张量（所有可能的RGB组合）
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        # 对每个RGB颜色计算索引：(R * 256 + G) * 256 + B(将该索引位置的值设为类别索引 i)
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label): # @save
    # 将VOC标签中的RGB值映射到它们的类别索引
    # 输入 colormap：形状为 (3, H, W) 的RGB标签图像
    # permute(1, 2, 0)：从 (3, H, W) 变为 (H, W, 3)
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    # colormap[:, :, 0/1/2]：红/绿/蓝色通道，形状 (H, W)
    # 结果 idx：形状 (H, W)，每个像素对应一个编码整数
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140],"\n", VOC_CLASSES[1]) # aeroplane

# 数据预处理
def voc_rand_crop(feature, label, height, width): # @save
    # 对特征图像和标签图像进行相同的随机裁剪
    # RandomCrop.get_params(...)：获取随机裁剪的参数，确保裁剪区域在图像范围内
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    # 使用相同的裁剪参数对特征图和标签图进行裁剪，保持特征和标签的空间对齐
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs, n = [], 5
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)

class VOCSegDataset(torch.utils.data.Dataset): # @save
    # 一个用于加载VOC数据集的自定义数据集
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')
    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    def filter(self, imgs):
        # 过滤太小图像：只保留高度 ≥ crop_size[0] 且宽度 ≥ crop_size[1] 的图像，确保可以进行指定尺寸的裁剪
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]
    def __getitem__(self, idx):
        # 获取第idx个样本，对特征和标签进行相同的随机裁剪
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # 返回：feature：标准化后的特征图像，形状 (3, H, W)；标签的类别索引，形状 (H, W)（通过颜色映射转换）
        return (feature, voc_label_indices(label, self.colormap2label))
    def __len__(self):
        return len(self.features)

crop_size, batch_size = (320, 480), 64
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
# 创建数据加载器：drop_last=True：丢弃最后一个不完整批次；num_workers：多进程数据加载
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
             drop_last=True, num_workers=d2l.get_dataloader_workers())

# 整合组件
def load_data_voc(batch_size, crop_size): # @save
    # 加载VOC语义分割数据集
    voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir),
                 batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir),
                 batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
print("over!!!")
