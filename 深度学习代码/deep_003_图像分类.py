import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l

d2l.use_svg_display()

def get_fashion_mnist_labels(labels): # @save
    # 返回Fashion-MNIST数据集的文本标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): # @save
    # 绘制图像列表，scale: 控制图像显示大小的缩放因子
    figsize = (num_cols * scale, num_rows * scale)
    # ‘ _ ’ 是一个变量名，表示：忽略这个返回值
    # plt.subplots() 返回两个值：(figure, axes)
    # 只需 axes（坐标轴对象），不要 figure（图形对象）
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() # 展平数组
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy()) # 图片张量
        else:
            ax.imshow(img) # PIL图片
        # ax 是 matplotlib.axes.Axes 类的实例，包括以下：
        # 坐标轴：x轴和y轴；刻度：坐标轴上的标记；标签：坐标轴标签
        # 绘图区域：实际绘制图形的区域；图例：数据标识；标题：图表标题
        ax.axes.get_xaxis().set_visible(False) # 隐藏x坐标轴
        ax.axes.get_yaxis().set_visible(False) # 隐藏y坐标轴
        if titles: # 如需要则设置标题
            ax.set_title(titles[i])
    return axes

# 读取小批量
def get_dataloader_workers(): # @save
    # 使用4个进程来读取数据
    return 4

# 整合所有代码
def load_data_fashion_mnist(batch_size, resize=None): # @save
    # 下载Fashion-MNIST数据集，然后将其加载到内存中
    # 通过 ToTensor 实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在 0～1 之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # root = "../data": 数据存储目录；train = True/False: 加载训练集/测试集
    # transform = trans: 应用上面定义的数据转换；download = True: 如果数据不存在则自动下载
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    # shuffle = True: 训练集打乱顺序；num_workers = get_dataloader_workers(): 使用多进程加载数据
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()),
    data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))

# 注意：过小的 batch (<10)会降低运行效率，时间反而更慢
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break