import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from d2l import torch as d2l


# 使用卷积神经网络，自动将一个图像中的风格应用在另一图像之上，即风格迁移
# 首先初始化合成图像，这是风格迁移唯一要更新的变量，即风格迁移所需迭代的模型参数
# 然后选择一个预训练卷积神经网络来抽取图像的特征，其中的模型参数在训练中无须更新
# CNN网络凭借多个层逐级抽取图像特征，可选择其中某些层输出作为内容特征或风格特征
# 接下来，通过前向传播(实线)计算风格迁移损失函数，通过反向传播(虚线)迭代模型参数
# 风格迁移常用损失函数由3部分组成：1.内容损失使合成图像与内容图像在内容特征上接近
# 2.风格损失使合成图像与风格图像在风格上接近;3.全变分损失有助于减少合成图像的噪点
# 最后，当模型训练结束时，输出风格迁移的模型参数，即得到最终的合成图像

d2l.set_figsize()
content_img = d2l.Image.open('/root/data/data/rainier.jpeg')
d2l.plt.imshow(content_img)
style_img = d2l.Image.open('/root/data/data/autumn-oak.jpeg')
d2l.plt.imshow(style_img)

# 预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    # 对输入图像做标准化，将结果变换成CNN接受输入格式
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(image_shape),
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)
def postprocess(img):
    # 将输出图像中的像素值还原回标准化之前的值
    img = img[0].to(rgb_std.device)
    # # torch.clamp(..., 0, 1)：限制到[0,1]范围，并转换回PIL图像格式
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

pretrained_net = torchvision.models.vgg19(pretrained=True)
# 越靠近输入层，越容易抽取图像的细节信息；反之，则越容易抽取图像的全局信息
# 为避免合成图像过多保留内容图像细节，选择VGG靠近输出的层来输出图像内容特征
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
# 使用VGG层抽取特征时，只需要用到从输入层到最靠近输出层的内容层或风格层之间的所有层
net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    # 逐层计算，并保留内容层和风格层的输出
    contents, styles = [], []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device):
    # 对内容图像抽取内容特征
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y
def get_styles(image_shape, device):
    # 对风格图像抽取风格特征
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# 损失函数：风格迁移损失函数由内容损失、风格损失和全变分损失3部分组成
def content_loss(Y_hat, Y):
    # 功能：衡量合成图像与内容图像在内容特征上的差异
    # Y.detach()：从计算图中分离，不计算梯度（固定目标）
    return torch.square(Y_hat - Y.detach()).mean()

# 这些向量的格拉姆矩阵X*X_T ∈ R(c×c)中，i行j列的元素x_ij即向量xi和xj的内积
# 表示通道i和通道j上风格特征相关性。当hw的值较大时，矩阵的元素容易出现较大的值
def gram(X):
    # Gram矩阵：捕捉特征通道之间的相关性
    # 输入 X 形状：(batch_size, channels, height, width)
    # num_channels：特征通道数；n = height * width：空间位置数
    # torch.matmul(X, X.T)：计算Gram矩阵 (channels, channels)
    # 除以 (num_channels * n)：归一化，避免数值过大
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

# 风格损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与风格图像的风格层输出
def style_loss(Y_hat, gram_Y):
    # 功能：衡量合成图像与风格图像在风格特征上的差异
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分去噪：假设xi,j表示坐标(i, j)处的像素值，降低全变分损失能够尽可能使邻近的像素值相似
def tv_loss(Y_hat):
    # 功能：平滑图像，减少噪声和伪影；计算水平和垂直方向的像素差异：
    # Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]：垂直方向相邻像素差
    # Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]：水平方向相邻像素差
    # L_tv = 0.5 * [Σ|Y[i+1,j] - Y[i,j]| + Σ|Y[i,j+1] - Y[i,j]|]
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean()
           + torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失，并对所有损失求和
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# 初始化合成图像
# SynthesizedImage将合成的图像视为模型参数
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))
    def forward(self):
        return self.weight

# get_inits创建合成图像的模型实例，并将其初始化为图像X
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    # 初始化训练所需组件；学习率调度器：每50个epoch将学习率乘以0.8
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs],
               legend=['content', 'style', 'TV'], ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        # 提取当前合成图像的特征：contents_Y_hat：内容特征；styles_Y_hat：风格特征
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            # 在右边显示当前合成图像；在左边添加损失数据点
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

