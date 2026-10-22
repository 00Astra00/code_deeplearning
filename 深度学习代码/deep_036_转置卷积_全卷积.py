import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 为了计算每个中间张量，输入张量中的每个元素都要乘以卷积核，从而使所得的kh × kw张量替换中间张量的一部分
# 请注意，每个中间张量被替换部分的位置与输入张量中元素的位置相对应。最后，所有中间结果相加以获得最终结果
# 转置卷积：用于逆转下采样导致的空间尺寸减小
def trans_conv(X, K):
    h, w = K.shape
    # 输出尺寸计算：输出高度 = 输入高度 + 核高度 - 1，创建全零输出张量
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    # 核心算法：将输入每个元素乘以卷积核，累加到输出对应位置
    # 每个输入元素 X[i,j] 生成一个 h×w 的响应区域，这些响应区域在输出中重叠相加
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

# 转置卷积标准计算过程（无填充，无步幅）
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 在转置卷积中，填充会减小输出尺寸（输出 = 输入 + 核大小 - 1 - 2×填充）
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

# 在转置卷积中，步幅会放大输出尺寸（输出 = (输入 - 1)×步幅 + 核大小 - 2×填充）
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 如果将X代入卷积层f来输出Y = f(X)，并创建与f具有相同的超参数，
# 但输出通道数量是X中通道数的转置卷积层g，那么g(Y)的形状与X相同
# 卷积：(1,10,16,16) → (1,20,6,6)；转置卷积：(1,20,6,6) → (1,10,16,16)
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# print(tconv(conv(X)).shape == X.shape) # True

# 转置卷积与矩阵变换的联系
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K) # ([[27., 37.], [57., 67.]])

def kernel2matrix(K):
    # 卷积的矩阵表示：将卷积操作表示为矩阵乘法
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K) # 验证卷积的矩阵表示
print(W, "\n", Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

Z = trans_conv(Y, K) # 转置卷积 = 卷积矩阵的转置，转置卷积(Y, K) = W_T × Y_flat
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))

# 转置卷积层能够交换卷积层的正向和反向传播函数：正向传播和反向传播函数将输入向量分别与W_T和W相乘



# 全卷积网络：采用卷积神经网络实现从图像像素到像素类别的变换
# 全卷积网络将中间层特征图的高和宽变换回输入图像的尺寸(由前文提及的转置卷积实现放大)
# 输出类别预测与输入图像在像素级别上有一一对应关系：通道维输出即该位置对应像素类别预测
# 全卷积网络先使用卷积神经网络抽取图像特征，然后通过1 × 1卷积层将通道数变换为类别个数，
# 最后通过转置卷积层将特征图的高和宽变为输入图像尺寸，输出通道包含空间位置像素类别预测
pretrained_net = torchvision.models.resnet18(pretrained=True)
# print(list(pretrained_net.children())[-3:])
# 全卷积网络net，复制ResNet‐18中大部分预训练层，除最后全局平均汇聚层和最接近输出全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape) # (1, 512, 10, 15)
num_classes = 21
# 1×1卷积：将通道数从512变为21（VOC的21个类别）；作用：相当于在每个空间位置进行全连接分类
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 转置卷积：上采样恢复原始空间尺寸；输出形状：(1, 21, 320, 480)，与输入图像相同高度和宽度
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

# 初始化转置卷积层
# 图像处理中，有时需要将图像放大，即上采样。双线性插值是常用上采样方法之一，它也经常用于初始化转置卷积层
# 双线性插值，给定输入图像，计算上采样输出图像上的每个像素
# 1.将输出图像的坐标(x, y)映射到输入图像的坐标(x′, y′)上
# 2.在输入图像上找到离坐标(x′, y′)最近的4个像素
# 3.输出图像在坐标(x, y)上的像素依据输入图像上这4个像素及其与(x′, y′)的相对距离来计算
# 双线性插值的上采样可以通过转置卷积层实现，内核由以下bilinear_kernel函数构造
def bilinear_kernel(in_channels, out_channels, kernel_size):
    # 计算插值参数：factor：缩放因子；center：卷积核中心位置
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    # 创建网格坐标：og[0]是行坐标，og[1]是列坐标
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    # 双线性权重计算：每个位置的权重与到中心的距离成反比：距离越近，权重越大
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    # 创建卷积权重张量，将对角线通道设置为双线性滤波核
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 创建转置卷积层，使用双线性初始化(参数：2倍上采样，核大小4)
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

# 加载测试图像，应用双线性上采样，准备可视化输出
img = torchvision.transforms.ToTensor()(d2l.Image.open('/root/data/data/img.png'))
X, Y = img.unsqueeze(0), conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)

# 关键步骤：用双线性核初始化转置卷积层：合理上采样结果，加速训练收敛
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# 训练模型
def loss(inputs, targets):
    # 语义分割损失函数：返回每个样本的平均损失
    # F.cross_entropy(inputs, targets, reduction='none')：
    # inputs: 形状 (batch_size, num_classes, height, width)
    # targets: 形状 (batch_size, height, width)
    # .mean(1): 在高度维度求平均 → 形状 (batch_size, width)
    # .mean(1): 在宽度维度求平均 → 形状 (batch_size,)
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    # 将预测索引转换为彩色图像
    # colormap[X, :]: 使用预测索引从颜色表中查找RGB值
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    return colormap[pred.long(), :] # 返回形状 (H, W, 3) 的彩色分割图

# 由于模型使用步幅32的转置卷积层，因此当输入图像高或宽无法被32整除时，转置卷积层输出的高或宽会与输入图像的尺寸有偏差
# 为解决问题，可在图像中截取多块高和宽为32整数倍的矩形区域，分别对区域中的像素做前向传播(这些区域并集要覆盖输入图像)
# 当一个像素被多个区域所覆盖时，它在不同区域前向传播中转置卷积层输出的平均值可以作为softmax运算的输入，从而预测类别

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    # 定义裁剪区域：从(0,0)开始，尺寸320×480，对每张图像进行裁剪
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    # 对裁剪后的图像进行预测，将预测结果转换为彩色图像
    pred = label2image(predict(X))
    # 收集图像：原始图像 X.permute()；预测分割图 pred.cpu()；真实标签图
    imgs += [X.permute(1,2,0), pred.cpu(), torchvision.transforms.
       functional.crop(test_labels[i], *crop_rect).permute(1,2,0)]
# imgs[::3]: 所有原始图像；imgs[1::3]: 所有预测图像；imgs[2::3]: 所有真实标签
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)



"""
# 习题3
def predict_full_image(model, image, device, crop_size=(320, 480), stride=32):
    # 预测整张图像的所有像素类别，处理尺寸对齐问题
    original_h, original_w = image.shape[1], image.shape[2]
    model.eval()
    
    # 如果图像尺寸是32的倍数，直接预测
    if original_h % 32 == 0 and original_w % 32 == 0:
        X = test_iter.dataset.normalize_image(image).unsqueeze(0)
        with torch.no_grad():
            pred = model(X.to(device)).argmax(dim=1)
        return pred[0].cpu()
    
    # 否则使用滑动窗口预测
    else:
        return sliding_window_predict(model, image, device, crop_size, stride)

def sliding_window_predict(model, image, device, crop_size, stride):
    # 滑动窗口预测，处理任意尺寸图像
    original_h, original_w = image.shape[1], image.shape[2]
    
    # 创建输出概率图
    output_probs = torch.zeros((21, original_h, original_w))
    count_map = torch.zeros((original_h, original_w))
    
    # 生成滑动窗口位置
    h_positions = range(0, original_h - crop_size[0] + 1, stride)
    w_positions = range(0, original_w - crop_size[1] + 1, stride)
    
    # 如果无法覆盖整个图像，添加边界窗口
    if len(h_positions) == 0 or h_positions[-1] + crop_size[0] < original_h:
        h_positions = list(h_positions) + [max(0, original_h - crop_size[0])]
    if len(w_positions) == 0 or w_positions[-1] + crop_size[1] < original_w:
        w_positions = list(w_positions) + [max(0, original_w - crop_size[1])]
    
    for h in h_positions:
        for w in w_positions:
            # 裁剪窗口
            crop = image[:, h:h+crop_size[0], w:w+crop_size[1]]
            
            # 预测
            X = test_iter.dataset.normalize_image(crop).unsqueeze(0)
            with torch.no_grad():
                window_pred = model(X.to(device))
            
            # 累加到输出图
            output_probs[:, h:h+crop_size[0], w:w+crop_size[1]] += window_pred[0].cpu()
            count_map[h:h+crop_size[0], w:w+crop_size[1]] += 1
    
    # 平均重叠区域的预测
    count_map[count_map == 0] = 1  # 避免除零
    averaged_probs = output_probs / count_map.unsqueeze(0)
    
    # 得到最终预测
    final_pred = averaged_probs.argmax(dim=0)
    return final_pred

def visualize_all_predictions(model, test_images, num_images=4):
    # 可视化多张测试图像的完整预测结果
    device = next(model.parameters()).device
    n = min(num_images, len(test_images))
    imgs = []
    
    for i in range(n):
        original_image = test_images[i]
        
        # 预测整张图像
        pred_mask = predict_full_image(model, original_image, device)
        
        # 转换为彩色图像
        pred_rgb = label2image(pred_mask.to(device))
        
        # 收集结果
        imgs.extend([
            original_image.permute(1, 2, 0),      # 原始图像
            pred_rgb.cpu(),                       # 预测结果
        ])
    
    # 显示结果 (2行 x n列)
    d2l.show_images(imgs[::2] + imgs[1::2], 2, n, scale=1.5)

# 使用示例
def complete_prediction_pipeline():
    # 完整的预测流程
    # 加载训练好的模型
    trained_net = net.to(devices[0])
    
    # 预测所有测试图像
    print("开始预测所有测试图像...")
    
    for i, test_image in enumerate(test_images):
        print(f"处理图像 {i+1}/{len(test_images)}")
        
        # 预测整张图像
        full_prediction = predict_full_image(trained_net, test_image, devices[0])
        
        # 保存结果
        pred_rgb = label2image(full_prediction.to(devices[0]))
        
        # 可视化（可选）
        if i < 4:  # 只显示前4张
            d2l.set_figsize((12, 4))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            
            # 显示原始图像
            ax1.imshow(test_image.permute(1, 2, 0))
            ax1.set_title('原始图像')
            ax1.axis('off')
            
            # 显示预测结果
            ax2.imshow(pred_rgb.cpu())
            ax2.set_title('语义分割结果')
            ax2.axis('off')
            
            plt.show()
    
    print("所有图像预测完成！")

# 计算评估指标
def evaluate_predictions(model, test_images, test_labels):
    # 评估预测精度
    device = next(model.parameters()).device
    total_pixels = 0
    correct_pixels = 0
    
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        # 预测
        pred = predict_full_image(model, image, device)
        
        # 转换为类别索引（真实标签）
        colormap2label = voc_colormap2label()
        true_indices = voc_label_indices(label, colormap2label)
        
        # 计算准确率
        correct_pixels += (pred == true_indices).sum().item()
        total_pixels += pred.numel()
        
        if (i + 1) % 10 == 0:
            accuracy = correct_pixels / total_pixels
            print(f"已处理 {i+1} 张图像，当前准确率: {accuracy:.4f}")
    
    final_accuracy = correct_pixels / total_pixels
    print(f"最终像素准确率: {final_accuracy:.4f}")
    return final_accuracy

# 执行完整预测
if __name__ == "__main__":
    # 1. 可视化部分预测结果
    visualize_all_predictions(net, test_images, num_images=4)
    
    # 2. 评估所有测试图像
    accuracy = evaluate_predictions(net, test_images, test_labels)
    
    # 3. 完整的预测流程
    complete_prediction_pipeline()
"""