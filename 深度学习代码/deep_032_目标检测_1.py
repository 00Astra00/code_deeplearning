import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l


""" 
# 图像有多个目标，想知道类别和在图像中的位置,称为目标检测
image_path = r"../zhou/img.png"
img = d2l.Image.open(image_path).convert('RGB')

# 边界框：描述对象空间位置。边界框是矩形的，由左上角及右下角的x、y坐标决定
# 另一种常用的边界框表示方法是边界框中心的(x, y)轴坐标以及框的宽度和高度
# box_corner_to_center从两角转换为中心宽度表示法
def box_corner_to_center(boxes): # @save
    # 从（左上，右下）转换到（中间，宽度，高度）
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

# box_center_to_corner从中心宽度转换为两角表示法
def box_center_to_corner(boxes): # @save
    # 从（中间，宽度，高度）转换到（左上，右下）
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy * 0.5 + h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

def bbox_to_rect(bbox, color): # @save
    # 辅助函数bbox_to_rect，将边界框表示成matplotlib的边界框格式
    # 边界框(左上x,左上y,右下x,右下y) -> matplotlib：((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
           height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2)

# 在图像上添加边界框之后，可以看到两个物体的轮廓基本上在两个框内
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show()
"""




# 锚框：目标检测算法会在输入图像中采样大量区域，判断其中是否包含感兴趣目标，并调整区域边界从而更准确预测目标真实边界框
# 不同模型区域采样方法可能不同。其中一种方法：以每个像素为中心，生成多个缩放比和宽高比不同的边界框，这些边界框被称为锚框
# torch.set_printoptions(2) # 精简输出精度
# 输入图像高h，宽w，以图像每个像素为中心生成不同形状锚框：缩放比为s ∈ (0, 1]，
# 宽高比为r > 0。那么锚框的宽度和高度分别是 hs * sqrt(r) 和 hs / sqrt(r)
# 使得缩放比取值 s_1, . . . , s_n 和许多宽高比取值 r_1, . . . , r_m 等
# 这些比例和长宽比的组合以每个像素为中心时，输入图像共有 w * h * n * m 个锚框
# 实践中只考虑包含s1或r1的组合：(s1,r1),...,(s1,rm),(s2,r1),...,(sn,r1)
# 同一像素为中心锚框数量：n + m −1，整个输入图像将生成 w*h (n + m −1) 个锚框

# 生成以每个像素为中心具有不同形状的锚框
def multibox_prior(data, sizes, ratios): # @save
    # 输入：输入数据、尺寸列表、宽高比列表
    in_height, in_width = data.shape[-2:] # 获取输入特征图的高度和宽度
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 公式：每个像素生成的锚框数 = 尺寸数 + 宽高比数 - 1
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # 为了将锚点移动到像素的中心，设置偏移量为0.5
    # 然后计算y轴和x轴的步长（归一化到[0,1]区间）
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步长
    steps_w = 1.0 / in_width # 在x轴上缩放步长
    # 生成所有可能的中心点y坐标和x坐标（归一化坐标）
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 创建网格：每个像素的中心坐标，并展平为一维张量，便于后续处理
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 生成“boxes_per_pixel”个高和宽，之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # 关键理解：第一部分：固定第一个宽高比，应用所有尺寸；第二部分：固定第一个尺寸，应用其他宽高比
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))\
        * in_height / in_width # * in_height / in_width：调整宽度以适应输入图像的宽高比
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # torch.stack((...))：创建偏移量矩阵；-w, -h：左上角偏移；w, h：右下角偏移
    # .T：转置，使每行对应一个锚框的四个偏移量；.repeat(...)：为每个像素点重复这些偏移量
    # / 2：因为 w, h 是完整宽度高度，需要除以2得到半宽半高
    anchor_manipulations = (torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2)
    # 每个中心点都将有“boxes_per_pixel”个锚框，所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations # 核心计算：中心点坐标 + 偏移量 = 最终的边界框坐标
    return output.unsqueeze(0) # 添加批次维度，返回形状为 (1, 总锚框数, 4) 的张量

image_path = r"../zhou/img.png"
img = d2l.Image.open(image_path).convert('RGB')
img_array = np.array(img)  # 转换为 NumPy 数组
h, w = img_array.shape[:2]  # 现在使用 shape 属性
# print(f"高度: {h}, 宽度: {w}") # [561, 728]

X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape) # torch.Size([1, 2042040, 4])

boxes = Y.reshape(h, w, 5, 4)
# 输出(250, 250)的第一个锚框坐标
# print(boxes[250, 250, 0, :])

# show_bboxes函数来在图像上绘制多个边界框
def show_bboxes(axes, bboxes, labels=None, colors=None): # @save
    # axes:坐标轴对象；bboxes:边界框列表，每个边界框为 [xmin, ymin, xmax, ymax]
    def _make_list(obj, default_values=None):
        # 统一处理输入参数，确保返回列表形式
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)] # 按序循环设置颜色
        # 关键转换：将边界框坐标转换为matplotlib的Rectangle对象
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect) # 将矩形边界框添加到坐标轴上显示
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            # 添加文本标签：在边界框左上角显示标签(labels[i])
            # rect.xy[0], rect.xy[1]：文本位置（边界框左上角坐标）
            # bbox=dict(facecolor=color, lw=0)：文本背景框
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                   fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))

d2l.set_figsize()
bbox_scale, fig = torch.tensor((w, h, w, h)), d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, ['s=0.75, '
       'r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
# d2l.plt.show() # 显示所有边界框

# 交并比：衡量锚框和真实边界框间的相似性，杰卡德系数是集合A和B交集的大小除以并集的大小
def box_iou(boxes1, boxes2): # @save
    # 计算两个锚框或边界框列表中成对的交并比；Lambda函数：计算边界框面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *(boxes[:, 3] - boxes[:, 1]))
    # boxes1：(boxes1的数量,4), boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,), areas2：(boxes2的数量,)
    areas1, areas2 = box_area(boxes1), box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters形状: (boxes1数量,boxes2数量,2)
    # 广播机制计算所有边界框对的交集左上角坐标，boxes1的形状从 (N, 4) 变为 (N, 1, 2)
    # 这样可与形状为 (M, 2) 的boxes2进行广播，torch.max()取两个边界框左上角坐标最大值
    # 结果：形状为 (N, M, 2)，表示所有边界框对交集的左上角坐标
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # 类似地计算交集右下角坐标；torch.min()：取两个边界框右下角坐标的最小值
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # inter_lowerrights - inter_upperlefts 计算交集的宽度和高度
    # .clamp(min=0)：负值钳制为0，形状(N,M,2)表示所有边界框对交集宽高
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    # 计算交集面积：形状为 (N, M) 的交集面积矩阵
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 计算并集面积：areas1[:, None]：将形状从 (N,) 变为 (N, 1)，便于广播
    # areas2：形状 (M,)，自动广播为 (N, M)，最终形状 (N, M) 的并集面积矩阵
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas # 计算交并比：交集面积 / 并集面积

# 训练集中，每个锚框视为一个训练样本。为训练目标检测模型，需要每个锚框类别和
# 偏移量标签(前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量)
# 预测时为每个图像生成多个锚框，预测所有锚框类别和偏移量，根据预测偏移量调整位
# 置以获得预测的边界框，最后只输出符合条件预测边界框

# 将真实边界框分配给锚框的算法：
# 给定图像，假设锚框是A1, A2,..., An，真实边界框是B1,B2,..., Bn，na ≥ nb
# 矩阵X∈R(na*nb)，其中第i行、第j列的元素x_ij是锚框Ai和真实边界框Bj的交并比：
# 1.在矩阵X中找到最大的元素，并将它的行和列索引表示为i_1和j_1，
# 将真实边界框B_j1分配给锚框A_i1，丢弃矩阵中i1行和j1列中所有元素
# 2.此时，矩阵X中一行和一列中的元素已被丢弃，继续直到丢弃掉矩阵X中
# nb列中的所有元素，此时已经为这nb个锚框各自分配了一个真实边界框
# 3.遍历剩下na − nb个锚框。给定锚框Ai，在矩阵X的第i行中找到与Ai的
# IoU最大的真实边界框Bj，当此IoU大于预定义的阈值时，才将Bj分配给Ai

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5): # @save
    # 将最接近的真实边界框分配给锚框：iou_threshold=0.5：IoU阈值，用于判断是否分配
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth) # x_ij是锚框i和真实边界框j的IoU
    # 初始化分配映射：-1 表示锚框未分配真实框，形状(num_anchors,)，每个元素表示对应锚框分配的真实框索引
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # max_ious：每个锚框与所有真实框的最大IoU值，ndices：每个锚框对应的最佳真实框索引
    max_ious, indices = torch.max(jaccard, dim=1)
    # 根据阈值决定是否分配真实边界框，找出所有最大IoU ≥ 阈值的锚框索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold] # 锚框对应真实框索引
    anchors_bbox_map[anc_i] = box_j # 符合条件的锚框分配给对应的真实框
    col_discard = torch.full((num_anchors,), -1) # 用于标记已分配的列（真实框）
    row_discard = torch.full((num_gt_boxes,), -1) # 用于标记已分配的行（锚框）
    for _ in range(num_gt_boxes):
        # 当前IoU矩阵中最大值索引(扁平化后索引)
        max_idx = torch.argmax(jaccard)
        # 关键计算：从扁平化索引还原行列索引
        # box_idx：真实框索引 = max_idx % num_gt_boxes
        # anc_idx：锚框索引 = max_idx / num_gt_boxes
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # 将当前最佳匹配的锚框分配给对应的真实框
        anchors_bbox_map[anc_idx] = box_idx
        # 关键步骤：标记已分配的位置，确保每个真实框和锚框只被分配一次
        jaccard[:, box_idx] = col_discard # 该真实框对应的所有列设为-1
        jaccard[anc_idx, :] = row_discard # 该锚框对应的所有行设为-1
    return anchors_bbox_map # 返回分配映射，形状 (num_anchors,)

# 给定框A和B，中心坐标分别为(xa, ya)和(xb, yb)，宽度为wa和wb，高度为ha和hb，将A的偏移量标记为：
# 详见图(13.4.3)，其中常量的默认值为 µx = µy = µw = µh = 0, σx = σy = 0.1 ，σw = σh = 0.2
def offset_boxes(anchors, assigned_bb, eps=1e-6): # @save
    # 计算锚框到对应真实边界框的偏移量
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    # 计算中心坐标归一化偏移；c_assigned_bb[:, :2] - c_anc[:, :2]：真实框中心与锚框中心的差值 (Δx, Δy)
    # / c_anc[:, 2:]：除以锚框的宽高进行归一化，得到相对偏移；10 *：缩放因子，对应注释中的 σx = σy = 0.1
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    # 计算宽高的对数比例偏移；c_assigned_bb[:, 2:] / c_anc[:, 2:]：真实框与锚框的宽高比例
    # .log(eps + ...)：取对数，将乘法关系转换为加法关系；5 *：缩放因子，对应 σw = σh = 0.2
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    # 中心点偏移和尺寸偏移拼接，输出[offset_x, offset_y, offset_w, offset_h]
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset # 形状(num_anchors, 4)

# 如果锚框没有分配真实边界框，需将锚框类别标记为背景。背景类锚框称为负类锚框
# 使用真实边界框实现multibox_target函数，来标记锚框的类别和偏移量
def multibox_target(anchors, labels): # @save
    # 功能：为锚框生成训练所需的标签（类别、偏移量、掩码）
    # anchors：锚框，形状通常为 (1, num_anchors, 4)
    # labels：真实标签，形状为 (batch_size, num_objects, 5)
    # 每个物体的格式：[class, xmin, ymin, xmax, ymax]
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    # 初始化三个列表，分别存储：偏移量；掩码（区分正负样本）；类别标签
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :] # 获取当前样本的真实标签
        # 关键步骤：将锚框分配给真实边界框，label[:, 1:]：提取边界框坐标（去除类别信息）
        # 返回形状 (num_anchors,)，每个元素：≥0：分配给的真实框索引，-1：未分配（背景）
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # 生成偏移量掩码(只有正样本的偏移量参与损失计算)：布尔掩码，正样本为True
        # .unsqueeze(-1)：(num_anchors, 1)，复制4次，变为 (num_anchors, 4)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # class_labels：初始化所有锚框类别为0；assigned_bb：初始化所有分配边界框坐标为0
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchors_bbox_map >= 0) # 正样本锚框索引(分配真实框的锚框)
        bb_idx = anchors_bbox_map[indices_true] # 获取这些锚框对应的真实框索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # 正样本分配类别标签
        assigned_bb[indices_true] = label[bb_idx, 1:] # 为正样本分配对应的真实边界框坐标
        # 计算锚框到真实框的偏移量，只保留正样本的偏移量，负样本偏移量为0
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        # 将当前样本的结果添加到批次列表中，展平为一维，便于后续处理
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    # 将批次中的所有样本堆叠起来，最终形状：
    # bbox_offset: (batch_size, num_anchors × 4)
    # bbox_mask: (batch_size, num_anchors × 4)
    # class_labels: (batch_size, num_anchors)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

fig = d2l.plt.imshow(img)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88,
                         0.98], [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))



# 非极大值抑制预测边界框
# offset_inverse函数，该函数将锚框和偏移量预测
# 作为输入，并应用逆偏移变换来返回预测的边界框坐标
def offset_inverse(anchors, offset_preds): # @save
    # 根据带有预测偏移量的锚框来预测边界框，是 offset_boxes 函数的逆操作
    # anchors：锚框坐标，形状 (N, 4)，格式 [xmin, ymin, xmax, ymax]
    # offset_preds：预测的偏移量，形状 (N, 4)，格式 [Δx, Δy, Δw, Δh]
    anc = d2l.box_corner_to_center(anchors)
    # 从偏移量还原实际中心坐标；/ 10：逆缩放，对应之前 offset_boxes 中的 ×10
    # anc[:, 2:]：锚框的宽高 [w, h]；+ anc[:, :2]：加上锚框的原始中心坐标
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    # 从偏移量还原实际宽高；/ 5：逆缩放，对应之前 offset_boxes 中的 ×5
    # torch.exp()：对应offset_boxes的log()；* anc[:, 2:]：乘以锚框原始宽高
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    # 预测中心坐标和宽高拼接，形状(N, 4)，[pred_cx,pred_cy,pred_w,pred_h]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

# 当有许多锚框时，会输出许多明显重叠的预测边界框，都围绕同一目标
# 为简化输出，可使用非极大值抑制合并属于同一目标的类似的预测边界框
# 非极大值抑制的工作原理：对于预测边界框B，目标检测模型会计算每个类别的预测概率
# 假设最大的预测概率为p，则概率对应类别B_为预测类别(p称为预测边界框B_的置信度)
# 同一图像中，所有预测的非背景边界框按置信度降序排序生成列表L，并按步骤操作列表L：
# 1.从L中选取置信度最高的预测边界框B1作为基准，将所有与B1的IoU超过预定阈值的非基准预测
# 边界框从L中移除。此时L保留了置信度最高的预测边界框，去除了与其太过相似的其他预测边界框
# 2.从L中选取置信度第二高预测边界框B2作为又一个基准，将所有与B2的IoU大于阈值的非基准预测边界框从L中移除
# 3.重复过程，直到L中所有预测边界框都被用作基准。L中任意一对预测边界框IoU都小于阈值，无一对边界框过于相似
# 4.输出列表L中的所有预测边界框
def nms(boxes, scores, iou_threshold): # @save
    # 功能：非极大值抑制，去除重叠度高的冗余边界框
    # scores：边界框的置信度分数，形状 (N,)
    # iou_threshold：超过此阈值的边界框会被抑制
    # nms函数按降序对置信度进行排序并返回其索引
    # 按置信度分数降序排序，返回索引，可从最高分的边界框开始处理
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = [] # 保留预测边界框的指标
    # 循环直到所有边界框都被处理完毕
    # B.numel()：返回B中元素的数量
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        # 关键计算：计算当前边界框与剩余所有边界框的IoU
        # boxes[i, :].reshape(-1, 4)：当前边界框，形状 (1, 4)
        # boxes[B[1:], :].reshape(-1, 4)：剩余边界框，形状 (len(B)-1, 4)
        # box_iou 返回形状 (1, len(B)-1)，.reshape(-1) 展平为 (len(B)-1,)
        iou = box_iou(boxes[i, :].reshape(-1, 4),boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 找出与当前边界框IoU ≤ 阈值的边界框索引，这些边界框可以保留
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # 更新剩余边界框列表：inds + 1：B[1:]相对于B的偏移量是1
        B = B[inds + 1] # 保留与当前边界框重叠度低的边界框
    return torch.tensor(keep, device=boxes.device) # 返回保留的边界框索引张量

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999): # @save
    # 定义multibox_detection函数来将非极大值抑制应用于预测边界框
    # cls_probs：类别概率(batch_size, num_classes, num_anchors)
    # offset_preds：偏移量预测，形状(batch_size, num_anchors * 4)
    # pos_threshold=0.009999999：正样本置信度阈值
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0) # 去除锚框的批次维度
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # 获取当前样本的类别概率和偏移量预测
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 关键步骤：计算每个锚框的最大置信度和对应类别；cls_prob[1:]：跳过背景类，只考虑物体类别
        # conf：最大置信度，形状 (num_anchors,)；class_id：对应的类别索引，形状 (num_anchors,)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred) # 锚框和预测偏移量转为实际边界框坐标
        keep = nms(predicted_bb, conf, nms_threshold) # 应用NMS，返回保留的边界框索引
        # 关键技巧：找出未被NMS保留的边界框
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx)) # combined：将保留索引和所有索引拼接
        uniques, counts = combined.unique(return_counts=True) # 找唯一值及出现次数
        # counts == 1：出现一次的值(未被保留边界框)，找到所有non_keep索引，将类设置为背景
        non_keep = uniques[counts == 1]
        # 重新排序：先NMS保留的边界框，然后是背景边界框
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1 # 非保留边界框标记为背景
        class_id = class_id[all_id_sorted] # 按新顺序重新排列类别ID
        # 按相同顺序重新排列置信度和边界框
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # 置信度阈值过滤：
        # conf < pos_threshold：找出置信度低于阈值的边界框
        # class_id[below_min_idx] = -1：将这些边界框标记为背景
        # conf[below_min_idx] = 1 - conf[below_min_idx]：对于背景框，置信度改为 1 - 原置信度
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        # 组装最终预测信息：
        # class_id.unsqueeze(1)：类别ID，形状 (num_anchors, 1)
        # conf.unsqueeze(1)：置信度，形状 (num_anchors, 1)
        # predicted_bb：边界框坐标，形状 (num_anchors, 4)
        # 最终形状：(num_anchors, 6)，每行：[class_id, confidence, xmin, ymin, xmax, ymax]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out) # 最终返回形状：(batch_size, num_anchors, 6)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
[0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4, # 背景的预测概率
[0.9, 0.8, 0.7, 0.1], # 狗的预测概率
[0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

output = multibox_detection(cls_probs.unsqueeze(dim=0),
offset_preds.unsqueeze(dim=0), anchors.unsqueeze(dim=0), nms_threshold=0.5)
fig = d2l.plt.imshow(img)
# 遍历批次中第一个样本的所有检测结果
for i in output[0].detach().numpy():
    if i[0] == -1: # 跳过背景检测
        continue
# 构建标签文本：i[0]：类别ID（0=狗，1=猫）；i[1]：置信度
label = ('dog=', 'cat=')[int(i[0])] + str(i[1]) 
show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
print("over!")
d2l.plt.show()
