import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from d2l import torch as d2l


""" 
# 前向传播：按顺序计算和存储神经网络中每层的结果
# 反向传播：计算神经网络参数梯度的方法
# 梯度爆炸问题：参数更新过大，破坏了模型的稳定收敛
# 梯度消失问题：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习
# Xavier初始化：让每一层输出方差尽可能等于输入方差，避免出现梯度消失或爆炸问题
# 数学原理：w 权重分布应符合：N(均值：0, 方差：sqrt(2 / (n_in + n_out)))

# 分布偏移：在许多情况下，训练集和测试集并不来自同一个分布
# 真实风险是从真实分布中抽取的所有数据的总体损失的预期
# 经验风险是训练数据的平均损失，用于近似真实风险
# 在相应的假设条件下，可以在测试时检测并纠正协变量偏移和标签偏移

# 以 sigmoid 函数为例说明梯度消失
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
# d2l.plt.show()

# 以下例子说明梯度爆炸
M = torch.normal(0, 1, size=(4,4))
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))
# print('乘以100个矩阵后\n', M)
"""



# 协变量偏移检测器(了解即可)
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')


class CovariateShiftDetector:
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        """
        协变量偏移检测器

        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout: dropout比率
        """
        self.input_dim = input_dim
        self.model = self._build_model(hidden_dims, dropout)
        self.scaler = None

    def _build_model(self, hidden_dims, dropout):
        """构建检测网络"""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def create_shifted_data(self, X_source, shift_type='mean', shift_factor=1.0):
        """
        创建有偏移的测试数据（用于演示）

        参数:
            X_source: 源数据
            shift_type: 偏移类型 ['mean', 'scale', 'noise', 'subset']
            shift_factor: 偏移程度
        """
        X_target = X_source.copy()

        if shift_type == 'mean':
            # 均值偏移
            X_target += shift_factor * np.random.randn(*X_target.shape)

        elif shift_type == 'scale':
            # 尺度偏移
            X_target *= (1 + shift_factor * np.random.randn(X_target.shape[1]))

        elif shift_type == 'noise':
            # 添加噪声
            noise = shift_factor * np.random.randn(*X_target.shape)
            X_target = 0.8 * X_target + 0.2 * noise

        elif shift_type == 'subset':
            # 特征子集偏移
            n_features = X_target.shape[1]
            mask = np.random.choice([0, 1], size=n_features,
                                    p=[0.3, 0.7])
            X_target = X_target * mask

        return X_target

    def prepare_data(self, X_train, X_test, val_size=0.2):
        """
        准备训练数据：标记训练数据为0，测试数据为1

        参数:
            X_train: 训练数据
            X_test: 测试数据
            val_size: 验证集比例
        """
        # 创建标签
        y_train = np.zeros(len(X_train))  # 训练数据标记为0
        y_test = np.ones(len(X_test))  # 测试数据标记为1

        # 合并数据
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        # 打乱数据
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]

        # 划分训练验证集
        X_train_detect, X_val_detect, y_train_detect, y_val_detect = train_test_split(
            X_combined, y_combined, test_size=val_size, random_state=42, stratify=y_combined
        )

        return (X_train_detect, y_train_detect), (X_val_detect, y_val_detect)

    def train(self, X_train, X_test, epochs=100, batch_size=64, lr=0.001):
        """
        训练协变量偏移检测器

        返回:
            auc_score: 验证集AUC分数
            history: 训练历史
        """
        # 准备数据
        (X_train_detect, y_train_detect), (X_val_detect, y_val_detect) = self.prepare_data(X_train, X_test)

        # 转换为Tensor
        X_train_t = torch.FloatTensor(X_train_detect)
        y_train_t = torch.FloatTensor(y_train_detect).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val_detect)
        y_val_t = torch.FloatTensor(y_val_detect).unsqueeze(1)

        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        # 训练
        history = {'train_loss': [], 'val_auc': []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            # 批量训练
            for i in range(0, len(X_train_t), batch_size):
                X_batch = X_train_t[i:i + batch_size]
                y_batch = y_train_t[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_preds = val_outputs.numpy()
                val_auc = roc_auc_score(y_val_t.numpy(), val_preds)

            history['train_loss'].append(epoch_loss / len(X_train_t))
            history['val_auc'].append(val_auc)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train_t):.4f}, Val AUC: {val_auc:.4f}')

        return history['val_auc'][-1], history

    def detect_shift(self, X_train, X_test, threshold=0.7):
        """
        检测是否存在显著的协变量偏移

        参数:
            threshold: AUC阈值，大于此值认为存在显著偏移

        返回:
            result: 检测结果字典
        """
        auc_score, history = self.train(X_train, X_test)

        result = {
            'auc_score': auc_score,
            'has_shift': auc_score > threshold,
            'shift_severity': min(1.0, max(0, (auc_score - 0.5) / 0.5)),  # 0-1之间的严重程度
            'confidence': 'High' if abs(auc_score - 0.5) > 0.3 else 'Low'
        }

        return result, history

    def plot_results(self, history, auc_score):
        """绘制训练结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(history['train_loss'], label='Training Loss', color='blue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        # AUC曲线
        ax2.plot(history['val_auc'], label='Validation AUC', color='red')
        ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random (No Shift)')
        ax2.axhline(y=auc_score, color='green', linestyle='--', label=f'Final AUC: {auc_score:.3f}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Covariate Shift Detection AUC')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 示例使用
def demo_covariate_shift_detection():
    """演示协变量偏移检测"""
    print("=== 协变量偏移检测演示 ===\n")

    # 生成示例数据
    X_source, y_source = make_classification(
        n_samples=1000, n_features=20, n_informative=15, random_state=42
    )

    # 创建检测器
    detector = CovariateShiftDetector(input_dim=20)

    # 测试1: 无偏移的情况
    print("1. 测试无偏移情况:")
    X_target_no_shift = X_source.copy()
    result_no_shift, history_no_shift = detector.detect_shift(X_source, X_target_no_shift)
    print(f"   AUC: {result_no_shift['auc_score']:.3f}")
    print(f"   存在偏移: {result_no_shift['has_shift']}")
    print(f"   严重程度: {result_no_shift['shift_severity']:.2f}")
    print(f"   置信度: {result_no_shift['confidence']}")

    # 测试2: 有偏移的情况
    print("\n2. 测试有偏移情况:")
    X_target_shift = detector.create_shifted_data(X_source, shift_type='mean', shift_factor=2.0)
    result_shift, history_shift = detector.detect_shift(X_source, X_target_shift)
    print(f"   AUC: {result_shift['auc_score']:.3f}")
    print(f"   存在偏移: {result_shift['has_shift']}")
    print(f"   严重程度: {result_shift['shift_severity']:.2f}")
    print(f"   置信度: {result_shift['confidence']}")

    # 绘制结果
    print("\n3. 可视化结果:")
    detector.plot_results(history_shift, result_shift['auc_score'])

    return result_no_shift, result_shift


# 运行演示
if __name__ == "__main__":
    demo_covariate_shift_detection()