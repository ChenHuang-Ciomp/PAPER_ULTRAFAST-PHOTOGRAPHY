# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:48:09 2025

@author: Administrator
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:25:19 2025

@author: apple
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
import joblib
# 设置字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
torch.manual_seed(12046)
# 设备选择，支持 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 数据集定义

# 数据增强：扩增样本数量，并进行标准化
class SpectralDataset(Dataset):
    def __init__(self, sample_path, g_matrix_path, augment_factor=30, save_scalers=True):
        data = sio.loadmat(sample_path)['sample']  # (B, 100, 100)
        labels = sio.loadmat(g_matrix_path)['G_matrix']  # (B, 100)
        
        
        

        # 数据标准化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 对数据进行标准化
        self.data = torch.tensor(self.scaler_x.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape), dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 100, 100)
        self.labels = torch.tensor(self.scaler_y.fit_transform(labels), dtype=torch.float32).to(device)  # (B, 100)
        
        self.augment_factor = augment_factor  # 乘法因子以扩充数据
        if save_scalers:
            # 保存标准化参数
            joblib.dump(self.scaler_x, 'scaler_x.pkl')
            joblib.dump(self.scaler_y, 'scaler_y.pkl')
        # 生成扩增数据
        augmented_data, augmented_labels = [], []
        for i in range(augment_factor):
            scale = np.random.uniform(0.8, 1.0)  # 乘以 0.91, 0.915, 0.92 ...
            augmented_data.append(self.data * scale )
            augmented_labels.append(self.labels * scale)
        
        self.data = torch.cat(augmented_data, dim=0)
        self.labels = torch.cat(augmented_labels, dim=0)
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



# 2. 模型定义：残差块、Transformer 模块、整体网络
# 改进后的残差块：预激活 Residual Block，并可选加入 dropout
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out += identity
        return out

# 位置编码模块：对 Transformer 输入加入位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, B, d_model)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x

class ConvTransformerNet(nn.Module):
    def __init__(self, num_residual=3, num_transformer_layers=2, nhead=8, dropout_rate=0.1):
        super(ConvTransformerNet, self).__init__()
        # 初始卷积层：将 1 通道映射到 64 通道，保持 16x16 的尺寸
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输出: (B,64,16,16)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # 为了适应小尺寸，去掉 maxpool
        
        # 残差块堆叠
        residual_blocks = []
        for _ in range(num_residual):
            residual_blocks.append(ResidualBlock(64, dropout_rate))
        self.res_layers = nn.Sequential(*residual_blocks)
        
        # 下采样层：为了减小 Transformer 序列长度，这里下采样到 (B,64,8,8)
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 输出: (B,64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Transformer部分：将二维特征图展平为序列，特征维度为 64
        # 这里的特征图大小为 8x8，因此序列长度为 64
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_encoder = PositionalEncoding(d_model=64, max_len=8*8)
        
        # 全局平均池化后输出
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 对序列维度做全局平均池化
        self.fc = nn.Linear(64, 101)  # 输出维度 101

    def forward(self, x):
        # 初始卷积
        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,16,16)
        
        # 残差层
        x = self.res_layers(x)  # (B,64,16,16)
        
        # 下采样
        x = self.downsample(x)  # (B,64,8,8)
        
        B, C, H, W = x.shape  # H=W=8
        # 将特征图 reshape 成序列： (seq_len, B, d_model)
        x = x.view(B, C, H * W)   # (B,64,64)
        x = x.permute(2, 0, 1)    # (64, B, 64)
        
        # 加入位置编码
        x = self.pos_encoder(x)   # (64, B, 64)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # (64, B, 64)
        
        # 全局平均池化：先将维度调整回来 (B,64,64)
        x = x.permute(1, 2, 0)    # (B,64,64)
        x = self.global_pool(x)   # (B,64,1)
        x = x.squeeze(-1)         # (B,64)
        
        # 全连接层输出 101 维
        x = self.fc(x)            # (B,101)
        return x

# 3. 准备数据集与 DataLoader
# 请根据实际的.mat文件路径修改下面的文件名

dataset = SpectralDataset('sample_no_divi_new3.mat', 'G_matrix_no__divi_new3.mat')
# 按80%/20%划分数据集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# 4. 模型、损失和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvTransformerNet().to(device)
criterion = nn.MSELoss()   # 根据任务选择合适的损失函数，例如回归任务常用 MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练过程
num_epochs = 1000
train_losses = []
val_losses = []

# 最佳验证集损失初始化为一个很大的值
best_val_loss = float('inf')
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 如果当前的验证损失比最小验证损失还小，保存模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = model.state_dict()  # 保存当前最佳模型的权重
        print('保存最佳模型')
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    # 终止训练条件
    #if avg_val_loss < 0.001:
    #    print("验证集损失低于0.01，提前终止训练。")
    #    break

torch.save(best_model_wts, "ConvTransformerNetmodel0511.pth") # 保存模型参数
def smooth_curve(points, factor=0.9):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

# 绘制平滑后的损失曲线
plt.figure(figsize=(8,6))
plt.plot(range(1, len(train_losses) + 1), smooth_curve(train_losses), label="Train Loss ")
plt.plot(range(1, len(val_losses) + 1), smooth_curve(val_losses), label="Validation Loss ")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# 训练完后加载模型
loaded_model = ConvTransformerNet().to(device)
loaded_model.load_state_dict(torch.load('ConvTransformerNetmodel0511.pth'))
loaded_model.eval()

# 验证集上的预测
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = loaded_model(inputs)
        predictions.append(outputs.cpu().numpy())  # 确保输出张量先移到 CPU
        actuals.append(targets.cpu().numpy())  # 确保目标张量先移到 CPU

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

# 反标准化
scaler_y = joblib.load('scaler_y.pkl')
predictions = scaler_y.inverse_transform(predictions)
actuals = scaler_y.inverse_transform(actuals)
# 计算均方误差（MSE）、平均绝对误差（MAE）和决定系数（R²）
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f'验证集 MAE: {mae:.6f}, MSE: {mse:.6f}, R²: {r2:.6f}')

# 可视化部分样本的预测结果
# 选前6个样本可视化
num_samples_to_plot = min(6, len(predictions))

plt.figure(figsize=(10, 6))
for i in range(num_samples_to_plot):
    plt.plot(actuals[i], label=f'实际{i+1}', linestyle='-')
    plt.plot(predictions[i], label=f'预测{i+1}', linestyle='--')
plt.xlabel('波长索引')
plt.ylabel('强度')
plt.title('前6个样本的光谱预测 vs 实际')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.io import savemat
savemat("spectral_predictions.mat", {
    "predictions": predictions,
    "actuals": actuals
})

