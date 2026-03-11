# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:57:03 2025

@author: Administrator
"""
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset

import matplotlib.pyplot as plt
from matplotlib import rcParams

import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
torch.manual_seed(12046)
import numpy as np
import math
import torch
import joblib



rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  
torch.manual_seed(12046)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
   

        residual_blocks = []
        for _ in range(num_residual):
            residual_blocks.append(ResidualBlock(64, dropout_rate))
        self.res_layers = nn.Sequential(*residual_blocks)
        
     
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_encoder = PositionalEncoding(d_model=64, max_len=8*8)
        
      
        self.global_pool = nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Linear(64, 101)  

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,16,16)
        

        x = self.res_layers(x)  # (B,64,16,16)
        

        x = self.downsample(x)  # (B,64,8,8)
        
        B, C, H, W = x.shape  # H=W=8

        x = x.view(B, C, H * W)   # (B,64,64)
        x = x.permute(2, 0, 1)    # (64, B, 64)

        x = self.pos_encoder(x)   # (64, B, 64)

        x = self.transformer_encoder(x)  # (64, B, 64)
        

        x = x.permute(1, 2, 0)    # (B,64,64)
        x = self.global_pool(x)   # (B,64,1)
        x = x.squeeze(-1)         # (B,64)
        

        x = self.fc(x)            # (B,100)
        return x




class SpectralDataset(Dataset):
    def __init__(self, sample_path, g_matrix_path, save_scalers=True):
        data = sio.loadmat(sample_path)['sample']  #
        labels = sio.loadmat(g_matrix_path)['G_matrix']  # 
        
        # 16×16 Patch
        patch_size = 16
        num_patches = 160 // patch_size  # 10
        

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.data = torch.tensor(self.scaler_x.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape), dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 100, 100)
        self.labels = torch.tensor(self.scaler_y.fit_transform(labels), dtype=torch.float32)  # (B, 101)
        if save_scalers:
      
            joblib.dump(self.scaler_x, 'scaler_xx.pkl')
            joblib.dump(self.scaler_y, 'scaler_yx.pkl')

        B, I ,H ,W = self.data.shape

        patches = self.data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        self.data = patches.permute(2, 3, 0, 1, 4, 5).reshape(num_patches * num_patches, B, 1, patch_size, patch_size)

        print(f"切割后每个 sample_n 形状: {self.data.shape}")  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SpectralDataset('sample_prepare_divi_new0511.mat', 'G_matrix_prepare__divi_new0511.mat')
# %%
patches = dataset.data

# %% train
for n in range(100):
    print(f"训练第 {n+1}/100 个模型...")



    model = ConvTransformerNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []

    sample_n = []
    sample_n = patches[n]  # (B, 1, 16, 16)
    sample_n = sample_n.to(device)


    target_Gmatrix = []
    target_Gmatrix = dataset.labels # 目标 (B, 100)

    target_Gmatrix = target_Gmatrix.to(device)
    dataset_n = []
    dataset_n = TensorDataset(sample_n, target_Gmatrix)

    train_size = int(0.8 * len(dataset_n))
    val_size = len(dataset_n) - train_size


    train_dataset, val_dataset = random_split(dataset_n, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    best_val_loss = float('inf')
    best_model_wts = None

    num_epochs = 1000
      
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
        
 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict() 
            print('保存最佳模型')
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        

        if avg_val_loss < 0.01:
            print("验证集损失低于0.01，提前终止训练。")
            break

    torch.save(best_model_wts, f"models4/model_{n}.pth")
print("所有 100 个模型训练完成 ✅")




patchest = patches[:, 19, :, :, :].unsqueeze(1)  # 只取一个样本 
Gmatrices = []
for n1 in range(100):
    model = ConvTransformerNet().to(device)
    model.load_state_dict(torch.load(f"models4/model_{n1}.pth"))
    model.eval()

    with torch.no_grad():
        Gmatrix_n = model(patchest[n1])  # (1, 100)
        Gmatrices.append(Gmatrix_n)
    print(f"Epoch [{n1+1}/{100}]")

preds = []
scaler_yx = joblib.load("scaler_yx.pkl")
for Gmatrix_n in Gmatrices:
    # 将 tensor 转换为 numpy 数组，并逆标准化
    pred_numpy = Gmatrix_n.cpu().detach().numpy()
    pred_original = scaler_yx.inverse_transform(pred_numpy)
    preds.append(pred_original) 
# %%


preds_numpy = np.concatenate(preds, axis=0)  


preds_reshaped = preds_numpy.reshape(10, 10, 101)

from scipy.io import savemat
savemat("preds_reshaped.mat", {
    "preds_reshaped": preds_reshaped,
})




plt.imshow(preds_reshaped[:, :, 38], cmap='viridis', vmin=0, vmax=1)  
plt.colorbar()  
plt.title('Channel 20')
plt.show()



actuals = dataset.labels
actuals = actuals.cpu().detach().numpy()
actuals_original = scaler_yx.inverse_transform(actuals)
#actuals = np.concatenate(actuals, axis=0)
plt.figure()
plt.plot(actuals_original[19], label='实际光谱')
from scipy.io import savemat
savemat("realvalue.mat", {
    "realvalue": actuals_original,
})



preds_numpy = np.concatenate(preds, axis=0) 

min_val = np.min(preds_numpy)
max_val = np.max(preds_numpy)

if max_val > min_val: 
    preds_numpy = (preds_numpy - min_val) / (max_val - min_val)


preds_numpy = preds_numpy.reshape(100, 100)

