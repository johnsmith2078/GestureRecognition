import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 设置数据文件夹路径
data_folder = './'  # 当前文件夹

# 读取所有csv文件的数据并合并
data_frames = []
labels = []
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        label = filename.split('.')[0]  # 假设文件名为类别名，例如"call.csv"表示"call"类
        df = pd.read_csv(os.path.join(data_folder, filename))
        data_frames.append(df)
        labels.extend([label] * len(df))  # 为每一行添加对应标签

# 合并所有数据
data = pd.concat(data_frames, ignore_index=True)
labels = np.array(labels)

# 提取特征和标签
X = data.iloc[:, 1:].values.astype(np.float32)  # 去掉图像名列，只保留特征数据
y = LabelEncoder().fit_transform(labels)  # 将标签编码为数值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为Tensor
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 设置超参数
input_size = X_train.shape[1]  # 输入维度
hidden_size = 128  # 隐藏层大小
num_classes = len(np.unique(y))  # 类别数
learning_rate = 0.001
num_epochs = 50

# 初始化模型、损失函数和优化器
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 保存模型
torch.save(model.state_dict(), "hand_gesture_model.pth")
