import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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

# 加载已保存的模型
input_size = 63  # 21个landmark的x, y, z三个坐标
hidden_size = 128
num_classes = 18  # 假设有18种手势类别，需要根据实际训练类别数更新
model = MLP(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("hand_gesture_model.pth"))
model.eval()

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 初始化OpenCV视频捕获
cap = cv2.VideoCapture(0)

# 手势标签
gesture_labels = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

def preprocess_landmarks(landmarks):
    """对21个关键点的坐标进行归一化处理"""
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    z_coords = [lm.z for lm in landmarks]

    # 归一化处理
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)

    x_normalized = [(x - x_min) / (x_max - x_min) if x_max > x_min else 0 for x in x_coords]
    y_normalized = [(y - y_min) / (y_max - y_min) if y_max > y_min else 0 for y in y_coords]
    z_normalized = [(z - z_min) / (z_max - z_min) if z_max > z_min else 0 for z in z_coords]

    # 合并所有归一化后的坐标
    return np.array(x_normalized + y_normalized + z_normalized, dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将图像转换为RGB格式并处理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # 检测到手部关键点时处理
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取并归一化关键点
            landmarks_array = preprocess_landmarks(hand_landmarks.landmark)
            landmarks_tensor = torch.tensor(landmarks_array).unsqueeze(0)  # 转换为张量并增加batch维度
            
            # 使用模型进行预测
            with torch.no_grad():
                output = model(landmarks_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            # 显示结果
            gesture_name = gesture_labels[predicted_class]
            cv2.putText(frame, f'Gesture: {gesture_name} ({confidence:.2f})',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 绘制手部关键点
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 显示摄像头画面
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
hands.close()
