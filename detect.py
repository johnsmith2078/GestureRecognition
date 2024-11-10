import cv2
import mediapipe as mp
import os
import csv

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 遍历所有子文件夹
base_folder = '.'
for root, dirs, files in os.walk(base_folder):
    if not files:
        continue  # 跳过空文件夹

    if root == base_folder:
        continue    
    
    # 设置输出文件名为当前文件夹名.csv
    output_csv = os.path.join('./', os.path.basename(root) + '.csv')
    print(f"Processing {output_csv}")
    
    # 创建CSV文件并写入表头
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV表头
        header = ['image'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
        writer.writerow(header)
        
        # 处理当前文件夹内的每张图片
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # 过滤图像文件
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Could not read image {image_path}")
                    continue

                # 将图像转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 处理图像，获取手部关键点
                results = hands.process(image_rgb)

                # 如果检测到手部，获取关键点坐标
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 提取每个关键点的坐标
                        landmarks = [filename]
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        z_coords = [lm.z for lm in hand_landmarks.landmark]
                        
                        # 归一化处理
                        if x_coords:
                            x_min, x_max = min(x_coords), max(x_coords)
                            x_coords = [(x - x_min) / (x_max - x_min) if x_max > x_min else 0 for x in x_coords]
                        if y_coords:
                            y_min, y_max = min(y_coords), max(y_coords)
                            y_coords = [(y - y_min) / (y_max - y_min) if y_max > y_min else 0 for y in y_coords]
                        if z_coords:
                            z_min, z_max = min(z_coords), max(z_coords)
                            z_coords = [(z - z_min) / (z_max - z_min) if z_max > z_min else 0 for z in z_coords]

                        # 合并坐标并写入CSV
                        landmarks.extend(x_coords + y_coords + z_coords)
                        writer.writerow(landmarks)

# 释放资源
hands.close()
