import os
import random
import shutil

# 定义源文件夹和目标文件夹
src_folder = "/home/yfl/workspace/HDRi_download/hdri_1k_copy" # 'path_to_your_folder'
train_folder = "/home/yfl/workspace/HDRi_download/train" # 'path_to_train_folder'
test_folder = "/home/yfl/workspace/HDRi_download/val" # 'path_to_test_folder'

# 列出文件夹中的所有图片
all_images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
random.shuffle(all_images)

# 计算80%的分割点
split_point = int(0.8 * len(all_images))

# 划分图片为训练集和测试集
train_images = all_images[:split_point]
test_images = all_images[split_point:]

# 将图片移动到相应的文件夹
for image in train_images:
    shutil.move(os.path.join(src_folder, image), os.path.join(train_folder, image))

for image in test_images:
    shutil.move(os.path.join(src_folder, image), os.path.join(test_folder, image))
