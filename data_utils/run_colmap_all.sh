#!/bin/bash

# 定义数据目录
DATA_DIR="../data/waymo_multi_view"

# 遍历数据目录中的每个文件夹
for SCENE_NAME in $(ls $DATA_DIR); do
  # 执行 run_colmap.sh 脚本
  bash run_colmap.sh "$DATA_DIR/$SCENE_NAME"
done
