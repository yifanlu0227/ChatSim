#!/bin/bash

# 定义源数据目录和目标数据目录
SOURCE_DIR="/home/yfl/workspace/ChatSim/data/waymo_multi_view"

# 遍历每个bad文件夹
bad_list=(
"segment-16646360389507147817_3320_000_3340_000_with_camera_labels"
)

for SCENE_NAME in ${bad_list[@]}; do
  # 定义源文件路径
  SRC_COLMAP_DIR="$SOURCE_DIR/$SCENE_NAME/colmap"

  # 删除colmap文件夹
  if [ -d "$SRC_COLMAP_DIR" ]; then
    rm -r "$SRC_COLMAP_DIR"
  else
    echo "Warning: $SRC_COLMAP_DIR does not exist."
  fi
done

for SCENE_NAME in ${bad_list[@]}; do
  # 定义源文件路径
  SRC_SCENE_DIR="$SOURCE_DIR/$SCENE_NAME"

  # 删除colmap文件夹
  if [ -d "$SRC_SCENE_DIR" ]; then
    bash run_colmap.sh $SRC_SCENE_DIR
  else
    echo "Warning: $SRC_SCENE_DIR does not exist."
  fi
done
