SCENE_NAME=segment-1172406780360799916_1660_000_1680_000_with_camera_labels

python train.py --config configs/chatsim/original.yaml source_path=data/waymo_multi_view/${SCENE_NAME}/colmap/sparse_undistorted model_path=output/${SCENE_NAME}