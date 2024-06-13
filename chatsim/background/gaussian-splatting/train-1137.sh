SCENE_NAME=segment-11379226583756500423_6230_810_6250_810_with_camera_labels

python train.py --config configs/chatsim/original.yaml source_path=data/waymo_multi_view/${SCENE_NAME}/colmap/sparse_undistorted model_path=output/${SCENE_NAME}