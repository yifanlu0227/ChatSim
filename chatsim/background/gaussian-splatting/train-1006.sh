SCENE_NAME=segment-10061305430875486848_1080_000_1100_000_with_camera_labels

python train.py --config configs/chatsim/original.yaml source_path=data/waymo_multi_view/${SCENE_NAME}/colmap/sparse_undistorted model_path=output/${SCENE_NAME}