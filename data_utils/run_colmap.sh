#!/bin/bash
# modified from https://github.com/LightwheelAI/street-gaussians-ns/blob/main/scripts/shells/run_colmap.sh

DATASET_PATH=$1

python transform2colmap_chatsim.py \
    --input_path $DATASET_PATH \

colmap feature_extractor \
   --database_path $DATASET_PATH/colmap/database.db \
   --image_path $DATASET_PATH/images 

python inject_to_database.py \
    --input_path $DATASET_PATH 

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/colmap/database.db

mkdir $DATASET_PATH/colmap/sparse
mkdir $DATASET_PATH/colmap/sparse/not_align # not aligned to waymo's scale

colmap mapper \
    --database_path $DATASET_PATH/colmap/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/colmap/sparse/not_align \
    --Mapper.init_max_forward_motion 1.0 \
    --Mapper.init_min_tri_angle 0.5 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.multiple_models 1 \
    --Mapper.ba_global_max_num_iterations 30 \
    --Mapper.ba_global_images_ratio 1.3 \
    --Mapper.ba_global_points_ratio 1.3 \
    --Mapper.ba_global_images_freq 2000 \
    --Mapper.ba_global_points_freq 35000 \
    --Mapper.filter_min_tri_angle 0.1 \

# undistortion is necessary for gaussian splatting, but not McNeRF
colmap image_undistorter \
    --image_path "$DATASET_PATH"/images \
    --input_path "$DATASET_PATH"/colmap/sparse/not_align/0 \
    --output_path "$DATASET_PATH"/colmap/sparse_undistorted \
    --output_type COLMAP

# This will not help increate points too much
# mkdir $DATASET_PATH/colmap/sparse_undistorted/sparse_triangulate
# colmap point_triangulator \
#     --database_path $DATASET_PATH/colmap/database.db \
#     --image_path $DATASET_PATH/colmap/sparse_undistorted/images \
#     --input_path $DATASET_PATH/colmap/sparse_undistorted/sparse \
#     --output_path $DATASET_PATH/colmap/sparse_undistorted/sparse_triangulate \

python convert_to_waymo.py -d $DATASET_PATH -c colmap