cmake . -B build
wait
cmake --build build --target main --config RelWithDebInfo -j
wait
python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=segment-12879640240483815315_5852_605_5872_605_with_camera_labels \
exp_name=old_calib_2999_calib_2999 dataset.shutter_coefficient=0.15 mode=train_hdr_shutter +work_dir=$(pwd) 