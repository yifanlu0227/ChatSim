cmake . -B build
wait
cmake --build build --target main --config RelWithDebInfo -j
wait
python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=segment-14424804287031718399_1281_030_1301_030_with_camera_labels \
exp_name=exp_coeff_0.15 dataset.shutter_coefficient=0.15 mode=train_hdr_shutter +work_dir=$(pwd) 