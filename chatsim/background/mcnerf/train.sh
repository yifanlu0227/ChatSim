cmake . -B build
wait
cmake --build build --target main --config RelWithDebInfo -j
wait
python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=segment-11379226583756500423_6230_810_6250_810_with_camera_labels \
exp_name=exp_0302_coeff_0.1_start_5 dataset.shutter_coefficient=0.1 mode=train_hdr_shutter +work_dir=$(pwd) 