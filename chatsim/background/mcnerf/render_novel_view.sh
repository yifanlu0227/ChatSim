cmake . -B build
wait
cmake --build build --target main --config RelWithDebInfo -j
wait
python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=segment-17761959194352517553_5448_420_5468_420_with_camera_labels \
exp_name=exp_1029 mode=render_wide_angle is_continue=true +work_dir=$(pwd)

# python scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=render_path is_continue=true +work_dir=$(pwd)