waymo-open-dataset-tf-2-4-0==1.4.1

step1: Run `python process_waymo_script.py --waymo_data_dir=/path/to/your/tfrecord/files --nerf_data_dir=/path/to/nerf/data/dir`
step2: Using Metashape to calibrate images and get `camera.xml`
step3: Run `python python convert_metashape_to_waymo_script.py --datadir=/path/to/nerf/data/dir`
