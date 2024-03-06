import os
import argparse

scene_setting_dict = {
    'segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-13469905891836363794_4429_660_4449_660_with_camera_labels.tfrecord': {'start_frame': 40, 'frame_nums': 40},
    'segment-14333744981238305769_5658_260_5678_260_with_camera_labels.tfrecord': {'start_frame': 40, 'frame_nums': 40},
    'segment-1172406780360799916_1660_000_1680_000_with_camera_labels.tfrecord': {'start_frame': 50, 'frame_nums': 40},
    'segment-4058410353286511411_3980_000_4000_000_with_camera_labels.tfrecord': {'start_frame': 90, 'frame_nums': 40},
    'segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord': {'start_frame': 30, 'frame_nums': 40},
    'segment-14869732972903148657_2420_000_2440_000_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-16646360389507147817_3320_000_3340_000_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-13238419657658219864_4630_850_4650_850_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-14424804287031718399_1281_030_1301_030_with_camera_labels.tfrecord': {'start_frame': 60, 'frame_nums': 40},
    'segment-15270638100874320175_2720_000_2740_000_with_camera_labels.tfrecord': {'start_frame': 60, 'frame_nums': 40},
    'segment-15349503153813328111_2160_000_2180_000_with_camera_labels.tfrecord': {'start_frame': 100, 'frame_nums': 40},
    'segment-15868625208244306149_4340_000_4360_000_with_camera_labels.tfrecord': {'start_frame': 110, 'frame_nums': 40},
    'segment-16608525782988721413_100_000_120_000_with_camera_labels.tfrecord': {'start_frame': 10, 'frame_nums': 40},
    'segment-17761959194352517553_5448_420_5468_420_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-3425716115468765803_977_756_997_756_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-3988957004231180266_5566_500_5586_500_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-9385013624094020582_2547_650_2567_650_with_camera_labels.tfrecord': {'start_frame': 130, 'frame_nums': 40},
    'segment-8811210064692949185_3066_770_3086_770_with_camera_labels.tfrecord': {'start_frame': 30, 'frame_nums': 40},
    'segment-10275144660749673822_5755_561_5775_561_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-10676267326664322837_311_180_331_180_with_camera_labels.tfrecord': {'start_frame': 100, 'frame_nums': 40},
    'segment-12879640240483815315_5852_605_5872_605_with_camera_labels.tfrecord': {'start_frame': 20, 'frame_nums': 40},
    'segment-13142190313715360621_3888_090_3908_090_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-13196796799137805454_3036_940_3056_940_with_camera_labels.tfrecord': {'start_frame': 70, 'frame_nums': 40},
    'segment-14348136031422182645_3360_000_3380_000_with_camera_labels.tfrecord': {'start_frame': 140, 'frame_nums': 40},
    'segment-15365821471737026848_1160_000_1180_000_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-16470190748368943792_4369_490_4389_490_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-13085453465864374565_2040_000_2060_000_with_camera_labels.tfrecord': {'start_frame': 110, 'frame_nums': 40},
    'segment-14004546003548947884_2331_861_2351_861_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
    'segment-15221704733958986648_1400_000_1420_000_with_camera_labels.tfrecord': {'start_frame': 70, 'frame_nums': 40},
    'segment-16345319168590318167_1420_000_1440_000_with_camera_labels.tfrecord': {'start_frame': 0, 'frame_nums': 40},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_data_dir", type=str, help="The original waymo data path")
    parser.add_argument("--nerf_data_dir", type=str, help="The path used to save the pre-processed data")
    args = parser.parse_args()
    
    downloaded_files = os.listdir(args.waymo_data_dir)

    for downloaded_file in downloaded_files:
        if downloaded_file not in scene_setting_dict:
            continue
        tfrecord_dir = os.path.join(args.waymo_data_dir, downloaded_file)
        scene_setting = scene_setting_dict[downloaded_file]
        start_frame = scene_setting['start_frame']
        frame_nums = scene_setting['frame_nums']
        os.system(f'python process_waymo.py --tfrecord_dir "{tfrecord_dir}" --nerf_data_dir {args.nerf_data_dir} --start_frame {start_frame} --frame_nums {frame_nums}')
