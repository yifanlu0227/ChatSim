import numpy as np
import os
from os.path import join as pjoin

if __name__ == '__main__':
    data_dir = '/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/data/waymo_multi_view/segment-17761959194352517553_5448_420_5468_420_with_camera_labels'
    total_nums = 30
    radius = 0.2
    poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)

    poses = poses[::24][1:2]
    center_y, center_z = poses[0, 1, -1], poses[0, 2, -1]
    t = np.linspace(0, 2 * np.pi, total_nums)
    y = center_y + radius * np.cos(t)
    z = center_z + radius * np.sin(t)

    poses_render = np.tile(poses, (total_nums, 1, 1))
    poses_render[:, 1, -1] = y
    poses_render[:, 2, -1] = z
    print(poses_render)

    np.save(pjoin(data_dir, 'poses_render.npy'), poses_render)