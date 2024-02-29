import numpy as np
import os
from os.path import join as pjoin

if __name__ == '__main__':
    data_dir = '/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/data/waymo_multi_view/segment-11379226583756500423_6230_810_6250_810_with_camera_labels'

    poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)

    ##########################################################################
    # poses = poses[::24]
    # np.save(pjoin(data_dir, 'poses_render.npy'), poses)

    ##########################################################################
    poses = poses[::24][1]

    theta = 1.5               # 1.5 for 1287
    theta = theta/ 180 * np.pi
    T_theta = np.array([
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]
                    ])
    
    poses = np.matmul(T_theta, poses)
    print(poses.shape)

    np.save(pjoin(data_dir, 'poses_render.npy'), poses)
    ##########################################################################