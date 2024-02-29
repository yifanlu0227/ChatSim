import click
import numpy as np
from os.path import join as pjoin

@click.command()
@click.option('--data_dir', type=str)
def hello(data_dir):
    poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)
    np.save(pjoin(data_dir, 'poses_render.npy'), poses)

    last_row = np.zeros((poses.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    poses = np.concatenate((poses, last_row), axis = 1)
    extrinsic_opencv = np.concatenate(
        (
            poses[:, :, 0:1],
            -poses[:, :, 1:2],
            -poses[:, :, 2:3],
            poses[:, :, 3:],
        ),
        axis = 2
    )

    np.save(pjoin(data_dir, 'extrinsics.npy'), extrinsic_opencv)



if __name__ == '__main__':
    hello()