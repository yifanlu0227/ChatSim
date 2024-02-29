import numpy as np
import trimesh

class camera_view_dir:
    def __init__(self, axis, pn):
        """
        axis : int
            0 -> x, 1 -> y, 2->z

        pn : 1 or -1
            positive or negative
        """
        self.axis = axis
        self.pn = pn

def visualize_poses(poses, camera_coord_axis_order='DRB', size=0.1):
    """
    Args:
        poses : numpy.ndarray
            shape [B, 3/4, 4]

        size : float
            size of axis

        camera_coord_axis_order : str
            https://zhuanlan.zhihu.com/p/593204605
            how camera coordinate's xyz related to the camera view
            For example, 'DRB' means x->down, y->right, z->back. 
            ======================
            OpenCV/Colmap: RDF
            LLFF: DRB
            OpenGL/NeRF: RUB
            Blender: RUB
            Mitsuba/Pytorch3D: LUF

    """
    # process the camera_coord_axis_order.
    # e.g. camera_front: (2, -1), means inverse(for -1) z-axis(for 2)
    try:
        camera_front = camera_view_dir(camera_coord_axis_order.index('F'), 1)
    except:
        camera_front = camera_view_dir(camera_coord_axis_order.index('B'), -1)
    try:
        camera_right = camera_view_dir(camera_coord_axis_order.index('R'), 1)
    except:
        camera_right = camera_view_dir(camera_coord_axis_order.index('L'), -1)
    try:
        camera_up = camera_view_dir(camera_coord_axis_order.index('U'), 1)
    except:
        camera_up = camera_view_dir(camera_coord_axis_order.index('D'), -1)


    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if poses.shape[1] == 3:
        pad_values = np.array([0, 0, 0, 1.0])
        poses = np.pad(poses, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        poses[:, -1, :] = pad_values

    for pose in poses:
        # plot the camera coord axis 
        axes = trimesh.creation.axis(
            transform=pose,
            axis_length=size,
        )
        objects.append(axes)

        # plot the camera view
        pos = pose[:3, 3]

        up_left = pos + camera_front.pn * size * pose[:3, camera_front.axis] \
                      + camera_up.pn * size * pose[:3, camera_up.axis] \
                      - camera_right.pn * size * pose[:3, camera_right.axis]

        up_right = pos + camera_front.pn * size * pose[:3, camera_front.axis] \
                       + camera_up.pn * size * pose[:3, camera_up.axis] \
                       + camera_right.pn * size * pose[:3, camera_right.axis]

        down_left = pos + camera_front.pn * size * pose[:3, camera_front.axis] \
                        - camera_up.pn * size * pose[:3, camera_up.axis] \
                        - camera_right.pn * size * pose[:3, camera_right.axis]

        down_right = pos + camera_front.pn * size * pose[:3, camera_front.axis] \
                         - camera_up.pn * size * pose[:3, camera_up.axis] \
                         + camera_right.pn * size * pose[:3, camera_right.axis]


        dir = (up_left + up_right + down_left + down_right) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 2

        up_middle = (up_left + up_right) / 2

        segs = np.array([[pos, up_left], [pos, up_right], [pos, down_left], [pos, down_right], 
                         [up_left, up_right], [up_right, down_right], [down_right, down_left], [down_left, up_left], 
                         [pos, o], [pos, up_middle]])
        segs = trimesh.load_path(segs)
        
        objects.append(segs)

    trimesh.Scene(objects).show()



if __name__=="__main__":
    # waymo_data_path = "waymo/poses_bounds.npy" # llff format
    # waymo_ext_int = np.load(waymo_data_path)[:, :15].reshape(-1, 3, 5)
    # waymo_ext = waymo_ext_int[:,:3,:4]
    # waymo_int = waymo_ext_int[:,:3, 4]
    # print(waymo_ext)
    # visualize_poses(waymo_ext)

    waymo_data_path = "waymo/cams_meta.npy" # nerf format
    waymo_ext_int = np.load(waymo_data_path)[:, :12].reshape(-1, 3, 4)
    waymo_ext = waymo_ext_int[:,:3,:4]
    print(waymo_ext)
    visualize_poses(waymo_ext, "RUB")