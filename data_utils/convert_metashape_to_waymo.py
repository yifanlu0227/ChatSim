import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])


def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)


def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def read_xml(data_dir):
    print("Parsing Metashape results")
    intrinsic = intrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    extrinsic = extrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    extrinsic = np.stack(extrinsic[0],axis=0)

    poses = extrinsic
    poses = np.concatenate((-poses[:,:,1:2],poses[:,:,0:1],poses[:,:,2:3],poses[:,:,3:]),axis=-1)
    poses = poses[:,:3,:]
    poses_output = np.zeros((poses.shape[0],17))

    f = intrinsic[0][0,0]
    w = intrinsic[0][0,-1]*2
    h = intrinsic[0][1,-1]*2
    # import ipdb; ipdb.set_trace()
    hwf = np.array([h,w,f]).reshape(1,3,1).repeat(poses.shape[0],axis=0)

    bds = np.array([0.1,999]).reshape(1,-1).repeat(poses.shape[0],axis=0)
    poses = np.concatenate((poses,hwf),axis=-1)
    poses = poses.reshape(-1,15)
    poses = np.concatenate((poses,bds),axis=-1)
    np.save(os.path.join(data_dir, 'poses_bounds_metashape.npy'),poses)


def align(data_dir):
    print("Aligning Metashape's coordinates with Waymo's coordinates")
    poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_metashape.npy'))

    poses_bounds = poses_bounds[:,:15].reshape(-1,3,5)

    extrinsic = poses_bounds[:,:,:4]
    last_row = np.zeros((extrinsic.shape[0],1,4))
    last_row[:,:,-1] = 1
    extrinsic_metashape = np.concatenate((extrinsic,last_row),axis=1)  #####shape [n, 4, 4] extrinsic from metashape

    hwf = poses_bounds[0,:,-1]

    intrinsic = np.array([[hwf[2],0,hwf[1]*0.5+2.37],
                        [0,hwf[2],hwf[0]*0.5-1.89],
                        [0,0,1] ])

    poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_waymo.npy'))

    poses_bounds = poses_bounds[:,:15].reshape(-1,3,5)

    extrinsic = poses_bounds[:,:,:4]
    last_row = np.zeros((extrinsic.shape[0],1,4))
    last_row[:,:,-1] = 1
    extrinsic_waymo = np.concatenate((extrinsic,last_row),axis=1)   #####shape [n, 4, 4] extrinsic from waymo

    scale = np.linalg.norm(extrinsic_metashape[1,:3,-1] - extrinsic_metashape[0,:3,-1])  \
        / np.linalg.norm(extrinsic_waymo[1,:3,-1] - extrinsic_waymo[0,:3,-1])   ## unit length scale

    rotate_0_waymo = extrinsic_waymo[0,:3,:3]
    rotate_0_metashape = extrinsic_metashape[0,:3,:3]

    rotate_metashape_to_waymo = rotate_0_waymo @ np.linalg.inv(rotate_0_metashape)  #### the rotation matrix convert metashape to waymo axis
    rotate_metashape_to_waymo = rotate_metashape_to_waymo[None,...]

    extrinsic_results = np.zeros_like(extrinsic_metashape)   #final output

    extrinsic_results[:,:3,:3] = rotate_metashape_to_waymo @ extrinsic_metashape[:,:3,:3]


    delta_translation_in_metashape = extrinsic_metashape[:,:3,-1:] - extrinsic_metashape[0:1,:3,-1:]  ###delta translation between each frame and frame0 in metashape

    delta_translation_in_metashape = delta_translation_in_metashape


    delta_translation_in_waymo = (rotate_metashape_to_waymo @ delta_translation_in_metashape)  / scale ####convert to waymo axis

    extrinsic_results[:,:3,-1:] = delta_translation_in_waymo + extrinsic_waymo[0:1,:3,-1:]

    extrinsic_results[:,-1,-1] = 1
    poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds_metashape.npy'))
    poses_bounds_extrinsic_and_intrinsic = poses_bounds[:,:15].reshape(-1,3,5)
    poses_bounds_extrinsic_and_intrinsic[:,:,:-1] = extrinsic_results[:,:3,:]

    extrinsic_results_to_save = poses_bounds_extrinsic_and_intrinsic.reshape(-1,15)

    poses_bounds[:,:15] = extrinsic_results_to_save
    np.save(os.path.join(data_dir, 'poses_bounds.npy'), poses_bounds)       # LLFF

    print("Converting LLFF coordinates to NeRF coordinates")
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, 4]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    bounds = poses_bounds[:, 15: 17]
    n_poses = len(poses)
    intri = np.zeros([n_poses, 3, 3])
    intri[:, :3, :3] = np.eye(3)
    intri[:, 0, 0] = hwf[:, 2] 
    intri[:, 1, 1] = hwf[:, 2] 
    intri[:, 0, 2] = hwf[:, 1] * .5 
    intri[:, 1, 2] = hwf[:, 0] * .5

    data = np.concatenate([
        poses.reshape(n_poses, -1),
        intri.reshape(n_poses, -1),
        np.zeros([n_poses, 4]),
        bounds.reshape(n_poses, -1)
    ], -1)

    data = np.ascontiguousarray(np.array(data).astype(np.float64))
    np.save(os.path.join(data_dir, 'cams_meta.npy'), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', type = str)
    args,_ = parser.parse_known_args()

    read_xml(args.datadir)
    align(args.datadir)

    