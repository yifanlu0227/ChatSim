import os
import yaml
import numpy as np
import torch
import copy 
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import cv2
import json
import imageio.v2 as imageio
import collections
from termcolor import colored
from tqdm import tqdm

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def dump_yaml(data, savepath):
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def check_and_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_video(scene, prompt):
    video_output_path = os.path.join(scene.output_dir, scene.logging_name)
    check_and_mkdirs(video_output_path)
    filename = prompt.replace(' ', '_')[:40]
    fps = scene.fps
    print(colored("[Compositing video]", 'blue', attrs=['bold']), "start...")

    writer = imageio.get_writer(os.path.join(video_output_path, f"{filename}.mp4"), 
                                fps=fps)
    for frame in tqdm(scene.final_video_frames):
        writer.append_data(frame)
    writer.close()
    # save frames to folder
    check_and_mkdirs(os.path.join(video_output_path, f"{filename}"))
    for i,img in enumerate(scene.final_video_frames):
        imageio.imsave(os.path.join(video_output_path, f"{filename}/{i}.png"),img)
    
    if not scene.save_cache:
        scene.clean_cache()

    print(colored("[Compositing video]", 'blue', attrs=['bold']), "done.")

def transform_nerf2opencv_convention(extrinsic):
    """
    Transform and pad NeRF convention extrinsic (RUB) [3, 4] to
                      OpenCV convention extrisic (RDF) [4, 4].

    Args:
        extrinsic : np.ndarray
            shape [3, 4] in NeRF convention extrinsic (RUB)
    Returns:
        extrinsic_opencv : np.ndarray
            shape [4, 4] in OpenCV convention extrinsic (RDF)
    """

    all_ones = np.array([[0, 0, 0, 1]])
    extrinsic_opencv = np.concatenate((extrinsic, all_ones), axis=0)

    extrinsic_opencv = np.concatenate(
        (
            extrinsic_opencv[:, 0:1],
            -extrinsic_opencv[:, 1:2],
            -extrinsic_opencv[:, 2:3],
            extrinsic_opencv[:, 3:],
        ),
        axis = 1
    )

    return extrinsic_opencv


def rotate(point, angle):
    """Rotates a point around the origin by the specified angle in radians."""
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rotation_matrix, point)


def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car["cx"]
    y = car["cy"]
    z = car["cz"]
    length = car["length"]
    width = car["width"]
    height = car["height"]
    heading = car["heading"]
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])

    # The relative positions of the vertices from the box center before rotation.
    relative_positions = (
        np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        * half_dims
    )

    # Rotate each relative position and add the box center position.
    vertices = np.asarray(
        [rotate(pos, heading) + box_center for pos in relative_positions]
    )
    return vertices


def get_outlines(corners, extrinsic, intrinsic, height, width):  # return [height , width] mask
    def generate_convex_hull(points):
        hull = ConvexHull(points)
        return points[hull.vertices]

    def polygon_to_mask(polygon, height, width):
        # polygon = np.flip(polygon,axis=1)
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in polygon], outline=1, fill=1)
        mask = np.array(img)
        return mask

    all_one = np.ones((corners.shape[0], 1))

    points = np.concatenate((corners, all_one), axis=1).T
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]

    cam_points = cam_points / cam_points[2:]

    points = (intrinsic @ cam_points).T[:, :2]
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)

    # polygon = generate_convex_hull(points)

    # mask = polygon_to_mask(polygon.astype(int), height, width)


    mask = np.zeros((height, width))
    points = points.astype(int)
    y_min = max(points[:, 1].min() - 50, 0)
    y_max = min(points[:, 1].max() + 50, height)
    x_min = max(points[:, 0].min() - 50, 0)
    x_max = min(points[:, 0].max() + 50, width)

    mask[y_min:y_max, x_min:x_max] = 1
    return mask, [y_min, y_max, x_min, x_max]


def get_attributes_for_one_car(car, extrinsic, intrinsic):
    # return dict{
    #      'x', 'y', 'depth'
    # }
    x = car["cx"]
    y = car["cy"]
    z = car["cz"]
    one_point = np.array([[x, y, z]])
    all_one = np.ones((one_point.shape[0], 1))
    points = np.concatenate((one_point, all_one), axis=1).T
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]
    cam_points_without_norm = copy.copy(cam_points)
    cam_points = cam_points / cam_points[2:]
    points = (intrinsic @ cam_points).T[:, :2]
    return {
        "u": points[0, 0],
        "v": points[0, 1],
        "depth": cam_points_without_norm[-1, 0],
    }



def scale_dense_depth_map(dense_depth_map, sparse_depth_map, depth_map_mask):
    """
    Scale the dense depth map to match the scale of the sparse depth map.

    :param dense_depth_map: [H, W] dense depth map
    :param sparse_depth_map: [H, W] sparse depth map
    :param depth_map_mask: [H, W] mask indicating valid values in the sparse depth map
    :return: Scaled dense depth map
    """
    # 确保所有输入都是 PyTorch 张量
    dense_depth_map = torch.tensor(dense_depth_map, dtype=torch.float32)
    sparse_depth_map = torch.tensor(sparse_depth_map, dtype=torch.float32)
    depth_map_mask = torch.tensor(depth_map_mask, dtype=torch.float32)
    
    # 提取稠密和稀疏深度图中的有效点
    valid_dense_depths = torch.masked_select(dense_depth_map, depth_map_mask.bool())
    valid_sparse_depths = torch.masked_select(sparse_depth_map, depth_map_mask.bool())
    
    # 计算缩放因子 alpha
    alpha_numerator = torch.sum(valid_dense_depths * valid_sparse_depths)
    alpha_denominator = torch.sum(valid_dense_depths ** 2)
    alpha = alpha_numerator / alpha_denominator
    print('scaling factor: ', alpha)
    
    # 缩放稠密深度图
    scaled_dense_depth_map = alpha * dense_depth_map
    
    return scaled_dense_depth_map

def srgb_gamma_correction(linear_image):
    """
    linear_image: np.ndarray
        shape: H*W*C
    """
    linear_image = np.clip(linear_image, 0, 1)  # 将值限制在0到1之间
    gamma_corrected_image = np.where(linear_image <= 0.0031308,
                          linear_image * 12.92,
                          1.055 * (linear_image ** (1 / 2.4)) - 0.055)
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)  # 将值再次限制在0到1之间

    return gamma_corrected_image


def srgb_inv_gamma_correction(gamma_corrected_image):
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)  # 将值限制在0到1之间
    linear_image = np.where(gamma_corrected_image <= 0.04045,
                            gamma_corrected_image / 12.92,
                            ((gamma_corrected_image + 0.055) / 1.055)**2.4)
                            
    return linear_image


def parse_config(path_to_json):
    with open(path_to_json) as f:
        data = json.load(f)
        args = Struct(**data)
    return args

def blending_hdr_sky(nerf_env_panorama, sky_dome_panorama, nerf_last_trans, sky_mask):
    """
    blending hdr sky dome with nerf panorama
    Args:
        nerf_env_panorama : np.ndarray
            shape [H1, W1, 3], In Linear space

        sky_dome_panorama : np.ndarray
            shape [H2, W2, 3], In Linear space

        nerf_last_trans : np.ndarray
            shape [H1, W1, 1], range (0-1)
        
    """
    H, W, _ = sky_dome_panorama.shape
    
    sky_mask = cv2.resize(sky_mask, (W, H))[:,:,:1]
    nerf_env_panorama = cv2.resize(nerf_env_panorama, (W, H))
    nerf_last_trans = cv2.resize(nerf_last_trans, (W, H))[:, :, np.newaxis]

    nerf_last_trans[sky_mask>255*0.5] = 1

    # final_hdr_sky = srgb_inv_gamma_correction(nerf_env_panorama / 255.0) + \
    #                 sky_dome_panorama * nerf_last_trans
    final_hdr_sky = nerf_env_panorama + sky_dome_panorama * nerf_last_trans

    return final_hdr_sky

def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    direction = np.concatenate((-z, -x, y), axis = 1)
    return direction

def generate_rays(insert_x, insert_y, int_ext_path, nerf_exp_dir):
    # bounds = torch.jit.load(nerf_exp_dir + '/bounds_tmp.pt').state_dict()['0']
    near = 1e-2
    # far = 2.7716e+2
    far = 1000.
    
    origin = np.array([insert_x, insert_y, 0.0])
    cam_meta = np.load(int_ext_path)
    extrinsic = cam_meta[:,:12].reshape(-1,3,4)
    translation = extrinsic[:, :3, 3].copy()
    center = np.mean(translation, axis =0)
    bias = translation - center[None]
    radius = np.linalg.norm(bias, 2, -1, False).max()
    translation = (translation - center[None]) / radius
    extrinsic[:, :, 3] = translation

    origin = (origin - center) / radius

    dy = np.linspace(0, 1, 1280)
    dx = np.linspace(0, 1, 1280*4)
    u, v = np.meshgrid(dx, dy)
    u, v = u.ravel()[..., None], v.ravel()[..., None]

    rays_d = skylatlong2world(u, v)

    rays_o = np.tile(origin[None], (len(rays_d), 1))

    bounds = np.array([[near, far]]).repeat(len(rays_d), axis=0)

    np.save(os.path.join(nerf_exp_dir, 'rays_o.npy'), rays_o.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'rays_d.npy'), rays_d.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'bounds.npy'), bounds.astype(np.float32))



def getColorList():
    dict = collections.defaultdict(list)
 
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red']=color_list

    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list
 
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
 
    return dict
 

def get_color(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    max_num = 0
    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        img, cnts = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask_num =  binary[binary == 255].shape[0]
        if mask_num > max_num:
            max_num = mask_num
            color = d
    return color

def interpolate_uniformly(track, num_points):
    """
    Interpolates a given track to a specified number of points, distributing them uniformly.

    :param track: A numpy array of shape (n, d) where n is the number of points and d is the dimension.
    :param num_points: The number of points in the output interpolated track.
    :return: A numpy array of shape (num_points, d) representing the uniformly interpolated track.
    """
    # Calculate the cumulative distance along the track
    distances = np.cumsum(np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Include the start point

    # Generate the desired number of equally spaced distances
    max_distance = distances[-1]
    uniform_distances = np.linspace(0, max_distance, num_points)

    # Interpolate for each dimension
    uniform_track = np.array([np.interp(uniform_distances, distances, track[:, dim]) for dim in range(track.shape[1])])

    return uniform_track.T