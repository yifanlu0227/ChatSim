from typing import List
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
import open3d as o3d
import tensorflow as tf

FILENAME = '/home/ubuntu/yifanlu/Chatsim2/ChatSim-release/data/waymo_tfrecords/1.4.2/segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

# Load only 2 frames. Note that using too many frames may be slow to display.
frames = []
count = 0
for data in dataset:
    frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    frames.append(frame)
    count += 1

print('#####################################################')


transform = np.reshape(np.array(frames[5].pose.transform), [4, 4])
transform = np.linalg.inv(transform)
road_edges = []
lanes = []
for i in range(len(frames[0].map_features)):
    if len(frames[0].map_features[i].lane.polyline) > 0:
        curr_lane = []
        for node in frames[0].map_features[i].lane.polyline:
            node_position = np.ones(4)
            node_position[0] = node.x
            node_position[1] = node.y
            node_position[2] = node.z
            curr_lane.append(node_position)
        curr_lane = np.stack(curr_lane)
        curr_lane = np.transpose(np.matmul(transform, np.transpose(curr_lane)))[:, 0:3]
        lanes.append(curr_lane)
    
    if len(frames[0].map_features[i].road_edge.polyline) > 0:
        curr_edge = []
        for node in frames[0].map_features[i].road_edge.polyline:
            node_position = np.ones(4)
            node_position[0] = node.x
            node_position[1] = node.y
            node_position[2] = node.z
            curr_edge.append(node_position)
        curr_edge = np.stack(curr_edge)
        curr_edge = np.transpose(np.matmul(transform, np.transpose(curr_edge)))[:, 0:3]
        road_edges.append(curr_edge)


x_min = -30
x_max = 50
y_min = -20
y_max = 20
cropped_road_edges = []
for edge in road_edges:
    new_road_edge = []
    for i in range(edge.shape[0]):
        if edge[i,0] < x_min or edge[i,0] > x_max or edge[i,1] < y_min or edge[i,1] > y_max:
            continue
        new_road_edge.append(edge[i])
    if len(new_road_edge) > 0:
        new_road_edge = np.stack(new_road_edge)
        cropped_road_edges.append(new_road_edge)

cropped_lanes = []
for lane in lanes:
    new_lane = []
    for i in range(lane.shape[0]):
        if lane[i,0] < x_min or lane[i,0] > x_max or lane[i,1] < y_min or lane[i,1] > y_max:
            continue
        new_lane.append(lane[i])
    if len(new_lane) > 0:
        new_lane = np.stack(new_lane)
        cropped_lanes.append(new_lane)

max_per_lane_node = 20

for edge in cropped_road_edges:
    edge = np.array(edge)
    # edge = edge[::5]
    plt.plot(edge[:,0],edge[:,1],c='red')

for lane in cropped_lanes:
    lane = np.array(lane)
    # lane = lane[::5]
    plt.plot(lane[:,0],lane[:,1],c='green')
#  - 29.623
#  - -4.65

def rotate(point, angle):
    """Rotates a point around the origin by the specified angle in radians."""
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, point)

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])

    # The relative positions of the vertices from the box center before rotation.
    relative_positions = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ]) * half_dims

    # Rotate each relative position and add the box center position.
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

# cars = np.load('/dssg/home/acct-giftxhy/giftxhy/yuxiwei/nerf-factory/data/waymo/segment-10247954040621004675_2180_000_2200_000_with_camera_labels/3d_boxes.npy', allow_pickle = True).item()

# vertices = generate_vertices(cars['0'])

# plt.plot(vertices[::2,0],vertices[::2,1])

# plt.scatter([29.623],[-4.65])
plt.savefig('/home/ubuntu/yuxiwei/debug/map.png')
import ipdb; ipdb.set_trace()
output = {"centerline":cropped_lanes,"boundary":cropped_road_edges}
import pickle 
with open('/dssg/home/acct-giftxhy/giftxhy/waymo_tfrecord/1.4.2/map_data.pkl', 'wb') as f:
    pickle.dump(output, f)

import ipdb; ipdb.set_trace()

    
def vis_map_debug(map, motion):
                from matplotlib import pyplot as plt
                cropped_road_edges = map['boundary']
                cropped_lanes = map['centerline']
                for edge in cropped_road_edges:
                    edge = np.array(edge)
                    # edge = edge[::5]
                    plt.plot(edge[:,0],edge[:,1],c='red')

                for lane in cropped_lanes:
                    lane = np.array(lane)
                    # lane = lane[::5]
                    plt.plot(lane[:,0],lane[:,1],c='green')

                if motion is not None:
                        
                        # lane = lane[::5]
                    plt.plot(motion[:,0],motion[:,1],c='blue')
                plt.savefig('/home/ubuntu/yuxiwei/debug/running_map.png')