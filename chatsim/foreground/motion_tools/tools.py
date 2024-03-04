from typing import List
from matplotlib import pyplot as plt
import numpy as np
import random
import openai

import ipdb
import os
import re
import ast
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline, make_interp_spline

def filter_forward_lane(input_map):
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    forward_index = (centerline[:,2:4] - centerline[:,0:2])[:,0] > 0
    filtered_centerline = centerline[forward_index,:]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output


def filter_right_lane(input_map,v=8):
    thres_min = -30
    thres_max = -5
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    right_index = (centerline[:,1] > thres_min) & (centerline[:,1] < thres_max)& ((centerline[:,3]-centerline[:,1]) <= 0)
    filtered_centerline = centerline[right_index,:]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_left_lane(input_map,v=8):
    thres_min = 5
    thres_max = 30
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    right_index = (centerline[:,1] > thres_min) & (centerline[:,1] < thres_max) & ((centerline[:,3]-centerline[:,1]) >= 0)
    filtered_centerline = centerline[right_index,:]

    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_right_lane_midpoint(input_map,v=8):
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    direction = centerline[:,2:4] - centerline[:,0:2]
    theta = np.arctan2(direction[:,1],direction[:,0]) / np.pi * 180
    right_index = (theta >= -120) & (theta <= 10)
    filtered_centerline = centerline[right_index,:]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_left_lane_midpoint(input_map,v=8):
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    direction = centerline[:,2:4] - centerline[:,0:2]
    theta = np.arctan2(direction[:,1],direction[:,0]) / np.pi * 180
    right_index = (theta <= 120) & (theta >= -10)
    filtered_centerline = centerline[right_index,:]

    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_front_lane(input_map):
    thres_max = 10
    thres_min = -10
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    front_index = ((centerline[:,1] > thres_min) & (centerline[:,1] < thres_max))
    filtered_centerline = centerline[front_index,:]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output


def filter_direction_lane(input_map, direction):
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    if direction == 'away':
        away_index = (centerline[:,-1] ==  1)
        filtered_centerline = centerline[away_index,:]

    else:
        close_index = (centerline[:,-1] ==  0)
        filtered_centerline = centerline[close_index,:]

    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output



def rotation_matrix_from_vector(v):
    # 计算角度
    angle = np.arctan2(v[1], v[0])
    
    # 创建旋转矩阵
    R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    return R

def transform_points_directly(points, source_vector, target_vector):
    # 获取源坐标系和目标坐标系的旋转矩阵
    R_source = rotation_matrix_from_vector(source_vector)
    R_target = rotation_matrix_from_vector(target_vector)
    
    # 计算从源坐标系到目标坐标系的直接转换矩阵
    R_direct = np.dot(R_target, np.linalg.inv(R_source))
    
    # 使用直接的转换矩阵进行点转换
    transformed_points = np.dot(points, R_direct.T)  # 注意转置操作，因为点是以行的形式给出的
    
    return transformed_points

def rot_and_trans_node(input_raw_map,current_pose):
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0,0.0])
    output_centerline = []
    output_boundary = []
    centerline = input_raw_map['centerline']
    boundary = input_raw_map['boundary']
    for line in centerline:
        line[:,:2] -= coordinate[None,...]
        line = transform_points_directly(line[:,:2],ego_vec,current_vec)
        output_centerline.append(line)

    for line in boundary:
        line[:,:2] -= coordinate[None,...]
        line = transform_points_directly(line[:,:2],ego_vec,current_vec)
        output_boundary.append(line)
    output = {}
    output['centerline'] = output_centerline
    output['boundary'] = output_boundary
    
    return output

        

def rot_and_trans(input_map,current_pose):
    # current_pose: (x,y,theta,xs,ys,xe,ye)
    centerline = input_map['centerline'].copy()
    boundary = input_map['boundary'].copy()
    output = {}
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0,0.0])

    centerline[:,0:2] -= coordinate
    centerline[:,2:4] -= coordinate
    boundary[:,0:2] -= coordinate
    boundary[:,2:4] -= coordinate

    centerline[:,0:2] = transform_points_directly(centerline[:,0:2],ego_vec,current_vec)
    centerline[:,2:4] = transform_points_directly(centerline[:,2:4],ego_vec,current_vec)

    boundary[:,0:2] = transform_points_directly(boundary[:,0:2],ego_vec,current_vec)
    boundary[:,2:4] = transform_points_directly(boundary[:,2:4],ego_vec,current_vec)


    output['centerline'] = centerline
    output['boundary'] = boundary
    return output

def rot_and_trans_bbox(input_bbox,current_pose):
    # current_pose: (x,y,theta,xs,ys,xe,ye)
    output_bbox = input_bbox.copy()
    if input_bbox.shape[0] == 0:
        return output_bbox
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0,0.0])

    output_bbox = output_bbox.reshape((-1,2))
    
    output_bbox[:,0:2] -= coordinate

    output_bbox[:,0:2] = transform_points_directly(output_bbox[:,0:2],ego_vec,current_vec)

    output_bbox = output_bbox.reshape((-1,4,2))
    return output_bbox

def inverse_rot_and_trans(input,current_pose):
    # current_pose: (x,y,theta,xs,ys,xe,ye)
    current_pose = np.array(current_pose).copy()
    input = np.array(input)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0,0.0])
    output = transform_points_directly(input,current_vec,ego_vec)
    output += coordinate
    return output

def crop_map(input_map):
    center_point = (input_map['boundary'][:,:2] + input_map['boundary'][:,2:4])/2
    mask = (center_point[:,0] > 0)
    input_map['boundary'] = input_map['boundary'][mask]

    center_point = (input_map['centerline'][:,:2] + input_map['centerline'][:,2:4])/2
    mask = (center_point[:,0] > 0)
    input_map['centerline'] = input_map['centerline'][mask]
    return input_map



def transform_node_to_lane(input_map,pre_transform=True):
    output_lane_map = {}
    edge_lanes = []
    for edge in input_map['boundary']:
        N = edge.shape[0]
        edge_lane = np.zeros((N,6))
        edge_lane[:,:2] = edge[:,:2]
        edge_lane[:-1,2:4] = edge[1:,:2]
        edge_lane[:,-2] = 0 # 0 for boundary and 1 for centerline
        edge_lane = edge_lane[:-1]
        edge_lanes.append(edge_lane)
    centerline_lanes = []
    for i, centerline in enumerate(input_map['centerline']):
        if pre_transform:
            centerline = centerline[centerline[:,0] > 0]
        N = centerline.shape[0]
        if N > 0:
            centerline_lane = np.zeros((N,6))
            centerline_lane[:,:2] = centerline[:,:2]
            centerline_lane[:-1,2:4] = centerline[1:,:2]
            centerline_lane[:,-2] = 1
            if (np.linalg.norm(centerline[-1]) - np.linalg.norm(centerline[0])) > 0:
                centerline_lane[:,-1] = 1 # drive away
            else:
                centerline_lane[:,-1] = 0 # drive close
            centerline_lane = centerline_lane[:-1]
            # if i != 5:
            centerline_lanes.append(centerline_lane)
        # print(i,centerline_lane[::5])
    # time.sleep(1000)
    # print('warning!transform_node_to_lane')
    output_lane_map['boundary'] = np.concatenate(edge_lanes,axis=0)
    output_lane_map['centerline'] = np.concatenate(centerline_lanes,axis=0)
    if pre_transform:
        output_lane_map = crop_map(output_lane_map)
    return output_lane_map
    
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

def extract_python_code(s):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, s, re.DOTALL)
    return match.group(1).strip()

def chat(system, user_assistant):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"
    system_msg = [{"role": "system", "content": system}]
    user_assistant_msgs = [
        {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
        for i in range(len(user_assistant))]

    msgs = system_msg + user_assistant_msgs
    response = openai.ChatCompletion.create(model="gpt-4",
                                            messages=msgs)
    # status_code = response["choices"][0]["finish_reason"]
    # assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]

def hermite_spline(P0, P1, T0, T1, num_points=100):
    # 生成参数t
    t = np.linspace(0, 1, num_points)
    # 计算Hermite基函数
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    spline_points = (h00[:, None] * P0 + h10[:, None] * T0 +
                    h01[:, None] * P1 + h11[:, None] * T1)
    return spline_points

def quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def hermite_spline_once(P0, P1, T0, T1, num_points=100):
    spline_points = hermite_spline(P0, P1, T0, T1, num_points)
    
    # file_name='work_dirs/vis_gpt_traj/traj_'+ str(agent) + '_' + str(time) + '.png'
    # if post_transform[0]:
    #     visualize(inverse_rot_and_trans(spline_points,post_transform[1]),file_name=file_name, input_map=input_map,obj=obj)
    # else:
    #     visualize(spline_points,file_name=file_name, input_map=input_map,obj=obj)

    return spline_points

def hermite_spline_twice(P0, P1, PM, T0, T1, TM, num_points=100):
    spline_points_1 = hermite_spline(P0, PM, T0, TM, num_points)
    spline_points_2 = hermite_spline(PM, P1, TM, T1, num_points)
    spline_points = np.vstack((spline_points_1, spline_points_2))
    
    # file_name='work_dirs/vis_gpt_traj/traj_'+ str(agent) + '_' + str(time) + '.png'
    # if post_transform[0]:
    #     visualize(inverse_rot_and_trans(spline_points,post_transform[1]),file_name=file_name, input_map=input_map,obj=obj)
    # else:
    #     visualize(spline_points,file_name=file_name, input_map=input_map,obj=obj)

    return spline_points

def hermite_spline_third(P0, P1, PM, PMM1, PMM2,T0, T1, TM, TMM1,TMM2, num_points=100,time=1,input_map=None,post_transform=(False,None),obj=None):
    spline_points_1 = hermite_spline(P0, PMM1, T0, TMM1, num_points)
    spline_points_2 = hermite_spline(PMM1, PM, TMM1, TM, num_points)
    spline_points_3 = hermite_spline(PM, PMM2, TM, TMM2, num_points)
    spline_points_4 = hermite_spline(PMM2, P1, TMM2, T1, num_points)

    spline_points = np.vstack((spline_points_1, spline_points_2,spline_points_3,spline_points_4))
    # spline_points = np.vstack((spline_points_3))

    # file_name='work_dirs/vis_gpt_traj/traj_'+ str(agent) + '_' + str(time) + '.png'
    # if post_transform[0]:
    #     visualize(inverse_rot_and_trans(spline_points,post_transform[1]),file_name=file_name, input_map=input_map,obj=obj)
    # else:
    #     visualize(spline_points,file_name=file_name, input_map=input_map,obj=obj)

    return spline_points

def cubic_bezier(p0, p1, p2, p3, t):

    return ((1-t)**3 * p0 +
            3 * (1-t)**2 * t * p1 +
            3 * (1-t) * t**2 * p2 +
            t**3 * p3)

def compute_bezier_points(p0, p1, p2, p3, num_points=100):

    return np.array([cubic_bezier(p0, p1, p2, p3, t) for t in np.linspace(0, 1, num_points)])

def transform_gpt_to_trajectory(answer,agent,time,input_map=None,post_transform=(False,None),obj=None):
    python_file = 'work_dirs/created_python_file/traj_'+ str(time) +'.py'
    if os.path.exists(python_file):
        os.remove(python_file)
    with open(python_file,'w') as f:
        f.write(extract_python_code(answer))

    python_command = "python " +  python_file
    result = os.popen(python_command)  
    res = result.read()  
    coordinates = ast.literal_eval(res) 


    return coordinates

def transform_coord_to_trajectory(answer,agent,time,input_map=None,post_transform=(False,None),obj=None):
    python_file = 'work_dirs/created_python_file/traj_'+ str(time) +'.py'
    if os.path.exists(python_file):
        os.remove(python_file)
    with open(python_file,'w') as f:
        f.write(extract_python_code(answer))

    python_command = "python " +  python_file
    result = os.popen(python_command)  
    res = result.read()  
    coordinates = ast.literal_eval(res) 
    

    return coordinates

def project_polygon_onto_axis(vertices, axis):
    
    min_val = max_val = np.dot(vertices[0], axis)
    for vertex in vertices[1:]:
        projection = np.dot(vertex, axis)
        min_val = min(min_val, projection)
        max_val = max(max_val, projection)
    return min_val, max_val

def is_projection_overlap(proj1, proj2):
    
    return max(proj1[0], proj2[0]) <= min(proj1[1], proj2[1])

def is_rectangles_overlap(rect1, rect2):
    
    for i in range(4):
        
        edge = rect1[i] - rect1[(i + 1) % 4]
        
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        
        
        proj1 = project_polygon_onto_axis(rect1, axis)
        proj2 = project_polygon_onto_axis(rect2, axis)
        
        
        if not is_projection_overlap(proj1, proj2):
            return False
    
    
    for i in range(4):
        edge = rect2[i] - rect2[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        
        proj1 = project_polygon_onto_axis(rect1, axis)
        proj2 = project_polygon_onto_axis(rect2, axis)
        
        if not is_projection_overlap(proj1, proj2):
            return False
    
    return True

def calculate_car_corners(trajectory, car_length=4.5, car_width=2):
    
    T = trajectory.shape[0]
    corners_trajectory = np.zeros((T, 4, 2))
    
    for i in range(1, T):
        
        direction = trajectory[i] - trajectory[i - 1]
        direction /= np.linalg.norm(direction)
        
        
        perpendicular = np.array([-direction[1], direction[0]])
        
        
        front = 0.5 * car_length * direction
        back = -0.5 * car_length * direction
        left = 0.5 * car_width * perpendicular
        right = -0.5 * car_width * perpendicular
        
        
        corners_trajectory[i, 0] = trajectory[i] + front + left
        corners_trajectory[i, 1] = trajectory[i] + front + right
        corners_trajectory[i, 2] = trajectory[i] + back + right
        corners_trajectory[i, 3] = trajectory[i] + back + left
        
    corners_trajectory[0] = corners_trajectory[1]
    
    return corners_trajectory

def is_tailgating(trajectory1,trajectory2):
    threshold = 0.2
    speed1 = np.diff(trajectory1, axis=0)
    speed2 = np.diff(trajectory2, axis=0)

    direction = trajectory2[t] - trajectory1[t]
    speed_direction1 = speed1[t-1] / np.linalg.norm(speed1[t-1])
    speed_direction2 = speed2[t-1] / np.linalg.norm(speed2[t-1])

    angle1 = np.arccos(np.clip(np.dot(direction, speed_direction1), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(-direction, speed_direction2), -1.0, 1.0))
    
    if angle1 < threshold and angle2 < threshold:
        return True  
    else:
        return False

def accerlate(trajectory, speed_increase=1.1):
    
    speeds = np.diff(trajectory, axis=0)
    
    
    speeds *= speed_increase
    
    
    new_trajectory = np.cumsum(np.vstack([trajectory[0], speeds]), axis=0)
    return new_trajectory

def deaccerlate(trajectory, speed_decrease=1.1):
    
    speeds = np.diff(trajectory, axis=0)
    
    
    speeds *= speed_decrease
    
   
    new_trajectory = np.cumsum(np.vstack([trajectory[0], speeds]), axis=0)
    return new_trajectory

def calculate_speed_increase(front_car_traj, rear_car_traj, safe_distance=7):
    
    distances = np.linalg.norm(rear_car_traj - front_car_traj, axis=1)
    
    
    front_car_speeds = np.linalg.norm(np.diff(front_car_traj, axis=0), axis=1)
    rear_car_speeds = np.linalg.norm(np.diff(rear_car_traj, axis=0), axis=1)
    
    
    relative_speeds = rear_car_speeds - front_car_speeds
    
    
    time_to_collision = (distances[1:] - safe_distance) / relative_speeds
    
    
    min_time_to_collision = np.min(time_to_collision[relative_speeds > 0])
    
    
    if np.any(distances > safe_distance) or min_time_to_collision > 0:
        return 1.0
    
    
    speed_increase = 1 + (safe_distance - distances[1:]) / (relative_speeds * min_time_to_collision)
    
    
    return np.max(speed_increase)

def check_collision_and_revise_waste(all_trajectory):
    all_trajectory = np.array(all_trajectory)
    N, T = all_trajectory.shape[0], all_trajectory.shape[1]
    car_length = 4.5
    car_width = 2.0
    safe_distance = 7
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(all_trajectory[n], car_length, car_width)


    for j in range(1,N):
        for t in range(T):
            if is_rectangles_overlap(all_corners_trajectory[0,t],all_corners_trajectory[j,t]):
                                 
                trajectory1 = all_trajectory[0]
                trajectory2 = all_trajectory[j]
                speed1 = np.linalg.norm(np.diff(trajectory1, axis=0))
                speed2 = np.linalg.norm(np.diff(trajectory2, axis=0))
                if is_tailgating(trajectory1,trajectory2) and speed1[t-1] < speed2[t-1]:
                    all_trajectory[0] = accerlate(all_trajectory[0],calculate_speed_increase(trajectory1,trajectory2))
                    break
                else: 
                    collision_point = trajectory1[t]
                    for t_safe in range(t+1,T):
                        if np.linalg.norm(trajectory2[t_safe] - collision_point) > safe_distance:
                            break
                    if t == T-1:
                        all_trajectory[0] = deaccerlate_to_zero(all_trajectory[0])
                    else:
                        all_trajectory[0] = deaccerlate(all_trajectory[0],t/t_safe)
    
    return all_trajectory

def check_collision_and_revise(all_trajectory):
    all_trajectory = np.array(all_trajectory)
    N, T = all_trajectory.shape[0], all_trajectory.shape[1]
    car_length = 4.5
    car_width = 2.0
    safe_distance = 7
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(all_trajectory[n], car_length, car_width)


    for j in range(1,N):
        for t in range(T):
            if is_rectangles_overlap(all_corners_trajectory[0,t],all_corners_trajectory[j,t]):
                                 
                trajectory1 = all_trajectory[0]
                trajectory2 = all_trajectory[j]
                speed1 = np.linalg.norm(np.diff(trajectory1, axis=0))
                speed2 = np.linalg.norm(np.diff(trajectory2, axis=0))
                if is_tailgating(trajectory1,trajectory2) and speed1[t-1] < speed2[t-1]: 
                    return 'accerlate'
                else: 
                    return 'deaccerlate'
    
    return 'no revise'

def visualize(input,file_name,input_map=None,multi_traj=False,obj=None):
    plt.cla()
    plt.figure(figsize=(10, 6))
    plt.xlabel("X (Front of the car) [meters]")
    plt.ylabel("Y (Right of the car) [meters]")
    
    if multi_traj:
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i/len(input)) for i in range(len(input))]
        for i in range(len(input)):
            if input[i][0] is not None:
                x_vals, y_vals = input[i][:,0],input[i][:,1]
                plt.plot(x_vals, y_vals, 'b-', color=colors[i], label=f"Trajectory{i}",lw=5)
                
    else:
        x_vals, y_vals = zip(*input)
        plt.cla()
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', label="Trajectory",lw=5)
        
        plt.xlabel("X (Front of the car) [meters]")
        plt.ylabel("Y (Right of the car) [meters]")

    if input_map is not None:
        centerline = input_map['centerline']
        boundary = input_map['boundary']
        for i in range(len(centerline)):
            lane_vec = centerline[i]
            
            plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="green",linewidth=1)
            plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)

        for i in range(len(boundary)):
            lane_vec = boundary[i]
            plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="red",linewidth=1)
            plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)

    if obj is not None:
        for i in range(len(obj)):
            plt.fill(obj[i,:,0], obj[i,:,1], 'r', fill=True)

    plt.grid(True)
    plt.legend()
    plt.savefig(file_name)
    plt.show()
    return

def visualize_placement(input_position,input_map):
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    plt.cla()

    for i in range(len(centerline)):
        
        lane_vec = centerline[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="green",linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)

    for i in range(len(boundary)):
        lane_vec = boundary[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="red",linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)

    vehicle_size_x = 2
    vehicle_size_y = 4.5
    l,w = vehicle_size_y/2, vehicle_size_x/2

    for item in input_position:
        if item[0] is not None:
            (xc,yc,theta,xs,ys,xe,ye) = item
            theta = np.arctan2((xe-xs), (ye-ys))
            x1, y1 = xc - w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) + w * np.sin(theta)
            x2, y2 = xc + w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) - w * np.sin(theta)
            x3, y3 = xc + w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) - w * np.sin(theta)
            x4, y4 = xc - w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) + w * np.sin(theta)

            

def find_closest_centerline(transformed_map_data,current_destination):
    thres = 0.3
    current_destination = np.array(current_destination)
    centerlines = transformed_map_data['centerline']
    centernodes = (centerlines[:,0:2] + centerlines[:,2:4]) / 2
    distances = np.linalg.norm(current_destination[None] - centernodes,axis=-1,ord=2)
    closest_centerline_index = np.argmin(distances)
    if distances[closest_centerline_index] < thres:
        return True, centerlines[closest_centerline_index]
    else:
        return False, centerlines[closest_centerline_index]

def savitzky_golay_2d(trajectory, window_size=5, polynomial_order=2):
    x_smooth = savgol_filter(trajectory[:, 0], window_size, polynomial_order)
    y_smooth = savgol_filter(trajectory[:, 1], window_size, polynomial_order)
    return np.column_stack((x_smooth, y_smooth))

def bspline_smooth(points, degree=3):
    if points is not None:
        points = np.array(points)
        x = np.linspace(0, 1, len(points))
        bspline = BSpline(x, points, k=degree)
        x_new = np.linspace(0, 1, len(points))
        points_smoothed = bspline(x_new)
        return points_smoothed
    return None

