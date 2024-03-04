from matplotlib import pyplot as plt
import numpy as np
import random
from chatsim.foreground.motion_tools.tools import transform_node_to_lane

def crop_map(input_map):
    center_point = (input_map['boundary'][:,:2] + input_map['boundary'][:,2:4])/2
    mask = (center_point[:,0] > 0)
    input_map['boundary'] = input_map['boundary'][mask]

    center_point = (input_map['centerline'][:,:2] + input_map['centerline'][:,2:4])/2
    mask = (center_point[:,0] > 0)
    input_map['centerline'] = input_map['centerline'][mask]
    return input_map



def conflict_check(centerline,index,current_vertices):
    thres = 4
    
    for i in range(current_vertices.shape[0]):
        point1 = (centerline[index][0:2]+centerline[index][2:4])/2
        point2 = current_vertices[i][:2]
        if np.sqrt(np.sum((np.array(point2) - np.array(point1))**2)) < thres:
            return False
    return True

def get_distance_from_point_to_line(point, line_point1, line_point2):
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

def get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading):
    cosangle = ego_lane_vec_heading.dot(cur_lane_vec_heading)/(np.linalg.norm(ego_lane_vec_heading) * np.linalg.norm(cur_lane_vec_heading))            
    angle = np.arccos(cosangle) * 180 / np.pi
    a1 = np.array([*ego_lane_vec_heading, 0])
    a2 = np.array([*cur_lane_vec_heading, 0])

    a3 = np.cross(a1, a2)

    if np.sign(a3[2]) < 0:
        angle = 360 - angle

    return angle

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

def vehicle_placement_specific(input_map,current_vertices,coord):
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    plt.cla()
    print(coord)
    print(centerline[:,0:2])
    for i in range(len(centerline)):
        lane_vec = centerline[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="green",linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)

    for i in range(len(boundary)):
        lane_vec = boundary[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]],color="red",linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black',s=1)
    
    center_coord = (centerline[:,0:2]+centerline[:,2:4])/2

    distance = np.linalg.norm(coord[None] - center_coord, ord=2,axis=-1)
    closest_distance_index = np.argmin(distance)

    lane_vec = centerline[closest_distance_index]

    x,y = coord[0], coord[1]
    xs,ys,xe,ye = lane_vec[0],lane_vec[1],lane_vec[2],lane_vec[3]
    theta = np.arctan2((xe-xs), (ye-ys))

    return (x,y,theta,xs,ys,xe,ye)

def vehicle_placement(input_map,current_vertices,direction,vehicle_mode,distance_constraint,distance_min_max,vehicle_size):

    # input_map = input_map.cpu().numpy()
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

    valid_lane_list = []
    ego_index = 0
    ego_dist = 999
    for i in range(centerline.shape[0]):
        valid_lane_list.append(i)
        center_coord = (centerline[i,0:2]+centerline[i,2:4])/2
        if np.linalg.norm(center_coord, ord=2) < ego_dist:
            ego_index = i
            ego_dist = np.linalg.norm(center_coord, ord=2)
    ego_lane_vec = centerline[ego_index]

    input_map = centerline
    result_list = []
    
    # parameter list
    vehicle_size_x = 2
    vehicle_size_y = 4.5
    distance_min_default = 0
    distance_max_default = 50


    mode = vehicle_mode

    if distance_constraint:
        distance_min = float(distance_min_max[0]) + 4
        distance_max = float(distance_min_max[1]) + 4

    l,w = vehicle_size_y/2, vehicle_size_x/2

    if mode == 'random':
        while True:
            cur_valid_lane_index_list = []
            for i in range(len(valid_lane_list)):
                center_coord = (input_map[valid_lane_list[i],0:2]+input_map[valid_lane_list[i],2:4])/2
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) >= distance_min and np.linalg.norm(center_coord, ord=2) <= distance_max:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                return (None,'No place to put cars')
                index = -1
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

    elif mode == 'front':
        while True:
            cur_valid_lane_index_list = []
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2]+cur_lane_vec[2:4])/2
                dist_to_lane_vec = get_distance_from_point_to_line(center_coord,ego_lane_vec[0:2],ego_lane_vec[2:4])
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and dist_to_lane_vec < 2 and center_coord[0]>0:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                return (None,'No place to put front cars')
                index = -1
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

    elif mode == 'left front':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = np.array([1.0,0.0]) #ego_lane_vec[2:4] - ego_lane_vec[0:2]
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2]+cur_lane_vec[2:4])/2
                dist_to_lane_vec = get_distance_from_point_to_line(center_coord,ego_lane_vec[0:2],ego_lane_vec[2:4])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading,cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and dist_to_lane_vec >= 1.5 and dist_to_lane_vec <= 10 and theta >= 3 and theta <= 60:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None,'No place to put left front cars')
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

    elif mode == 'right front':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0,0.0]) #ego_lane_vec[2:4] - ego_lane_vec[0:2]
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2]+cur_lane_vec[2:4])/2
                dist_to_lane_vec = get_distance_from_point_to_line(center_coord,ego_lane_vec[0:2],ego_lane_vec[2:4])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading,cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and dist_to_lane_vec >= 1.5 and dist_to_lane_vec <= 10 and theta >= 300 and theta <= 357:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None,'No place to put right front cars')
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

    elif mode == 'left':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0,0.0]) # ego_lane_vec[2:4] - ego_lane_vec[0:2]
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2]+cur_lane_vec[2:4])/2
                dist_to_lane_vec = get_distance_from_point_to_line(center_coord,ego_lane_vec[0:2],ego_lane_vec[2:4])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading,cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and dist_to_lane_vec >= 1.5 and dist_to_lane_vec <= 10 and theta > 75 and theta <= 105:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None,'No place to put cars on the left')
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

    elif mode == 'right':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0,0.0]) #ego_lane_vec[2:4] - ego_lane_vec[0:2]
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2]+cur_lane_vec[2:4])/2
                dist_to_lane_vec = get_distance_from_point_to_line(center_coord,ego_lane_vec[0:2],ego_lane_vec[2:4])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading,cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and dist_to_lane_vec >= 1.5 and dist_to_lane_vec <= 10 and theta > 255 and theta < 285:
                    if direction == 'close' and input_map[valid_lane_list[i],-1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i],-1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0,len(cur_valid_lane_index_list)-1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return(None,'No place to put cars on the right')
                break

            del valid_lane_list[random_lane_index]
            if conflict_check(centerline,index,current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break

        # if v == 0:
        #     lane_vec = ego_lane_vec
        # else:
    if index < 0:
        return(None,'No place to put cars')

    lane_vec = input_map[index]
    
    xs,ys,xe,ye = lane_vec[0],lane_vec[1],lane_vec[2],lane_vec[3]
    xc,yc = (xs+xe)/2, (ys+ye)/2
    theta = np.arctan2((xe-xs), (ye-ys))

    x1, y1 = xc - w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) + w * np.sin(theta)
    x2, y2 = xc + w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) - w * np.sin(theta)
    x3, y3 = xc + w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) - w * np.sin(theta)
    x4, y4 = xc - w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) + w * np.sin(theta)

    
    
    return  (xc,yc,theta,xs,ys,xe,ye)

if __name__ == '__main__':
    import pickle
    map_file = '/dssg/home/acct-umjpyb/umjpyb/ziwang/waymo_tfrecord/1.4.2/map_data.pkl'
    cars = np.load('/dssg/home/acct-umjpyb/umjpyb/ziwang/web/data/waymo/segment-10247954040621004675_2180_000_2200_000_with_camera_labels/3d_boxes.npy', allow_pickle = True).item()

    all_current_vertices = []
    for k in cars.keys():
        current_vertices = generate_vertices(cars[k])
        all_current_vertices.append(current_vertices)
    all_current_vertices = np.array(all_current_vertices)
    with open(map_file, 'rb') as f:
        map_data = pickle.load(f)

    transformed_map_data = transform_node_to_lane(map_data)
    vehicle_num = 1

    vehicle_mode_list = ['right front','left front']
    distance_constraint_list = [True,False]
    distance_min_max_list = [(8.0,10.0),'default']
    vehicle_size_list = ['default','default']

    result = vehicle_placement(transformed_map_data,all_current_vertices,vehicle_num,vehicle_mode_list=vehicle_mode_list,distance_constraint_list=distance_constraint_list,distance_min_max_list=distance_min_max_list,vehicle_size_list=vehicle_size_list)
    # result = vehicle_placement_specific(transformed_map_data,all_current_vertices,coord=np.array([30,-14.5]))

    print(result)

    # import ipdb; ipdb.set_trace()

    
    